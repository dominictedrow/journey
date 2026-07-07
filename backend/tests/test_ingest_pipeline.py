"""Full component sweep of the data-ingest pipeline (decoders -> normalize -> data_api ->
history seam -> indexer -> live tail). Exploratory + regression; each test observes one
component's real behavior so the seam fix can be reasoned about against the whole flow.
"""

from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest
from sqlalchemy import select

from app.blockchain import listener as listener_mod
from app.blockchain.decoders import (
    ORDER_FILLED_TOPIC,
    address_topic,
    decode_order_filled,
)
from app.blockchain.indexer import get_checkpoint, set_checkpoint
from app.blockchain.normalize import (
    compute_positions,
    fill_to_trade,
    insert_trades,
    load_token_market_map,
)
from app.database.models import Market, Position, Trade, Wallet
from app.ingest import data_api as data_api_mod
from app.ingest import history as history_mod
from app.ingest.data_api import _page_window, get_holders, get_positions, get_trades, iter_trades
from app.ingest.history import (
    _resolve_handoff,
    _with_synthetic_log_index,
    backfill_wallet_history,
    data_api_trade_to_row,
)

WALLET = "0x" + "cc" * 20
OTHER = "0x" + "dd" * 20
TS = datetime(2026, 6, 1, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def raw_log(
    *,
    maker: str,
    maker_asset_id: int,
    taker_asset_id: int,
    maker_amount: int,
    taker_amount: int,
    tx: str = "0x" + "ab" * 32,
    log_index: int = 3,
    block: int = 100,
    topic0: str = ORDER_FILLED_TOPIC,
) -> dict:
    def word(n: int) -> bytes:
        return n.to_bytes(32)

    data = (
        word(maker_asset_id)
        + word(taker_asset_id)
        + word(maker_amount)
        + word(taker_amount)
        + word(0)  # fee
    )
    return {
        "topics": [
            topic0,
            "0x" + "00" * 32,  # orderHash (unused)
            address_topic(maker),
            address_topic("0x" + "ee" * 20),  # taker
        ],
        "data": "0x" + data.hex(),
        "transactionHash": tx,
        "logIndex": log_index,
        "blockNumber": block,
    }


class FakePool:
    """Stand-in for RpcPool: fixed head + block->timestamp map."""

    def __init__(self, head: int, ts_map: dict[int, int], logs: list[dict] | None = None) -> None:
        self.head = head
        self.ts_map = ts_map
        self.logs = logs or []

    async def get_block_number(self) -> int:
        return self.head

    async def get_block(self, number: int):  # noqa: ANN201
        return {"timestamp": self.ts_map[number]}

    async def get_logs(self, params):  # noqa: ANN001, ANN201
        return self.logs


def mock_httpx(monkeypatch, handler) -> None:
    """Route httpx.AsyncClient through a MockTransport (capture the real class first to
    avoid recursing into the patched name)."""
    real_client = httpx.AsyncClient

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return real_client(**kwargs, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(data_api_mod.httpx, "AsyncClient", factory)


# --------------------------------------------------------------------------------------
# 1. decoders
# --------------------------------------------------------------------------------------
def test_decode_order_filled_buy_and_sell() -> None:
    buy = decode_order_filled(
        raw_log(
            maker=WALLET, maker_asset_id=0, taker_asset_id=777, maker_amount=50, taker_amount=100
        )
    )
    assert buy is not None
    assert buy.maker == WALLET
    assert buy.maker_asset_id == 0 and buy.taker_asset_id == 777
    assert buy.maker_amount == 50 and buy.taker_amount == 100

    sell = decode_order_filled(
        raw_log(
            maker=WALLET, maker_asset_id=777, taker_asset_id=0, maker_amount=100, taker_amount=60
        )
    )
    assert sell is not None and sell.maker_asset_id == 777


def test_decode_rejects_foreign_topic() -> None:
    assert (
        decode_order_filled(
            raw_log(
                maker=WALLET,
                maker_asset_id=0,
                taker_asset_id=1,
                maker_amount=1,
                taker_amount=1,
                topic0="0x" + "12" * 32,
            )
        )
        is None
    )


# --------------------------------------------------------------------------------------
# 2. normalize.fill_to_trade
# --------------------------------------------------------------------------------------
def test_fill_to_trade_buy_and_sell_shape() -> None:
    fill = decode_order_filled(
        raw_log(
            maker=WALLET,
            maker_asset_id=0,
            taker_asset_id=777,
            maker_amount=50_000_000,
            taker_amount=100_000_000,
        )  # 50 USDC -> 100 tokens
    )
    row = fill_to_trade(fill, TS, {"777": "market-1"})
    assert row["side"] == "BUY"
    assert row["token_id"] == "777"
    assert row["size"] == Decimal("100")
    assert row["usdc_amount"] == Decimal("50")
    assert row["price"] == Decimal("0.5")
    assert row["market_id"] == "market-1"
    assert row["source"] == "chain"

    sell_fill = decode_order_filled(
        raw_log(
            maker=WALLET,
            maker_asset_id=777,
            taker_asset_id=0,
            maker_amount=100_000_000,
            taker_amount=70_000_000,
        )  # 100 tokens -> 70 USDC
    )
    sell = fill_to_trade(sell_fill, TS, {})
    assert sell["side"] == "SELL" and sell["price"] == Decimal("0.7")
    assert sell["market_id"] is None  # token not in map -> null, not a crash


# --------------------------------------------------------------------------------------
# 3. normalize.insert_trades — idempotency, batching, dedup
# --------------------------------------------------------------------------------------
def _row(
    log_index: int,
    source: str,
    block: int | None,
    *,
    wallet: str = WALLET,
    tx: str = "0x" + "ab" * 32,
    size: str = "100",
    ts: datetime = TS,
) -> dict:
    return {
        "tx_hash": tx,
        "log_index": log_index,
        "wallet_address": wallet,
        "market_id": None,
        "token_id": "t1",
        "side": "BUY",
        "price": Decimal("0.50"),
        "size": Decimal(size),
        "usdc_amount": Decimal("50"),
        "block_number": block,
        "timestamp": ts,
        "source": source,
    }


async def test_insert_trades_idempotent_and_batches_over_limit(session) -> None:
    # 2500 distinct rows crosses the 1000-row batch boundary (bind-param safety).
    rows = [_row(i, "data_api", None, tx="0x" + f"{i:064x}") for i in range(2500)]
    assert await insert_trades(session, rows) == 2500
    await session.commit()
    # re-insert same rows -> all deduped
    assert await insert_trades(session, rows) == 0
    await session.commit()
    assert (
        await session.scalar(select(Trade).where(Trade.tx_hash == "0x" + f"{7:064x}")) is not None
    )


# --------------------------------------------------------------------------------------
# 4. normalize.compute_positions — data_api (null block) interleave ordering
# --------------------------------------------------------------------------------------
def test_compute_positions_orders_by_timestamp_across_sources() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = datetime(2026, 1, 2, tzinfo=UTC)
    trades = [
        Trade(
            tx_hash="0x1",
            log_index=0,
            wallet_address=WALLET,
            token_id="t1",
            side="BUY",
            price=Decimal("0.40"),
            size=Decimal("100"),
            block_number=None,
            timestamp=t0,
        ),  # data_api
        Trade(
            tx_hash="0x2",
            log_index=5,
            wallet_address=WALLET,
            token_id="t1",
            side="SELL",
            price=Decimal("0.60"),
            size=Decimal("100"),
            block_number=200,
            timestamp=t1,
        ),  # chain
    ]
    pos = compute_positions(trades)[0]
    assert pos["net_size"] == 0
    assert pos["realized_pnl"] == Decimal("20")  # (0.60-0.40)*100 -> BUY seen before SELL


# --------------------------------------------------------------------------------------
# 5. normalize.load_token_market_map + rebuild_positions (DB)
# --------------------------------------------------------------------------------------
async def test_load_token_market_map(session) -> None:
    session.add(Market(id="m1", question="q", token_ids=["111", "222"]))
    session.add(Market(id="m2", question="q2", token_ids=None))
    await session.commit()
    mapping = await load_token_market_map(session)
    assert mapping == {"111": "m1", "222": "m1"}


# --------------------------------------------------------------------------------------
# 6. data_api._page_window — offset cap, graceful 400, short-page stop
# --------------------------------------------------------------------------------------
async def test_page_window_stops_at_offset_cap(monkeypatch) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", "0"))
        calls.append(offset)
        return httpx.Response(200, json=[{"i": offset}] * 1000)  # always full page

    mock_httpx(monkeypatch, handler)
    async with httpx.AsyncClient(base_url="http://x") as client:
        out = await _page_window(client, "/trades", {"user": "x"})
    assert calls == [0, 1000, 2000, 3000]  # stops at MAX_OFFSET, never requests 4000
    assert len(out) == 4000


async def test_page_window_graceful_400(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params.get("offset", "0"))
        if offset >= 2000:
            return httpx.Response(400, json={"error": "cap"})
        return httpx.Response(200, json=[{"i": offset}] * 1000)

    mock_httpx(monkeypatch, handler)
    async with httpx.AsyncClient(base_url="http://x") as client:
        out = await _page_window(client, "/trades", {"user": "x"})
    assert len(out) == 2000  # 400 at offset 2000 -> return what we have, no raise


async def test_page_window_first_request_400_raises(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad"})

    mock_httpx(monkeypatch, handler)
    with pytest.raises(httpx.HTTPStatusError):
        async with httpx.AsyncClient(base_url="http://x") as client:
            await _page_window(client, "/trades", {"user": "x"})


# --------------------------------------------------------------------------------------
# 7. data_api.get_trades — descending-window walk beyond the offset cap, no dup/loss
# --------------------------------------------------------------------------------------
def _make_trades_dataset(n: int) -> list[dict]:
    # distinct descending timestamps n..1, newest-first
    return [
        {
            "transactionHash": "0x" + f"{ts:064x}",
            "timestamp": ts,
            "asset": "t1",
            "side": "BUY",
            "price": "0.5",
            "size": "1",
            "proxyWallet": WALLET,
        }
        for ts in range(n, 0, -1)
    ]


def _trades_api_handler(dataset: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        offset = int(params.get("offset", "0"))
        limit = int(params.get("limit", "1000"))
        if offset > 3000:
            return httpx.Response(400, json={"error": "cap"})
        rows = dataset
        end = params.get("end")
        if end is not None:
            rows = [r for r in rows if r["timestamp"] <= int(end)]
        return httpx.Response(200, json=rows[offset : offset + limit])

    return handler


async def test_get_trades_windows_beyond_offset_cap(monkeypatch) -> None:
    dataset = _make_trades_dataset(5000)  # > MAX_OFFSET+PAGE_SIZE, distinct timestamps
    mock_httpx(monkeypatch, _trades_api_handler(dataset))
    out = await get_trades(WALLET)
    got = [t["timestamp"] for t in out]
    assert len(got) == 5000
    assert len(set(got)) == 5000  # no duplicates across windows
    assert set(got) == set(range(1, 5001))  # no gaps


async def test_iter_trades_yields_incrementally_bounded_memory(monkeypatch) -> None:
    """iter_trades must be a true generator: consuming N windows never touches window N+1, so
    peak memory for a backfill is one window (~4000 fills), not the wallet's whole history."""
    dataset = _make_trades_dataset(9000)  # spans 3 full offset-cap windows
    mock_httpx(monkeypatch, _trades_api_handler(dataset))
    gen = iter_trades(WALLET)
    first = await gen.__anext__()
    assert len(first) == 4000  # one full offset-cap window (0,1000,2000,3000 pages)
    assert min(t["timestamp"] for t in first) == 5001  # newest-first: 9000..5001
    second = await gen.__anext__()
    assert len(second) == 4000
    assert set(t["timestamp"] for t in first).isdisjoint(t["timestamp"] for t in second)
    await gen.aclose()


async def test_iter_trades_end_cursor_skips_newer_trades_at_the_source(monkeypatch) -> None:
    """Passing `end` must be honored server-side (the API call itself), not just filtered after
    the fact — this is what lets a backfill avoid fetching data the live tail already owns."""
    dataset = _make_trades_dataset(100)
    requested_ends: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_ends.append(request.url.params.get("end"))
        return _trades_api_handler(dataset)(request)

    mock_httpx(monkeypatch, handler)
    out = [t async for window in iter_trades(WALLET, end=50) for t in window]
    assert requested_ends[0] == "50"  # first request already scoped to the cutoff
    assert all(t["timestamp"] <= 50 for t in out)
    assert len(out) == 50


# --------------------------------------------------------------------------------------
# 8. data_api.get_positions / get_holders
# --------------------------------------------------------------------------------------
async def test_get_positions_and_holders(monkeypatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "/positions" in request.url.path:
            return httpx.Response(200, json=[{"asset": "t1", "size": "5"}])
        if "/holders" in request.url.path:
            return httpx.Response(200, json=[{"token": "t1", "holders": [{"proxyWallet": OTHER}]}])
        return httpx.Response(404)

    mock_httpx(monkeypatch, handler)
    assert (await get_positions(WALLET))[0]["asset"] == "t1"
    holders = await get_holders(["cond1", "cond2"], limit=50)
    assert holders[0]["holders"][0]["proxyWallet"] == OTHER


# --------------------------------------------------------------------------------------
# 9. history._with_synthetic_log_index — deterministic per-tx ordinals
# --------------------------------------------------------------------------------------
def test_synthetic_log_index_is_deterministic_per_tx() -> None:
    trades = [
        {"transactionHash": "0xA", "asset": "t1", "side": "BUY", "price": "0.5", "size": "2"},
        {"transactionHash": "0xA", "asset": "t1", "side": "BUY", "price": "0.5", "size": "1"},
        {"transactionHash": "0xB", "asset": "t2", "side": "SELL", "price": "0.9", "size": "3"},
    ]
    out1 = _with_synthetic_log_index(trades)
    out2 = _with_synthetic_log_index(list(reversed(trades)))
    # same (tx -> {size: index}) mapping regardless of input order
    idx1 = {(r["transactionHash"], r["size"]): i for r, i in out1}
    idx2 = {(r["transactionHash"], r["size"]): i for r, i in out2}
    assert idx1 == idx2
    assert idx1[("0xA", "1")] == 0 and idx1[("0xA", "2")] == 1  # sorted by size within tx
    assert idx1[("0xB", "3")] == 0


def test_data_api_trade_to_row_shape() -> None:
    trade = {
        "transactionHash": "0xabc",
        "proxyWallet": WALLET.upper(),
        "asset": "555",
        "side": "buy",
        "price": "0.33",
        "size": "12",
        "timestamp": 1_700_000_000,
    }
    row = data_api_trade_to_row(trade, 2, {"555": "mkt"})
    assert row["wallet_address"] == WALLET  # lowercased
    assert row["side"] == "BUY" and row["log_index"] == 2
    assert row["block_number"] is None and row["source"] == "data_api"
    assert row["market_id"] == "mkt"
    assert row["usdc_amount"] == Decimal("12") * Decimal("0.33")


# --------------------------------------------------------------------------------------
# 10. history._resolve_handoff — tracked / analyse-only / frozen
# --------------------------------------------------------------------------------------
async def test_resolve_handoff_tracked_first_run_seeds_live_cursor(session) -> None:
    pool = FakePool(head=1000, ts_map={})  # head-confirmations handled inside
    h = await _resolve_handoff(session, pool, WALLET, tracked=True)
    # head = 1000 - chain_confirmations(5) = 995; live cursor seeded to that
    assert h == 995
    assert await get_checkpoint(session, listener_mod.LIVE_STREAM) == 995
    assert await get_checkpoint(session, f"history:{WALLET}") == 995


async def test_resolve_handoff_uses_existing_live_cursor(session) -> None:
    await set_checkpoint(session, listener_mod.LIVE_STREAM, 500)
    await session.commit()
    pool = FakePool(head=1000, ts_map={})
    h = await _resolve_handoff(session, pool, WALLET, tracked=True)
    assert h == 500  # frozen at the live tail's current cursor, not head


async def test_resolve_handoff_frozen_write_once(session) -> None:
    await set_checkpoint(session, f"history:{WALLET}", 42)
    await session.commit()
    pool = FakePool(head=9999, ts_map={})
    assert await _resolve_handoff(session, pool, WALLET, tracked=False) == 42  # never re-derived


async def test_resolve_handoff_analyse_only_takes_head(session) -> None:
    pool = FakePool(head=1000, ts_map={})
    h = await _resolve_handoff(session, pool, OTHER, tracked=False)
    assert h == 995  # full history to head; no live cursor written
    assert await get_checkpoint(session, listener_mod.LIVE_STREAM) is None


# --------------------------------------------------------------------------------------
# 11. history.backfill_wallet_history — full flow, idempotent
# --------------------------------------------------------------------------------------
def _wire_backfill(monkeypatch, session, pool: FakePool, trades: list[dict]) -> None:
    """Point history.py at our fake pool, canned Data-API trades, and the test session.

    Mirrors the real iter_trades contract: honors the `end` cursor server-side (history.py no
    longer filters client-side) and yields in windows so window-count/order behavior matches
    production, just without real HTTP.
    """
    monkeypatch.setattr(history_mod, "RpcPool", lambda urls: pool)

    async def fake_iter_trades(address: str, end: int | None = None, window_size: int = 4):
        pool_ = [t for t in trades if end is None or int(t["timestamp"]) <= end]
        pool_.sort(key=lambda t: -int(t["timestamp"]))  # newest-first, like the real API
        for i in range(0, len(pool_), window_size):
            yield pool_[i : i + window_size]

    monkeypatch.setattr(history_mod, "iter_trades", fake_iter_trades)

    class _CtxFactory:
        def __call__(self):
            return self

        async def __aenter__(self):
            return session

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(history_mod, "async_session_factory", _CtxFactory())


def _dapi_trade(tx: str, ts: int, size: str = "100") -> dict:
    return {
        "transactionHash": tx,
        "proxyWallet": WALLET,
        "asset": "t1",
        "side": "BUY",
        "price": "0.5",
        "size": size,
        "timestamp": ts,
    }


async def test_backfill_wallet_history_idempotent(session, monkeypatch) -> None:
    session.add(Wallet(address=WALLET, first_seen=TS, last_active=TS, is_tracked=False))
    await session.commit()
    handoff = 995
    pool = FakePool(head=1000, ts_map={handoff: 1_700_000_100})
    trades = [_dapi_trade("0x" + f"{i:064x}", 1_700_000_000 + i) for i in range(10)]
    _wire_backfill(monkeypatch, session, pool, trades)

    first = await backfill_wallet_history(WALLET)
    assert first == 10
    second = await backfill_wallet_history(WALLET)  # frozen handoff -> no new rows
    assert second == 0
    pos = (await session.scalars(select(Position).where(Position.wallet_address == WALLET))).all()
    assert pos[0].net_size == Decimal("1000")  # 10 buys * 100, counted once


async def test_backfill_respects_cutoff_timestamp(session, monkeypatch) -> None:
    """Trades newer than timestamp(H) belong to the live tail and must be skipped."""
    session.add(Wallet(address=WALLET, first_seen=TS, last_active=TS, is_tracked=False))
    await session.commit()
    pool = FakePool(head=1000, ts_map={995: 1_700_000_005})  # cutoff ts = ...005
    trades = [_dapi_trade("0x" + f"{i:064x}", 1_700_000_000 + i) for i in range(10)]  # ts 000..009
    _wire_backfill(monkeypatch, session, pool, trades)
    inserted = await backfill_wallet_history(WALLET)
    assert inserted == 6  # ts 000..005 kept; 006..009 are past the handoff


# --------------------------------------------------------------------------------------
# 12. THE SEAM — live tail already ingested a block the backfill re-covers
# --------------------------------------------------------------------------------------
async def test_seam_no_double_count_when_live_ingested_first(session, monkeypatch) -> None:
    """A tracked wallet whose fill the live tail already stored (source='chain') must not be
    re-counted when the Data-API backfill covers the same block."""
    session.add(Wallet(address=WALLET, first_seen=TS, last_active=TS, is_tracked=True))
    # live tail already advanced to block 995 and stored this wallet's fill there
    await set_checkpoint(session, listener_mod.LIVE_STREAM, 995)
    chain_ts = 1_700_000_003
    await insert_trades(
        session,
        [
            {
                "tx_hash": "0xseam",
                "log_index": 7,
                "wallet_address": WALLET,
                "market_id": None,
                "token_id": "t1",
                "side": "BUY",
                "price": Decimal("0.5"),
                "size": Decimal("100"),
                "usdc_amount": Decimal("50"),
                "block_number": 995,
                "timestamp": datetime.fromtimestamp(chain_ts, tz=UTC),
                "source": "chain",
            },
        ],
    )
    await session.commit()

    # Data-API returns the SAME fill (same tx) plus older history, all <= handoff timestamp.
    pool = FakePool(head=1000, ts_map={995: 1_700_000_005})
    trades = [
        _dapi_trade("0xseam", chain_ts),  # duplicate of the live-tail fill
        _dapi_trade("0x" + f"{1:064x}", 1_700_000_001),  # genuinely older, data_api-only
    ]
    _wire_backfill(monkeypatch, session, pool, trades)
    await backfill_wallet_history(WALLET)

    pos = (await session.scalars(select(Position).where(Position.wallet_address == WALLET))).all()
    # 2 distinct fills of size 100 each -> 200, NOT 300 (which is the double-count bug).
    assert pos[0].net_size == Decimal("200"), f"double-counted: {pos[0].net_size}"


# --------------------------------------------------------------------------------------
# 13. indexer checkpoints — monotonic
# --------------------------------------------------------------------------------------
async def test_checkpoint_monotonic(session) -> None:
    await set_checkpoint(session, "s1", 100)
    await set_checkpoint(session, "s1", 50)  # backward -> ignored
    assert await get_checkpoint(session, "s1") == 100
    await set_checkpoint(session, "s1", 150)
    assert await get_checkpoint(session, "s1") == 150


# --------------------------------------------------------------------------------------
# 14. listener.poll_live_tail — first run seeds head; range run ingests + advances
# --------------------------------------------------------------------------------------
async def test_poll_live_tail_first_run_seeds_head(session, monkeypatch) -> None:
    pool = FakePool(head=1000, ts_map={})
    monkeypatch.setattr(listener_mod, "RpcPool", lambda urls: pool)
    monkeypatch.setattr(listener_mod, "async_session_factory", _session_factory(session))
    monkeypatch.setattr(listener_mod.get_settings(), "polygon_rpc_urls", "http://x", raising=False)
    inserted = await listener_mod.poll_live_tail()
    assert inserted == 0
    assert await get_checkpoint(session, listener_mod.LIVE_STREAM) == 995  # head-confirmations


def _session_factory(session):
    class _CtxFactory:
        def __call__(self):
            return self

        async def __aenter__(self):
            return session

        async def __aexit__(self, *exc):
            return False

    return _CtxFactory()
