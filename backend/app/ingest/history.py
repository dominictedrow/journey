"""Historical backfill via Polymarket's Data-API (free, unlimited) instead of RPC getLogs.

Free RPC tiers cap eth_getLogs hard (Alchemy: 10 blocks, QuickNode Discover: 5), so a
range-walking backfill is impossible there. The Data-API returns a wallet's full trade
history directly, at no cost, so RPC is left to do only the live tail.

Bulletproof seam with the live tail
-----------------------------------
A transaction is mined in exactly one block, so a tx_hash sits entirely on one side of any
block boundary. We split coverage at a per-wallet **handoff block H**, frozen once in
`indexer_state` under stream ``history:{address}``:

    - Data-API ingests fills with block timestamp <= timestamp(H)   (blocks <= H)
    - the live tail ingests blocks > H

Polygon enforces strictly-increasing block timestamps, so the split is exact — no overlap,
no gap. The two sources therefore touch **disjoint tx_hash sets** and can never collide on
the (tx_hash, log_index) unique constraint, even though Data-API rows carry no real log
index. Within a tx we assign a deterministic synthetic log index (stable sort of that tx's
fills), so re-running a backfill is perfectly idempotent.

The handoff is frozen (write-once) so repeat backfills never widen the Data-API window into
range the live tail already owns (which would double-count). When an analyse-only wallet is
later promoted to tracked, clear its ``history:{address}`` state before re-backfilling so the
handoff is re-derived at the current live checkpoint.
"""

from collections import defaultdict
from datetime import UTC
from decimal import Decimal

from sqlalchemy import func, select

from app.blockchain.client import RpcPool
from app.blockchain.indexer import get_checkpoint, set_checkpoint
from app.blockchain.listener import LIVE_STREAM
from app.blockchain.normalize import (
    PRICE_QUANTUM,
    insert_trades,
    load_token_market_map,
    rebuild_positions,
    utc_from_unix,
)
from app.config import get_settings
from app.database.base import async_session_factory
from app.database.models import Trade, Wallet
from app.ingest.data_api import iter_trades


def _with_synthetic_log_index(trades: list[dict]) -> list[tuple[dict, int]]:
    """Assign each fill a deterministic per-tx ordinal as its log_index.

    Grouping is per transactionHash and the sort key uses only fields the Data-API provides,
    so the same immutable history always yields the same (tx, index) pairs — idempotent even
    for byte-identical fills in one tx (they get consecutive ordinals, re-inserted as no-ops).
    """
    by_tx: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_tx[t["transactionHash"]].append(t)
    out: list[tuple[dict, int]] = []
    for group in by_tx.values():
        ordered = sorted(
            group,
            key=lambda r: (str(r["asset"]), str(r["side"]), str(r["price"]), str(r["size"])),
        )
        for index, row in enumerate(ordered):
            out.append((row, index))
    return out


def data_api_trade_to_row(trade: dict, log_index: int, token_map: dict[str, str]) -> dict:
    """Map one Data-API /trades record to a `trades` row (mirrors fill_to_trade's shape)."""
    size = Decimal(str(trade["size"]))
    price = Decimal(str(trade["price"])).quantize(PRICE_QUANTUM)
    token_id = str(trade["asset"])
    return {
        "tx_hash": trade["transactionHash"],
        "log_index": log_index,
        "wallet_address": trade["proxyWallet"].lower(),
        # token_map only holds synced markets, so market_id stays null (not an FK violation)
        # when the market isn't local yet — same contract as the on-chain path.
        "market_id": token_map.get(token_id),
        "token_id": token_id,
        "side": str(trade["side"]).upper(),  # BUY / SELL (the user's action)
        "price": price,
        "size": size,
        "usdc_amount": size * price,
        "block_number": None,  # Data-API carries no block; positions order by timestamp
        "timestamp": utc_from_unix(int(trade["timestamp"])),
        "source": "data_api",
    }


async def _resolve_handoff(session, pool: RpcPool, address: str, tracked: bool) -> int:
    """Return the frozen handoff block for this wallet, deriving + freezing it on first call."""
    stream = f"history:{address}"
    frozen = await get_checkpoint(session, stream)
    if frozen is not None:
        return frozen
    head = await pool.get_block_number() - get_settings().chain_confirmations
    if tracked:
        # Hand off at the live tail's cursor so it owns every block after H. Initialise the
        # shared cursor to `head` if the tail has never run, so history and live tail meet.
        live_cp = await get_checkpoint(session, LIVE_STREAM)
        if live_cp is None:
            await set_checkpoint(session, LIVE_STREAM, head)
            await session.commit()
            live_cp = await get_checkpoint(session, LIVE_STREAM)
        handoff = live_cp if live_cp is not None else head
    else:
        handoff = head  # analyse-only wallet: no live coverage, so take full history to head
    await set_checkpoint(session, stream, handoff)
    await session.commit()
    return handoff


async def backfill_wallet_history(address: str) -> int:
    """Backfill one wallet's fills from the Data-API up to its handoff block, then rebuild
    positions. Idempotent; returns the number of new rows inserted."""
    settings = get_settings()
    pool = RpcPool(settings.polygon_rpc_url_list)
    address = address.lower()
    async with async_session_factory() as session:
        tracked = bool(
            await session.scalar(select(Wallet.is_tracked).where(Wallet.address == address))
        )
        handoff = await _resolve_handoff(session, pool, address, tracked)
        cutoff_ts = int((await pool.get_block(handoff))["timestamp"])
        # The disjoint-source guarantee (history owns blocks <= H, live tail owns > H) breaks if
        # the live tail already ingested blocks <= H for this wallet — which happens whenever the
        # wallet was marked is_tracked before this backfill ran, since LIVE_STREAM is a single
        # cursor that advances over every tracked wallet. Those fills are already stored as
        # source='chain' with real log indices; re-fetching them from the Data-API (synthetic
        # log_index) would not dedup on (tx_hash, log_index) and would double-count in positions.
        # Cap the Data-API window strictly below the earliest existing chain fill so the two
        # sources stay disjoint regardless of the track/backfill ordering. No-op when the live
        # tail has never stored a chain row for this wallet (the common first-backfill case).
        earliest_chain = await session.scalar(
            select(func.min(Trade.timestamp)).where(
                Trade.wallet_address == address, Trade.source == "chain"
            )
        )
        if earliest_chain is not None:
            # SQLite hands datetimes back tz-naive; treat those as UTC (Postgres is tz-aware).
            if earliest_chain.tzinfo is None:
                earliest_chain = earliest_chain.replace(tzinfo=UTC)
            cutoff_ts = min(cutoff_ts, int(earliest_chain.timestamp()) - 1)
        token_map = await load_token_market_map(session)
        # Stream windows (seeded at the cutoff so the API skips trades the live tail owns) and
        # insert each before fetching the next, so peak memory stays at one window rather than
        # the whole history. A transaction's fills share one timestamp and so land entirely in a
        # single window, so the synthetic per-tx log index is identical to the whole-history
        # computation — still fully idempotent. Per-window commits also make a killed backfill
        # resumable: the frozen handoff + (tx_hash, log_index) dedup absorb the re-run.
        inserted = 0
        async for window in iter_trades(address, end=cutoff_ts):
            rows = [
                data_api_trade_to_row(t, index, token_map)
                for t, index in _with_synthetic_log_index(window)
            ]
            inserted += await insert_trades(session, rows)
            await session.commit()
        await rebuild_positions(session, address)
        await session.commit()
    return inserted
