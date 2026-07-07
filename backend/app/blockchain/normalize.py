"""Turn decoded fills into unified `trades` rows and derive `positions`.

Dedup guarantee: inserts ignore conflicts on the (tx_hash, log_index) unique constraint,
so re-scanning any block range (backfill restarts, listener overlap) never duplicates.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast

from sqlalchemy import CursorResult, delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.blockchain.decoders import Fill
from app.database.models import Market, Position, Trade, Wallet

SCALE = Decimal(10) ** 6  # collateral and CTF outcome tokens both use 6 decimals
PRICE_QUANTUM = Decimal("0.000001")


def fill_to_trade(fill: Fill, timestamp: datetime, token_to_market: dict[str, str]) -> dict:
    """Map one OrderFilled to a trades row for the maker wallet."""
    if fill.maker_asset_id == 0:  # maker paid collateral for outcome tokens
        side, token_id = "BUY", str(fill.taker_asset_id)
        usdc, size = fill.maker_amount, fill.taker_amount
    else:  # maker sold outcome tokens for collateral
        side, token_id = "SELL", str(fill.maker_asset_id)
        usdc, size = fill.taker_amount, fill.maker_amount
    size_d = Decimal(size) / SCALE
    usdc_d = Decimal(usdc) / SCALE
    return {
        "tx_hash": fill.tx_hash,
        "log_index": fill.log_index,
        "wallet_address": fill.maker.lower(),
        "market_id": token_to_market.get(token_id),
        "token_id": token_id,
        "side": side,
        "price": (usdc_d / size_d).quantize(PRICE_QUANTUM) if size_d else Decimal(0),
        "size": size_d,
        "usdc_amount": usdc_d,
        "block_number": fill.block_number,
        "timestamp": timestamp,
        "source": "chain",
    }


def _insert_ignore(session: AsyncSession) -> Any:
    return pg_insert if session.bind.dialect.name == "postgresql" else sqlite_insert  # type: ignore[union-attr]


# Postgres caps a statement at 32767 bind params; at ~12 cols/row, batch well under that so a
# large Data-API backfill (tens of thousands of fills) doesn't overflow in one INSERT.
_INSERT_BATCH = 1000


async def insert_trades(session: AsyncSession, rows: list[dict]) -> int:
    """Idempotent batched insert; ensures wallet rows exist first. Returns rows actually added."""
    if not rows:
        return 0
    insert = _insert_ignore(session)
    wallets = [{"address": a} for a in {r["wallet_address"] for r in rows}]
    await session.execute(insert(Wallet).values(wallets).on_conflict_do_nothing())
    added = 0
    for start in range(0, len(rows), _INSERT_BATCH):
        result = await session.execute(
            insert(Trade)
            .values(rows[start : start + _INSERT_BATCH])
            .on_conflict_do_nothing(index_elements=["tx_hash", "log_index"])
        )
        added += cast(CursorResult[Any], result).rowcount or 0
    return added


def compute_positions(trades: Sequence[Trade]) -> list[dict]:
    """Average-cost positions per (wallet, token) from chronologically ordered trades.

    Long-only (CLOB sells require holding the token); over-sells from unindexed transfers
    clamp at zero rather than going short.
    """
    # Order by timestamp (present on every row) so Data-API history (block_number is null)
    # interleaves correctly with on-chain rows; block/log_index only break intra-block ties.
    state: dict[tuple[str, str], dict] = {}
    for t in sorted(trades, key=lambda t: (t.timestamp, t.block_number or 0, t.log_index)):
        pos = state.setdefault(
            (t.wallet_address, t.token_id or ""),
            {
                "wallet_address": t.wallet_address,
                "market_id": t.market_id,
                "token_id": t.token_id,
                "net_size": Decimal(0),
                "avg_entry_price": None,
                "realized_pnl": Decimal(0),
                "opened_at": t.timestamp,
                "closed_at": None,
            },
        )
        pos["market_id"] = pos["market_id"] or t.market_id
        net, avg = pos["net_size"], pos["avg_entry_price"] or Decimal(0)
        if t.side == "BUY":
            pos["avg_entry_price"] = ((avg * net + t.price * t.size) / (net + t.size)).quantize(
                PRICE_QUANTUM
            )
            pos["net_size"] = net + t.size
            pos["closed_at"] = None
        else:
            matched = min(t.size, net)
            pos["realized_pnl"] += (t.price - avg) * matched
            pos["net_size"] = net - matched
            if pos["net_size"] == 0:
                pos["closed_at"] = t.timestamp
    for pos in state.values():
        pos["is_open"] = pos["net_size"] > 0
    return list(state.values())


async def rebuild_positions(session: AsyncSession, wallet_address: str) -> None:
    """Recompute a wallet's positions from its trades (delete + insert; idempotent)."""
    trades = (
        await session.scalars(select(Trade).where(Trade.wallet_address == wallet_address))
    ).all()
    await session.execute(delete(Position).where(Position.wallet_address == wallet_address))
    session.add_all(Position(**row) for row in compute_positions(trades))


async def load_token_market_map(session: AsyncSession) -> dict[str, str]:
    """token_id -> market_id (condition_id) from synced markets."""
    rows = await session.execute(select(Market.id, Market.token_ids))
    return {str(t): mid for mid, tokens in rows if tokens for t in tokens}


def utc_from_unix(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)
