"""Dedup + checkpoint guarantees against a real database (in-memory SQLite via conftest).

Postgres enforces the same (tx_hash, log_index) unique constraint in production;
insert_trades picks the matching ON CONFLICT dialect at runtime.
"""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.blockchain.indexer import get_checkpoint, set_checkpoint
from app.blockchain.normalize import insert_trades, rebuild_positions
from app.database.models import Position, Trade, Wallet

WALLET = "0x" + "aa" * 20


def trade_row(log_index: int) -> dict:
    return {
        "tx_hash": "0x" + "ab" * 32,
        "log_index": log_index,
        "wallet_address": WALLET,
        "market_id": None,
        "token_id": "t1",
        "side": "BUY",
        "price": Decimal("0.5"),
        "size": Decimal("100"),
        "usdc_amount": Decimal("50"),
        "block_number": 45_000_000,
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
    }


async def test_insert_trades_dedups_on_tx_hash_log_index(session: AsyncSession) -> None:
    rows = [trade_row(0), trade_row(1)]
    assert await insert_trades(session, rows) == 2
    # Re-scanning the same range (restart / overlap) inserts nothing new.
    assert await insert_trades(session, rows) == 0
    assert await insert_trades(session, [trade_row(1), trade_row(2)]) == 1
    assert await session.scalar(select(func.count()).select_from(Trade)) == 3
    # Wallet row was auto-created exactly once.
    assert await session.scalar(select(func.count()).select_from(Wallet)) == 1


async def test_checkpoint_is_monotonic(session: AsyncSession) -> None:
    assert await get_checkpoint(session, "backfill:x") is None
    await set_checkpoint(session, "backfill:x", 100)
    await set_checkpoint(session, "backfill:x", 90)  # overlapping run must not rewind
    assert await get_checkpoint(session, "backfill:x") == 100
    await set_checkpoint(session, "backfill:x", 110)
    assert await get_checkpoint(session, "backfill:x") == 110


async def test_rebuild_positions_is_idempotent(session: AsyncSession) -> None:
    await insert_trades(session, [trade_row(0)])
    await rebuild_positions(session, WALLET)
    await session.commit()
    await rebuild_positions(session, WALLET)
    await session.commit()
    positions = (await session.scalars(select(Position))).all()
    assert len(positions) == 1
    assert positions[0].net_size == Decimal("100")
    assert positions[0].is_open is True
