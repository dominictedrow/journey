"""Module 10 — mark-to-market and snapshots.

`revalue` prices every open position at its token's latest observed trade price, updates
each holding's unrealized PnL, and appends a `portfolio_snapshots` row (cash, total value,
realized/unrealized PnL, open-position count). The snapshot series is both the equity curve
the UI draws and the baseline the risk gate's loss limits compare against.
"""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import PortfolioPosition, PortfolioSnapshot, Trade
from app.portfolio.service import (
    QUANTUM,
    current_cash,
    open_positions,
    realized_pnl_total,
)


def _dec(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM)


async def latest_price(session: AsyncSession, token_id: str | None) -> Decimal | None:
    """Most recent observed price for a token (the copy portfolio marks against source prints)."""
    if token_id is None:
        return None
    price = await session.scalar(
        select(Trade.price)
        .where(Trade.token_id == token_id)
        .order_by(Trade.timestamp.desc(), Trade.block_number.desc(), Trade.log_index.desc())
        .limit(1)
    )
    return Decimal(str(price)) if price is not None else None


async def mark_position(
    session: AsyncSession, position: PortfolioPosition
) -> tuple[Decimal, Decimal]:
    """Return (market_value, unrealized_pnl) for a position and update its unrealized PnL."""
    price = await latest_price(session, position.token_id)
    if price is None:
        price = position.avg_entry_price or Decimal(0)
    market_value = _dec(position.size * price)
    unrealized = _dec((price - (position.avg_entry_price or Decimal(0))) * position.size)
    position.unrealized_pnl = unrealized
    return market_value, unrealized


async def revalue(
    session: AsyncSession, mode: str = "paper", now: datetime | None = None
) -> PortfolioSnapshot:
    """Mark every open position, write a fresh snapshot, and return it (does not commit)."""
    now = now or datetime.now(UTC)
    positions = await open_positions(session, mode)

    held_value = Decimal(0)
    unrealized_total = Decimal(0)
    for position in positions:
        market_value, unrealized = await mark_position(session, position)
        held_value += market_value
        unrealized_total += unrealized

    cash = await current_cash(session, mode)
    snapshot = PortfolioSnapshot(
        timestamp=now,
        mode=mode,
        cash=cash,
        total_value=_dec(cash + held_value),
        realized_pnl=await realized_pnl_total(session, mode),
        unrealized_pnl=_dec(unrealized_total),
        open_positions=len(positions),
    )
    session.add(snapshot)
    return snapshot


async def summarize(session: AsyncSession, mode: str = "paper") -> dict:
    """Live portfolio totals for the API (marks positions without writing a snapshot)."""
    positions = await open_positions(session, mode)
    held_value = Decimal(0)
    unrealized_total = Decimal(0)
    for position in positions:
        market_value, unrealized = await mark_position(session, position)
        held_value += market_value
        unrealized_total += unrealized

    cash = await current_cash(session, mode)
    last_snapshot_at = await session.scalar(
        select(PortfolioSnapshot.timestamp)
        .where(PortfolioSnapshot.mode == mode)
        .order_by(PortfolioSnapshot.timestamp.desc())
        .limit(1)
    )
    return {
        "mode": mode,
        "cash": cash,
        "total_value": _dec(cash + held_value),
        "realized_pnl": await realized_pnl_total(session, mode),
        "unrealized_pnl": _dec(unrealized_total),
        "open_positions": len(positions),
        "last_snapshot_at": last_snapshot_at,
        "positions": positions,
    }
