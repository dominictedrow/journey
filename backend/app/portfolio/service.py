"""Module 10 — copy-portfolio bookkeeping.

`apply_fill` folds one executed copy order into the portfolio's own positions: a BUY
opens or averages up a `(mode, token, source_wallet)` holding, a SELL reduces it and books
realized PnL, closing it at zero. Positions are keyed by source wallet so per-wallet exposure
and attribution survive (a token bought on two wallets' signals is two holdings). Cash is
derived from the filled copy orders rather than stored, so it is always reconstructable.
"""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database.models import CopyOrder, PortfolioPosition, PortfolioSnapshot

QUANTUM = Decimal("0.000001")


def _dec(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM)


async def get_position(
    session: AsyncSession, mode: str, token_id: str, source_wallet: str | None
) -> PortfolioPosition | None:
    """The open holding for one (mode, token, source wallet), if any."""
    return await session.scalar(
        select(PortfolioPosition).where(
            PortfolioPosition.mode == mode,
            PortfolioPosition.token_id == token_id,
            PortfolioPosition.source_wallet == source_wallet,
            PortfolioPosition.is_open.is_(True),
        )
    )


async def apply_fill(
    session: AsyncSession,
    *,
    mode: str,
    source_wallet: str | None,
    market_id: str | None,
    token_id: str,
    side: str,
    filled_size: Decimal,
    fill_price: Decimal,
    now: datetime | None = None,
) -> PortfolioPosition | None:
    """Apply a filled copy order to the portfolio; returns the affected position."""
    now = now or datetime.now(UTC)
    position = await get_position(session, mode, token_id, source_wallet)

    if side == "BUY":
        if position is None:
            position = PortfolioPosition(
                mode=mode,
                market_id=market_id,
                token_id=token_id,
                source_wallet=source_wallet,
                size=Decimal(0),
                avg_entry_price=Decimal(0),
                realized_pnl=Decimal(0),
                unrealized_pnl=Decimal(0),
                is_open=True,
                opened_at=now,
            )
            session.add(position)
        prior = (position.avg_entry_price or Decimal(0)) * position.size
        new_size = position.size + filled_size
        position.avg_entry_price = (
            _dec((prior + fill_price * filled_size) / new_size) if new_size > 0 else Decimal(0)
        )
        position.size = _dec(new_size)
        position.is_open = True
        return position

    # SELL — reduce the holding and realize PnL against the average entry.
    if position is None or position.size <= 0:
        return None
    sell_size = min(filled_size, position.size)
    realized = _dec((fill_price - (position.avg_entry_price or Decimal(0))) * sell_size)
    position.realized_pnl = _dec((position.realized_pnl or Decimal(0)) + realized)
    position.size = _dec(position.size - sell_size)
    if position.size <= 0:
        position.size = Decimal(0)
        position.is_open = False
        position.unrealized_pnl = Decimal(0)
        position.closed_at = now
    return position


def starting_cash(mode: str, settings: Settings | None = None) -> Decimal:
    """Baseline cash/equity before any fills or snapshots exist for this mode.

    Paper has a configured play-money balance; live has no on-chain balance read yet (a
    follow-up once on-chain redemption exists), so it uses a configured fallback instead."""
    settings = settings or get_settings()
    return settings.paper_starting_cash if mode == "paper" else settings.risk_live_starting_equity


async def current_cash(
    session: AsyncSession, mode: str, settings: Settings | None = None
) -> Decimal:
    """Starting cash minus net spend across all filled copy orders for this mode."""
    starting = starting_cash(mode, settings)
    rows = await session.execute(
        select(CopyOrder.side, CopyOrder.filled_size, CopyOrder.fill_price).where(
            CopyOrder.mode == mode,
            CopyOrder.filled_size.is_not(None),
            CopyOrder.fill_price.is_not(None),
        )
    )
    cash = Decimal(starting)
    for side, filled, price in rows:
        notional = _dec(Decimal(filled) * Decimal(price))
        cash += -notional if side == "BUY" else notional
    return _dec(cash)


async def open_positions(session: AsyncSession, mode: str) -> list[PortfolioPosition]:
    return list(
        (
            await session.scalars(
                select(PortfolioPosition)
                .where(
                    PortfolioPosition.mode == mode,
                    PortfolioPosition.is_open.is_(True),
                    PortfolioPosition.size > 0,
                )
                .order_by(PortfolioPosition.opened_at.desc())
            )
        ).all()
    )


async def realized_pnl_total(session: AsyncSession, mode: str) -> Decimal:
    total = await session.scalar(
        select(func.coalesce(func.sum(PortfolioPosition.realized_pnl), 0)).where(
            PortfolioPosition.mode == mode
        )
    )
    return _dec(Decimal(str(total or 0)))


async def current_equity(
    session: AsyncSession, mode: str, settings: Settings | None = None
) -> Decimal:
    """Latest marked equity for this mode, or the starting baseline before any snapshot
    exists. The shared basis for percentage-sizing and the portfolio exposure cap."""
    snapshot = await session.scalar(
        select(PortfolioSnapshot)
        .where(PortfolioSnapshot.mode == mode)
        .order_by(PortfolioSnapshot.timestamp.desc())
        .limit(1)
    )
    if snapshot is not None:
        return Decimal(snapshot.total_value)
    return starting_cash(mode, settings)
