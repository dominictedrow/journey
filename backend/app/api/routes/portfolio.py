"""/portfolio endpoints — live summary, positions, equity history, revalue, manual close
(Phase 5; close added with the autonomy wiring)."""

from datetime import UTC, datetime
from decimal import Decimal

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.config import get_settings
from app.copy_trading.executors import Executor, LiveExecutor, PaperExecutor, build_live_client
from app.database.models import CopyOrder, PortfolioPosition, PortfolioSnapshot
from app.deps import DbSession
from app.portfolio.service import apply_fill
from app.portfolio.valuation import latest_price, revalue, summarize
from app.risk.models import TradeIntent
from app.schemas.portfolio import (
    ClosePositionOut,
    PortfolioHistoryOut,
    PortfolioPositionOut,
    PortfolioSnapshotOut,
    PortfolioSummaryOut,
)

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.get("", response_model=PortfolioSummaryOut)
async def portfolio_summary(session: DbSession, mode: str = "paper") -> PortfolioSummaryOut:
    """Cash, marked total value, realized/unrealized PnL, and open positions right now."""
    summary = await summarize(session, mode)
    return PortfolioSummaryOut(
        mode=summary["mode"],
        cash=summary["cash"],
        total_value=summary["total_value"],
        realized_pnl=summary["realized_pnl"],
        unrealized_pnl=summary["unrealized_pnl"],
        open_positions=summary["open_positions"],
        last_snapshot_at=summary["last_snapshot_at"],
        positions=[PortfolioPositionOut.model_validate(p) for p in summary["positions"]],
    )


@router.get("/positions", response_model=list[PortfolioPositionOut])
async def list_positions(
    session: DbSession, mode: str = "paper", open_only: bool = True
) -> list[PortfolioPosition]:
    query = select(PortfolioPosition).where(PortfolioPosition.mode == mode)
    if open_only:
        query = query.where(PortfolioPosition.is_open.is_(True))
    query = query.order_by(PortfolioPosition.opened_at.desc())
    return list((await session.scalars(query)).all())


@router.get("/history", response_model=PortfolioHistoryOut)
async def portfolio_history(
    session: DbSession, mode: str = "paper", limit: int = Query(500, ge=1, le=5000)
) -> PortfolioHistoryOut:
    """The snapshot series (equity curve), oldest first."""
    rows = (
        await session.scalars(
            select(PortfolioSnapshot)
            .where(PortfolioSnapshot.mode == mode)
            .order_by(PortfolioSnapshot.timestamp.desc())
            .limit(limit)
        )
    ).all()
    items = [PortfolioSnapshotOut.model_validate(s) for s in reversed(rows)]
    return PortfolioHistoryOut(mode=mode, items=items)


@router.post("/positions/{position_id}/close", response_model=ClosePositionOut)
async def close_position(session: DbSession, position_id: int) -> ClosePositionOut:
    """Manually close one open position — the human override the autonomous loop keeps.

    Deliberately bypasses the risk gate: exits pass every rule anyway, and an engaged kill
    switch must never trap a human trying to flatten a position. Paper positions sell
    against the simulated book; live positions submit a real FAK sell (requires an armed
    live deployment). The fill is recorded as a manual-close CopyOrder so cash and realized
    PnL stay on the copy_orders spine."""
    position = await session.get(PortfolioPosition, position_id)
    if position is None:
        raise HTTPException(status_code=404, detail="position not found")
    if not position.is_open or position.size <= 0:
        raise HTTPException(status_code=409, detail="position is already closed")

    settings = get_settings()
    executor: Executor
    if position.mode == "live":
        if not (settings.execution_mode == "live" and settings.live_trading_armed):
            raise HTTPException(
                status_code=409,
                detail="live position, but this deployment is not armed for live execution",
            )
        executor = LiveExecutor(client=build_live_client(settings), settings=settings)
    else:
        executor = PaperExecutor()

    price = await latest_price(session, position.token_id)
    if price is None:
        price = position.avg_entry_price or Decimal("0.5")
    intent = TradeIntent(
        source_wallet=position.source_wallet or "",
        token_id=position.token_id or "",
        side="SELL",
        size=position.size,
        price=price,
        market_id=position.market_id,
        delay_seconds=0,
        mode=position.mode,
    )
    result = await executor.execute(intent, position.size)

    now = datetime.now(UTC)
    order = CopyOrder(
        created_at=now,
        mode=position.mode,
        source_wallet=position.source_wallet,
        source_trade_id=None,
        market_id=position.market_id,
        token_id=position.token_id,
        side="SELL",
        requested_size=position.size,
        requested_price=price,
        status=result.status,
        filled_size=result.filled_size,
        fill_price=result.fill_price,
        delay_ms=result.delay_ms,
        slippage=result.slippage,
        error=result.error or "manual close",
        external_order_id=result.external_order_id,
    )
    session.add(order)
    if result.filled_size > 0 and result.fill_price is not None:
        await apply_fill(
            session,
            mode=position.mode,
            source_wallet=position.source_wallet,
            market_id=position.market_id,
            token_id=position.token_id or "",
            side="SELL",
            filled_size=result.filled_size,
            fill_price=result.fill_price,
            now=now,
        )
        await revalue(session, position.mode, now=now)
    await session.commit()
    await session.refresh(position)
    return ClosePositionOut(
        status=result.status,
        filled_size=result.filled_size,
        fill_price=result.fill_price,
        error=result.error,
        position=PortfolioPositionOut.model_validate(position),
    )


@router.post("/revalue", response_model=PortfolioSnapshotOut, status_code=201)
async def revalue_portfolio(session: DbSession, mode: str = "paper") -> PortfolioSnapshot:
    """Mark positions to latest prices and persist a fresh snapshot."""
    snapshot = await revalue(session, mode)
    await session.commit()
    return snapshot
