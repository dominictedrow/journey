"""/portfolio endpoints — live summary, positions, equity history, revalue (Phase 5)."""

from fastapi import APIRouter, Query
from sqlalchemy import select

from app.database.models import PortfolioPosition, PortfolioSnapshot
from app.deps import DbSession
from app.portfolio.valuation import revalue, summarize
from app.schemas.portfolio import (
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


@router.post("/revalue", response_model=PortfolioSnapshotOut, status_code=201)
async def revalue_portfolio(session: DbSession, mode: str = "paper") -> PortfolioSnapshot:
    """Mark positions to latest prices and persist a fresh snapshot."""
    snapshot = await revalue(session, mode)
    await session.commit()
    return snapshot
