"""/strategies endpoints — optimizer output: current active, history, rationale (Phase 6).

The Optimization dashboard reads `active` (what's copying now) against the recommendation
history; each row carries the optimizer's `rationale`. `POST /optimize` queues a fresh run on
the worker (it backtests the whole candidate grid, so it never runs inline on the request).
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from app.database.models import Strategy
from app.deps import DbSession
from app.schemas.strategy import (
    OptimizeQueuedOut,
    StrategyDetailOut,
    StrategyListOut,
    StrategyOut,
)

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.post("/optimize", response_model=OptimizeQueuedOut, status_code=202)
async def run_optimizer() -> OptimizeQueuedOut:
    """Queue a strategy-optimization run (`strategies.optimize`) on the worker."""
    from app.workers.celery_app import celery_app  # deferred: not needed at API import time

    result = celery_app.send_task("strategies.optimize")
    return OptimizeQueuedOut(task_id=result.id, message="optimization queued")


@router.get("", response_model=StrategyListOut)
async def list_strategies(
    session: DbSession,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> StrategyListOut:
    """Strategy history, newest first (each optimizer run records its recommendation)."""
    total = await session.scalar(select(func.count()).select_from(Strategy)) or 0
    rows = (
        await session.scalars(
            select(Strategy).order_by(Strategy.id.desc()).limit(limit).offset(offset)
        )
    ).all()
    items = [StrategyOut.model_validate(row) for row in rows]
    return StrategyListOut(items=items, total=total, limit=limit, offset=offset)


@router.get("/active", response_model=StrategyDetailOut)
async def get_active_strategy(session: DbSession) -> Strategy:
    """The strategy currently driving copy selection (its params + adoption rationale)."""
    row = await session.scalar(select(Strategy).where(Strategy.is_active.is_(True)))
    if row is None:
        raise HTTPException(status_code=404, detail="no active strategy")
    return row


@router.get("/{strategy_id}", response_model=StrategyDetailOut)
async def get_strategy(session: DbSession, strategy_id: int) -> Strategy:
    """One strategy with its full optimizer rationale."""
    row = await session.get(Strategy, strategy_id)
    if row is None:
        raise HTTPException(status_code=404, detail="strategy not found")
    return row
