"""/backtests endpoints — create/status/result for Module-7 runs (Phase 4)."""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from app.backtesting.models import BacktestConfig
from app.database.models import Backtest
from app.deps import DbSession
from app.schemas.backtest import BacktestDetailOut, BacktestListOut, BacktestOut

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", response_model=BacktestOut, status_code=202)
async def create_backtest(session: DbSession, config: BacktestConfig) -> Backtest:
    """Persist the run (status `queued`) and hand it to the worker (`backtests.run`)."""
    from app.workers.celery_app import celery_app  # deferred: not needed at API import time

    row = Backtest(params=config.model_dump(mode="json"), status="queued")
    session.add(row)
    await session.commit()
    celery_app.send_task("backtests.run", args=[row.id])
    return row


@router.get("", response_model=BacktestListOut)
async def list_backtests(
    session: DbSession,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> BacktestListOut:
    total = await session.scalar(select(func.count()).select_from(Backtest)) or 0
    rows = (
        await session.scalars(
            select(Backtest).order_by(Backtest.id.desc()).limit(limit).offset(offset)
        )
    ).all()
    items = [BacktestOut.model_validate(row) for row in rows]
    return BacktestListOut(items=items, total=total, limit=limit, offset=offset)


@router.get("/{backtest_id}", response_model=BacktestDetailOut)
async def get_backtest(session: DbSession, backtest_id: int) -> Backtest:
    """Full result: status, metrics, equity curve, and trade log."""
    row = await session.get(Backtest, backtest_id)
    if row is None:
        raise HTTPException(status_code=404, detail="backtest not found")
    return row
