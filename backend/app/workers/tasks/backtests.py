"""Phase 4 tasks: backtest execution (Module 7)."""

from app.backtesting.engine import execute_backtest
from app.database.base import async_session_factory
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="backtests.run")
def run_backtest_task(backtest_id: int) -> str:
    """Execute one queued `backtests` row; the row carries status/result."""

    async def job() -> str:
        async with async_session_factory() as session:
            row = await execute_backtest(session, backtest_id)
            return row.status

    return run_async(job())
