"""Phase 6 task: weekly strategy optimization (Module 6)."""

from typing import Any

from app.database.base import async_session_factory
from app.strategy_optimizer.optimizer import optimize
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="strategies.optimize")
def optimize_strategies_task() -> dict[str, Any]:
    """Rescore, backtest the candidate grid, and adopt the best if it beats thresholds+risk."""

    async def job() -> dict[str, Any]:
        async with async_session_factory() as session:
            result = await optimize(session)
            return {
                "adopted": result.adopted,
                "strategy_id": result.strategy_id,
                "objective": result.objective,
                "decision": result.rationale.get("decision"),
                "summary": result.rationale.get("summary"),
                "candidates": len(result.candidates),
            }

    return run_async(job())
