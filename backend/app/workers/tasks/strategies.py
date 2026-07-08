"""Phase 6 tasks: weekly strategy optimization + strategy enforcement (Module 6)."""

from typing import Any

from app.database.base import async_session_factory
from app.strategy_optimizer.apply import apply_active_strategy
from app.strategy_optimizer.optimizer import optimize
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


def _enqueue_backfills(addresses: list[str]) -> None:
    """Queue seam-safe re-backfills for wallets newly promoted to tracked. Serialized by the
    worker pool + Data-API throttle, so a large promotion wave drains politely."""
    for address in addresses:
        celery_app.send_task("analytics.promote_and_backfill", args=[address])


@celery_app.task(name="strategies.optimize")
def optimize_strategies_task() -> dict[str, Any]:
    """Rescore, backtest the candidate grid, adopt the best if it beats thresholds+risk,
    then enforce the (possibly new) active strategy on the wallet flags."""

    async def job() -> dict[str, Any]:
        async with async_session_factory() as session:
            result = await optimize(session)
            applied = await apply_active_strategy(session)
            return {
                "adopted": result.adopted,
                "strategy_id": result.strategy_id,
                "objective": result.objective,
                "decision": result.rationale.get("decision"),
                "summary": result.rationale.get("summary"),
                "candidates": len(result.candidates),
                "applied": applied,
            }

    outcome = run_async(job())
    _enqueue_backfills(outcome["applied"].get("needs_backfill") or [])
    return outcome


@celery_app.task(name="strategies.apply_active")
def apply_active_strategy_task() -> dict[str, Any]:
    """Hourly enforcement: converge wallet copy flags back to the adopted strategy."""

    async def job() -> dict[str, Any]:
        async with async_session_factory() as session:
            return await apply_active_strategy(session)

    outcome = run_async(job())
    _enqueue_backfills(outcome.get("needs_backfill") or [])
    return outcome
