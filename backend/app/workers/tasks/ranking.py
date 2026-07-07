"""Phase 3 tasks: composite scoring + ranking generations (Modules 4-5)."""

from typing import Any

from app.database.base import async_session_factory
from app.ranking.rankings import rebuild_rankings
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="ranking.rebuild")
def rebuild_rankings_task() -> dict[str, Any]:
    """Rescore every wallet with metrics and write a fresh rankings generation."""

    async def job() -> dict[str, Any]:
        async with async_session_factory() as session:
            result = await rebuild_rankings(session)
            return {
                "computed_at": result.computed_at.isoformat(),
                "wallets_scored": result.wallets_scored,
                "categories": result.category_counts,
            }

    return run_async(job())
