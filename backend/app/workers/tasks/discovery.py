"""Phase 2 task: periodic wallet-discovery sweep (Module 2)."""

from app.database.base import async_session_factory
from app.wallet_discovery import run_discovery
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="discovery.run")
def run_discovery_task() -> dict[str, int]:
    async def job() -> dict[str, int]:
        async with async_session_factory() as session:
            return await run_discovery(session)

    return run_async(job())
