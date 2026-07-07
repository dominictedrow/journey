"""Phase 1 tasks: markets sync, wallet backfill, live chain tail."""

from app.blockchain.listener import poll_live_tail
from app.database.base import async_session_factory
from app.ingest.gamma import sync_markets
from app.ingest.history import backfill_wallet_history
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="ingest.sync_markets")
def sync_markets_task() -> int:
    async def job() -> int:
        async with async_session_factory() as session:
            return await sync_markets(session)

    return run_async(job())


@celery_app.task(name="chain.backfill_wallet")
def backfill_wallet_task(address: str) -> int:
    return run_async(backfill_wallet_history(address))


@celery_app.task(name="chain.live_tail", time_limit=120)
def live_tail_task() -> int:
    return run_async(poll_live_tail())
