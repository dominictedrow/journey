"""Phase 2 tasks: wallet metrics computation (Module 3)."""

from app.database.base import async_session_factory
from app.database.models import Wallet
from app.ingest.history import backfill_wallet_history
from app.wallet_analytics.metrics import compute_wallet_metrics, wallets_with_trades
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="analytics.compute_wallet_metrics")
def compute_wallet_metrics_task(address: str) -> str:
    async def job() -> str:
        async with async_session_factory() as session:
            row = await compute_wallet_metrics(session, address)
            return row.wallet_address

    return run_async(job())


@celery_app.task(name="analytics.refresh_all_metrics")
def refresh_all_metrics_task() -> int:
    """Recompute metrics for every wallet that has indexed fills."""

    async def job() -> int:
        async with async_session_factory() as session:
            addresses = await wallets_with_trades(session)
            for address in addresses:
                await compute_wallet_metrics(session, address)
            return len(addresses)

    return run_async(job())


@celery_app.task(name="analytics.backfill_and_compute")
def backfill_and_compute_task(address: str) -> int:
    """Full Data-API history backfill for one wallet, then fresh metrics (UI 'backfill' button)."""

    async def job() -> int:
        inserted = await backfill_wallet_history(address)
        async with async_session_factory() as session:
            await compute_wallet_metrics(session, address)
        return inserted

    return run_async(job())


@celery_app.task(name="analytics.promote_and_backfill")
def promote_and_backfill_task(address: str) -> int:
    """Analyse-only → tracked promotion (wallet PATCH route, strategy application): mark
    tracked, drop any frozen history handoff from an untracked-era backfill so the seam is
    re-derived at the live cursor (no coverage gap), then full backfill + fresh metrics."""

    async def job() -> int:
        from app.ingest.history import clear_history_checkpoint

        lowered = address.lower()
        async with async_session_factory() as session:
            wallet = await session.get(Wallet, lowered)
            if wallet is None:
                return 0
            wallet.is_tracked = True
            await clear_history_checkpoint(session, lowered)
            await session.commit()
        inserted = await backfill_wallet_history(lowered)
        async with async_session_factory() as session:
            await compute_wallet_metrics(session, lowered)
        return inserted

    return run_async(job())
