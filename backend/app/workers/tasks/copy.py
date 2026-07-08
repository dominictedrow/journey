"""Phase 5 tasks: live copy detection + portfolio valuation (Modules 9–10).
Phase 8 adds live-order reconciliation."""

from app.config import get_settings
from app.copy_trading.engine import CopyEngine
from app.copy_trading.executors import build_live_client
from app.copy_trading.recovery import reconcile_live_orders
from app.database.base import async_session_factory
from app.portfolio.settlement import settle_resolved_positions
from app.portfolio.valuation import revalue
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async


@celery_app.task(name="copy.detect_and_copy")
def detect_and_copy_task() -> int:
    """Sweep fresh, uncopied fills from copy-enabled wallets through the pipeline."""

    async def job() -> int:
        async with async_session_factory() as session:
            orders = await CopyEngine().detect_and_copy(session)
            return len(orders)

    return run_async(job())


@celery_app.task(name="copy.reconcile_live")
def reconcile_live_task() -> int:
    """Reconcile live CLOB orders against our records (Phase 8). No-op unless armed live —
    building the live client needs credentials, so we skip entirely on a paper deployment."""

    settings = get_settings()
    if not (settings.execution_mode == "live" and settings.live_trading_armed):
        return 0

    async def job() -> int:
        async with async_session_factory() as session:
            touched = await reconcile_live_orders(session, build_live_client(settings))
            return len(touched)

    return run_async(job())


@celery_app.task(name="copy.settle_resolutions")
def settle_resolutions_task() -> list[int]:
    """Close positions in resolved markets at their $0/$1 payout (Module 10 settlement)."""

    async def job() -> list[int]:
        async with async_session_factory() as session:
            return await settle_resolved_positions(session)

    return run_async(job())


@celery_app.task(name="copy.revalue_portfolio")
def revalue_portfolio_task(mode: str = "paper") -> str | None:
    """Mark the copy portfolio to market and append a snapshot. No-op for the live beat entry
    on a paper-only deployment, so an unarmed box doesn't grow a live snapshot series for a
    mode that isn't actually trading."""
    if mode == "live" and get_settings().execution_mode != "live":
        return None

    async def job() -> str:
        async with async_session_factory() as session:
            snapshot = await revalue(session, mode)
            await session.commit()
            return snapshot.timestamp.isoformat()

    return run_async(job())
