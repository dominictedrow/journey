"""/discovery endpoints — candidate-pool status + manual sweep trigger (Phase 2)."""

from fastapi import APIRouter
from sqlalchemy import func, select

from app.database.models import Wallet
from app.deps import DbSession
from app.schemas.wallet import DiscoverySourceCount, DiscoveryStatusOut, TaskEnqueuedOut

router = APIRouter(prefix="/discovery", tags=["discovery"])


@router.get("", response_model=DiscoveryStatusOut)
async def discovery_status(session: DbSession) -> DiscoveryStatusOut:
    total = await session.scalar(select(func.count()).select_from(Wallet)) or 0
    tracked = (
        await session.scalar(
            select(func.count()).select_from(Wallet).where(Wallet.is_tracked.is_(True))
        )
        or 0
    )
    last = await session.scalar(select(func.max(Wallet.discovered_at)))
    by_source = [
        DiscoverySourceCount(source=source, count=count)
        for source, count in await session.execute(
            select(Wallet.discovery_source, func.count())
            .group_by(Wallet.discovery_source)
            .order_by(func.count().desc())
        )
    ]
    return DiscoveryStatusOut(
        total_wallets=total, tracked_wallets=tracked, last_discovered_at=last, by_source=by_source
    )


@router.post("/run", response_model=TaskEnqueuedOut, status_code=202)
async def run_discovery() -> TaskEnqueuedOut:
    """Queue a discovery sweep across all sources (Celery; also runs on the beat schedule)."""
    from app.workers.celery_app import celery_app  # deferred: not needed at API import time

    result = celery_app.send_task("discovery.run")
    return TaskEnqueuedOut(task_id=result.id)
