"""Phase 2 tasks: periodic wallet-discovery sweep + autonomous promotion (Module 2)."""

import logging
from typing import Any

from app.database.base import async_session_factory
from app.ingest.history import backfill_wallet_history
from app.wallet_analytics.metrics import compute_wallet_metrics
from app.wallet_discovery import run_discovery
from app.wallet_discovery.autopromote import (
    mark_tracked,
    promotion_budget,
    promotion_candidates,
)
from app.workers.celery_app import celery_app
from app.workers.tasks import run_async

logger = logging.getLogger(__name__)


@celery_app.task(name="discovery.run")
def run_discovery_task() -> dict[str, int]:
    async def job() -> dict[str, int]:
        async with async_session_factory() as session:
            return await run_discovery(session)

    return run_async(job())


# time_limit stays under the hourly cadence so a slow batch (whale histories) can't pile up
# behind itself; an interrupted batch self-heals next run (see autopromote module docstring).
@celery_app.task(name="discovery.autopromote", time_limit=3300)
def autopromote_task() -> dict[str, Any]:
    """Promote a budgeted batch of discovered wallets: track, backfill history, compute
    metrics — so rankings grow without a human clicking Track."""

    async def job() -> dict[str, Any]:
        async with async_session_factory() as session:
            budget = await promotion_budget(session)
            addresses = await promotion_candidates(session, budget)
            await mark_tracked(session, addresses)
        promoted: list[dict[str, Any]] = []
        for address in addresses:
            # One wallet failing (Data-API hiccup mid-history) shouldn't sink the batch;
            # it has no frozen handoff yet, so the next hourly run retries it.
            try:
                inserted = await backfill_wallet_history(address)
                async with async_session_factory() as session:
                    await compute_wallet_metrics(session, address)
            except Exception:
                logger.exception("autopromote backfill failed for %s", address)
                continue
            promoted.append({"address": address, "trades": inserted})
        return {"budget": budget, "promoted": promoted}

    return run_async(job())
