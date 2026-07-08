"""Module 2 → 3 bridge — autonomous promotion of discovered wallets into the scored cohort.

Discovery (leaderboard/holder sweeps) inserts bare addresses; metrics and rankings only see
wallets with indexed fills. This module closes that gap without a human: each run picks a
*budgeted* batch of never-backfilled wallets, marks them tracked (so the live tail streams
their fills from now on — one extra maker topic on the same eth_getLogs call, near-zero RPC
cost), and pulls their full Data-API history serially through the shared throttle. Metrics
follow immediately, so the hourly ranking rebuild scores them on its next pass.

The budget (`autopromote_per_run`, hourly beat) is the Data-API politeness valve: full
histories are the heaviest reads we do, so the cohort grows steadily — ~200/day at the
default — instead of in one unthrottled burst. `max_tracked_wallets` caps cohort size so
the maker-topic list and hourly metrics sweep stay bounded.

"Never backfilled" is keyed on the absence of a frozen `history:{address}` handoff, which
`backfill_wallet_history` writes exactly once per wallet — so zero-trade wallets are not
re-selected forever, and a batch interrupted mid-backfill self-heals: unfinished wallets
still have no handoff and are picked up again next run.
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database.models import IndexerState, Wallet

logger = logging.getLogger(__name__)


async def promotion_budget(session: AsyncSession) -> int:
    """How many wallets this run may promote: per-run budget, clamped by cohort headroom."""
    settings = get_settings()
    tracked = (
        await session.scalar(
            select(func.count()).select_from(Wallet).where(Wallet.is_tracked.is_(True))
        )
        or 0
    )
    return min(settings.autopromote_per_run, max(0, settings.max_tracked_wallets - tracked))


async def promotion_candidates(session: AsyncSession, budget: int) -> list[str]:
    """Newest-discovered wallets that have never been backfilled, up to `budget`."""
    if budget <= 0:
        return []
    backfilled = (
        select(func.replace(IndexerState.stream, "history:", ""))
        .where(IndexerState.stream.like("history:%"))
        .scalar_subquery()
    )
    rows = await session.scalars(
        select(Wallet.address)
        .where(Wallet.address.not_in(backfilled))
        .order_by(Wallet.discovered_at.desc().nulls_last(), Wallet.address)
        .limit(budget)
    )
    return list(rows)


async def mark_tracked(session: AsyncSession, addresses: list[str]) -> None:
    """Flag the batch tracked *before* backfilling, so each wallet's history handoff is
    derived at the live cursor and the tail owns everything after it — no seam gap."""
    if not addresses:
        return
    for wallet in (
        await session.scalars(select(Wallet).where(Wallet.address.in_(addresses)))
    ).all():
        wallet.is_tracked = True
    await session.commit()
