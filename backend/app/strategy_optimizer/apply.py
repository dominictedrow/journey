"""Module 6 — strategy application: the adopted strategy drives who is actually copied.

`apply_active_strategy` closes the optimizer → execution loop. The optimizer adopts a
strategy whose params carry the resolved wallet set + weights (see `optimize`); this
enforces that selection on the `wallets` table:

  - every strategy wallet gets `copy_enabled=True` (and `is_tracked=True`, so the live
    tail streams its fills);
  - every other copy-enabled wallet is demoted with a recorded reason — decayed wallets
    rotate out without anyone clicking anything.

There is no manual veto: the optimizer's selection is authoritative and unconditionally
enforced, so a wallet that rotated out is re-enabled the moment it's back in the active
strategy's selection. Runs from the weekly optimizer task and an hourly enforcement beat,
so drift or a restored DB converges back to the adopted strategy within the hour.

Returns a summary dict; `needs_backfill` lists wallets that were newly promoted to
tracked and therefore need their history seam re-derived (`history:{address}` cleared +
Data-API backfill). The Celery task layer enqueues those — this function stays
broker-free so it is directly testable.
"""

import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import Strategy, Wallet

logger = logging.getLogger(__name__)


async def get_active_strategy(session: AsyncSession) -> Strategy | None:
    return await session.scalar(
        select(Strategy).where(Strategy.is_active.is_(True)).order_by(Strategy.id.desc()).limit(1)
    )


async def apply_active_strategy(session: AsyncSession) -> dict:
    """Enforce the active strategy's wallet selection; commits. See module docstring."""
    active = await get_active_strategy(session)
    if active is None:
        return {"applied": False, "reason": "no active strategy"}
    selected = [a.lower() for a in (active.params or {}).get("wallets") or []]
    if not selected:
        # Pre-wiring strategy rows carry only the spec; nothing concrete to enforce.
        return {"applied": False, "reason": "active strategy has no resolved wallets"}

    enabled: list[str] = []
    needs_backfill: list[str] = []
    rows = (await session.scalars(select(Wallet).where(Wallet.address.in_(selected)))).all()
    for wallet in rows:
        if not wallet.is_tracked:
            needs_backfill.append(wallet.address)
            wallet.is_tracked = True
        if not wallet.copy_enabled:
            wallet.copy_enabled = True
            enabled.append(wallet.address)
        wallet.disabled_reason = None

    demoted: list[str] = []
    others = (
        await session.scalars(
            select(Wallet).where(Wallet.copy_enabled.is_(True), Wallet.address.not_in(selected))
        )
    ).all()
    for wallet in others:
        wallet.copy_enabled = False
        wallet.disabled_reason = f"rotated out by optimizer (strategy {active.id})"
        demoted.append(wallet.address)

    await session.commit()
    if enabled or demoted:
        logger.info(
            "strategy %s applied: %d enabled, %d demoted, %d need backfill",
            active.id,
            len(enabled),
            len(demoted),
            len(needs_backfill),
        )
    return {
        "applied": True,
        "strategy_id": active.id,
        "enabled": enabled,
        "demoted": demoted,
        "needs_backfill": needs_backfill,
    }
