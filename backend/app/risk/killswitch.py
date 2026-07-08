"""Module 8 — global kill switch.

A single DB flag (`system_flags["kill_switch"]`) that halts *all* copy execution. It is
checked first by `RiskEngine.check` and is durable across restarts and shared by every
process (API, worker, beat) through Postgres. Engaging/disengaging records the reason and
writes a `risk_event` for the audit trail (Transparency principle).
"""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import RiskEvent, SystemFlag

KILL_SWITCH_KEY = "kill_switch"


async def is_engaged(session: AsyncSession) -> bool:
    flag = await session.get(SystemFlag, KILL_SWITCH_KEY)
    return bool(flag and flag.enabled)


async def get_state(session: AsyncSession) -> SystemFlag | None:
    return await session.get(SystemFlag, KILL_SWITCH_KEY)


async def set_engaged(
    session: AsyncSession, engaged: bool, reason: str | None = None, *, commit: bool = True
) -> SystemFlag:
    """Engage/disengage the switch; logs a `risk_event` for the transition. Commits unless
    the caller is mid-transaction (`commit=False`) and will commit atomically itself."""
    now = datetime.now(UTC)
    flag = await session.get(SystemFlag, KILL_SWITCH_KEY)
    if flag is None:
        flag = SystemFlag(key=KILL_SWITCH_KEY, enabled=engaged, reason=reason, updated_at=now)
        session.add(flag)
    else:
        flag.enabled = engaged
        flag.reason = reason
        flag.updated_at = now
    session.add(
        RiskEvent(
            timestamp=now,
            type="kill_switch",
            severity="critical" if engaged else "info",
            context={"reason": reason},
            action_taken="engaged" if engaged else "disengaged",
        )
    )
    if commit:
        await session.commit()
    return flag
