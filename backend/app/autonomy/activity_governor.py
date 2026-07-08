"""Module D — the activity governor.

Guarantees the "≥1 copy trade/day" floor the full-autonomy remodel targets without a human
touching a dial: a beat task checks trailing trade frequency and, if the book has gone quiet,
widens the wallet-discovery funnel a notch — the autopromote budget, the tracked-wallet cap,
and the optimizer's top-N selection — each capped at a hard multiple of its static default so
the governor can never run away to an unbounded cohort no matter how many quiet days it sees.
Every widen is logged as a `RiskEvent` for the audit trail; the widened values themselves live
in `runtime_overrides` (see `app.autonomy.overrides`) since `Settings` can't be mutated live.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.autonomy.overrides import get_override, set_override
from app.config import Settings, get_settings
from app.database.models import CopyOrder, RiskEvent

# ≥1 trade/day is the whole point of this module (the stated autonomy target); a lower
# trailing count triggers a widen.
TARGET_TRADES_PER_DAY = 1

# The governor never widens a lever past this multiple of its *static* settings default,
# however many consecutive quiet days it sees — the runaway guard.
MAX_MULTIPLIER = Decimal("5")

AUTOPROMOTE_KEY = "autopromote_per_run"
MAX_TRACKED_KEY = "max_tracked_wallets"
TOP_N_EXTRA_KEY = "optimizer_top_n_extra"


async def _trailing_trade_count(session: AsyncSession, now: datetime) -> int:
    cutoff = now - timedelta(days=1)
    return (
        await session.scalar(
            select(func.count())
            .select_from(CopyOrder)
            .where(CopyOrder.status.in_(("filled", "partial")), CopyOrder.created_at >= cutoff)
        )
        or 0
    )


async def _widen(
    session: AsyncSession,
    key: str,
    baseline: int,
    ceiling: int,
    step: int,
    reason: str,
) -> int | None:
    """Bump `key`'s effective value by one `step`, capped at `ceiling`. Returns the new value,
    or None if the lever is already at (or above) its cap — a no-op, not an error."""
    current = await get_override(session, key, baseline)
    if current >= ceiling:
        return None
    new_value = min(current + step, ceiling)
    if new_value == current:
        return None
    await set_override(session, key, new_value, reason, commit=False)
    return new_value


async def run_activity_governor(
    session: AsyncSession, *, settings: Settings | None = None, now: datetime | None = None
) -> dict[str, int | None]:
    """Widen the wallet funnel when trailing trade activity has gone quiet. Commits."""
    settings = settings or get_settings()
    now = now or datetime.now(UTC)
    trailing = await _trailing_trade_count(session, now)

    result: dict[str, int | None] = {
        "trailing_trades": trailing,
        AUTOPROMOTE_KEY: None,
        MAX_TRACKED_KEY: None,
        TOP_N_EXTRA_KEY: None,
    }
    if trailing >= TARGET_TRADES_PER_DAY:
        return result

    reason = f"trailing 24h trades ({trailing}) below target ({TARGET_TRADES_PER_DAY})"
    result[AUTOPROMOTE_KEY] = await _widen(
        session,
        AUTOPROMOTE_KEY,
        settings.autopromote_per_run,
        int(settings.autopromote_per_run * MAX_MULTIPLIER),
        max(1, settings.autopromote_per_run // 2),
        reason,
    )
    result[MAX_TRACKED_KEY] = await _widen(
        session,
        MAX_TRACKED_KEY,
        settings.max_tracked_wallets,
        int(settings.max_tracked_wallets * MAX_MULTIPLIER),
        max(1, settings.max_tracked_wallets // 2),
        reason,
    )
    # Baseline 0 (no extra top-N by default), capped at the open-position ceiling candidate_specs
    # already clamps every top_n against — widening past it would have no effect anyway.
    result[TOP_N_EXTRA_KEY] = await _widen(
        session,
        TOP_N_EXTRA_KEY,
        0,
        settings.risk_max_open_positions,
        max(1, settings.risk_max_open_positions // 4),
        reason,
    )

    widened = {AUTOPROMOTE_KEY, MAX_TRACKED_KEY, TOP_N_EXTRA_KEY}
    if any(result[key] is not None for key in widened):
        session.add(
            RiskEvent(
                timestamp=now,
                type="activity_governor_widened",
                severity="info",
                context={
                    "trailing_trades": trailing,
                    "target": TARGET_TRADES_PER_DAY,
                    AUTOPROMOTE_KEY: result[AUTOPROMOTE_KEY],
                    MAX_TRACKED_KEY: result[MAX_TRACKED_KEY],
                    TOP_N_EXTRA_KEY: result[TOP_N_EXTRA_KEY],
                },
                action_taken="widened",
            )
        )
    await session.commit()
    return result
