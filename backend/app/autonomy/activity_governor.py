"""Module D — the activity governor.

Guarantees the "≥1 copy trade/day" floor the full-autonomy remodel targets without a human
touching a dial: a beat task checks trailing trade frequency and, if the book has gone quiet,
widens the wallet-discovery funnel a notch — the autopromote budget, the tracked-wallet cap,
and the optimizer's top-N selection — each capped at a hard multiple of its static default so
the governor can never run away to an unbounded cohort no matter how many quiet days it sees.
Once activity recovers to target it decays those same levers a notch back toward their static
defaults, so the funnel self-corrects instead of ratcheting permanently wide. Every change is
logged as a `RiskEvent` for the audit trail (`activity_governor_widened` / `..._narrowed`); the
override values themselves live in `runtime_overrides` (see `app.autonomy.overrides`) since
`Settings` can't be mutated live.
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai_advisor import narrate
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


async def _narrow(
    session: AsyncSession,
    key: str,
    baseline: int,
    step: int,
    reason: str,
) -> int | None:
    """Decay `key`'s effective value one `step` back toward `baseline`, never below it. Returns
    the new value, or None if the lever is already at (or below) baseline — a no-op, not an error.

    This is the counterpart to `_widen`: without it the governor is a one-way ratchet that leaves
    the funnel permanently widened after activity recovers. Decaying back keeps it self-correcting
    while staying fully autonomous — the floor is always the static settings default."""
    current = await get_override(session, key, baseline)
    if current <= baseline:
        return None
    new_value = max(current - step, baseline)
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

    autopromote_base = settings.autopromote_per_run
    tracked_base = settings.max_tracked_wallets
    # Baseline 0 (no extra top-N by default), capped at the open-position ceiling candidate_specs
    # already clamps every top_n against — widening past it would have no effect anyway.
    top_n_ceiling = settings.risk_max_open_positions

    if trailing < TARGET_TRADES_PER_DAY:
        # Book has gone quiet — widen the funnel a notch toward its hard ceiling.
        direction = "widened"
        reason = f"trailing 24h trades ({trailing}) below target ({TARGET_TRADES_PER_DAY})"
        result[AUTOPROMOTE_KEY] = await _widen(
            session,
            AUTOPROMOTE_KEY,
            autopromote_base,
            int(autopromote_base * MAX_MULTIPLIER),
            max(1, autopromote_base // 2),
            reason,
        )
        result[MAX_TRACKED_KEY] = await _widen(
            session,
            MAX_TRACKED_KEY,
            tracked_base,
            int(tracked_base * MAX_MULTIPLIER),
            max(1, tracked_base // 2),
            reason,
        )
        result[TOP_N_EXTRA_KEY] = await _widen(
            session,
            TOP_N_EXTRA_KEY,
            0,
            top_n_ceiling,
            max(1, top_n_ceiling // 4),
            reason,
        )
    else:
        # Target met — decay any prior widening a notch back toward baseline so the funnel
        # doesn't stay permanently inflated once activity recovers. No-op at baseline.
        direction = "narrowed"
        reason = (
            f"trailing 24h trades ({trailing}) at/above target "
            f"({TARGET_TRADES_PER_DAY}); decaying funnel toward baseline"
        )
        result[AUTOPROMOTE_KEY] = await _narrow(
            session, AUTOPROMOTE_KEY, autopromote_base, max(1, autopromote_base // 2), reason
        )
        result[MAX_TRACKED_KEY] = await _narrow(
            session, MAX_TRACKED_KEY, tracked_base, max(1, tracked_base // 2), reason
        )
        result[TOP_N_EXTRA_KEY] = await _narrow(
            session, TOP_N_EXTRA_KEY, 0, max(1, top_n_ceiling // 4), reason
        )

    levers = {AUTOPROMOTE_KEY, MAX_TRACKED_KEY, TOP_N_EXTRA_KEY}
    if any(result[key] is not None for key in levers):
        context = {
            "trailing_trades": trailing,
            "target": TARGET_TRADES_PER_DAY,
            "direction": direction,
            AUTOPROMOTE_KEY: result[AUTOPROMOTE_KEY],
            MAX_TRACKED_KEY: result[MAX_TRACKED_KEY],
            TOP_N_EXTRA_KEY: result[TOP_N_EXTRA_KEY],
        }
        # Advisory only: annotate the change for the audit trail. `narrate` fails open (None on
        # any error), so the governor's decision above is unaffected by the LLM being down.
        context["ai_narrative"] = await narrate(_narration_prompt(context))
        session.add(
            RiskEvent(
                timestamp=now,
                type=f"activity_governor_{direction}",
                severity="info",
                context=context,
                action_taken=direction,
            )
        )
    await session.commit()
    return result


def _narration_prompt(context: dict) -> str:
    """A compact prompt describing the widen/narrow decision for the advisory LLM to narrate."""
    direction = context.get("direction", "widened")
    if direction == "narrowed":
        premise = (
            "automated decision to narrow the wallet-discovery funnel back toward its baseline "
            "because copy-trade activity has recovered to target"
        )
    else:
        premise = (
            "automated decision to widen the wallet-discovery funnel because copy-trade activity "
            "fell below target"
        )
    return (
        "You are a trading-operations analyst. In 2-3 sentences, plainly explain the following "
        f"{premise}. Describe what changed and why; no caveats or recommendations.\n"
        f"trailing_24h_trades: {context['trailing_trades']}\n"
        f"target_trades_per_day: {context['target']}\n"
        f"new_autopromote_per_run: {context[AUTOPROMOTE_KEY]}\n"
        f"new_max_tracked_wallets: {context[MAX_TRACKED_KEY]}\n"
        f"new_optimizer_top_n_extra: {context[TOP_N_EXTRA_KEY]}"
    )
