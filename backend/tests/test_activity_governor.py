"""Module D — the activity governor: widens the wallet funnel when trailing trade activity
has gone quiet, is a no-op when the ≥1 trade/day floor already holds, and never runs away past
a hard multiple of each lever's static default."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.autonomy.activity_governor import (
    AUTOPROMOTE_KEY,
    MAX_MULTIPLIER,
    MAX_TRACKED_KEY,
    TOP_N_EXTRA_KEY,
    run_activity_governor,
)
from app.autonomy.overrides import get_override
from app.config import Settings
from app.database.models import CopyOrder, RiskEvent

NOW = datetime(2026, 7, 8, 12, 0, tzinfo=UTC)


def _settings(**over) -> Settings:
    base = dict(autopromote_per_run=8, max_tracked_wallets=300, risk_max_open_positions=10)
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


async def test_governor_widens_when_quiet(session: AsyncSession) -> None:
    settings = _settings()

    result = await run_activity_governor(session, settings=settings, now=NOW)

    assert result["trailing_trades"] == 0
    assert result[AUTOPROMOTE_KEY] == 12  # 8 + step(4)
    assert result[MAX_TRACKED_KEY] == 450  # 300 + step(150)
    assert result[TOP_N_EXTRA_KEY] == 2  # 0 + step(max(1, 10//4))

    assert await get_override(session, AUTOPROMOTE_KEY, 0) == 12
    event = await session.scalar(
        select(RiskEvent).where(RiskEvent.type == "activity_governor_widened")
    )
    assert event is not None
    assert event.action_taken == "widened"
    assert event.context["trailing_trades"] == 0


async def test_governor_noop_when_activity_already_meets_target(session: AsyncSession) -> None:
    settings = _settings()
    session.add(
        CopyOrder(
            created_at=NOW - timedelta(hours=1),
            mode="paper",
            side="BUY",
            token_id="tok",
            requested_size=Decimal("10"),
            filled_size=Decimal("10"),
            fill_price=Decimal("0.5"),
            status="filled",
        )
    )
    await session.commit()

    result = await run_activity_governor(session, settings=settings, now=NOW)

    assert result["trailing_trades"] == 1
    assert result[AUTOPROMOTE_KEY] is None
    assert result[MAX_TRACKED_KEY] is None
    assert result[TOP_N_EXTRA_KEY] is None
    assert await session.scalar(select(RiskEvent)) is None
    assert await get_override(session, AUTOPROMOTE_KEY, settings.autopromote_per_run) == (
        settings.autopromote_per_run
    )


async def test_governor_bounded_widen_never_exceeds_max_multiplier(
    session: AsyncSession,
) -> None:
    settings = _settings()
    ceiling = int(settings.autopromote_per_run * MAX_MULTIPLIER)

    last_result: dict[str, int | None] = {}
    for _ in range(20):  # far more runs than needed to reach the ceiling
        last_result = await run_activity_governor(session, settings=settings, now=NOW)

    assert await get_override(session, AUTOPROMOTE_KEY, 0) == ceiling
    # Once capped, further quiet runs are a genuine no-op on this lever.
    assert last_result[AUTOPROMOTE_KEY] is None

    events = (
        await session.scalars(
            select(RiskEvent).where(RiskEvent.type == "activity_governor_widened")
        )
    ).all()
    # Widening stops once every lever is capped; not one event per call.
    assert len(events) < 20
