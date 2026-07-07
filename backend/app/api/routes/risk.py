"""/risk endpoints — kill switch, configured limits, and the risk-event feed (Phase 5)."""

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.config import get_settings
from app.database.models import RiskEvent
from app.deps import DbSession
from app.risk.killswitch import get_state, set_engaged
from app.schemas.risk import (
    KillSwitchOut,
    KillSwitchUpdate,
    RiskEventListOut,
    RiskEventOut,
    RiskLimitsOut,
    RiskStatusOut,
)

router = APIRouter(prefix="/risk", tags=["risk"])

RECENT_EVENTS = 20


async def _kill_switch_out(session: DbSession) -> KillSwitchOut:
    flag = await get_state(session)
    return KillSwitchOut(
        engaged=bool(flag and flag.enabled),
        reason=flag.reason if flag else None,
        updated_at=flag.updated_at if flag else None,
    )


@router.get("", response_model=RiskStatusOut)
async def risk_status(session: DbSession) -> RiskStatusOut:
    """Kill-switch state, configured limits, and the latest risk events."""
    events = (
        await session.scalars(
            select(RiskEvent).order_by(RiskEvent.timestamp.desc()).limit(RECENT_EVENTS)
        )
    ).all()
    settings = get_settings()
    return RiskStatusOut(
        kill_switch=await _kill_switch_out(session),
        limits=RiskLimitsOut.from_settings(settings),
        execution_mode=settings.execution_mode,
        live_armed=settings.execution_mode == "live" and settings.live_trading_armed,
        recent_events=[RiskEventOut.model_validate(e) for e in events],
    )


@router.get("/events", response_model=RiskEventListOut)
async def list_events(
    session: DbSession,
    type: str | None = None,
    severity: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> RiskEventListOut:
    query = select(RiskEvent)
    if type is not None:
        query = query.where(RiskEvent.type == type)
    if severity is not None:
        query = query.where(RiskEvent.severity == severity)
    total = await session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = (
        await session.scalars(
            query.order_by(RiskEvent.timestamp.desc(), RiskEvent.id.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return RiskEventListOut(
        items=[RiskEventOut.model_validate(e) for e in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/kill-switch", response_model=KillSwitchOut)
async def kill_switch_status(session: DbSession) -> KillSwitchOut:
    return await _kill_switch_out(session)


@router.post("/kill-switch", response_model=KillSwitchOut)
async def set_kill_switch(session: DbSession, update: KillSwitchUpdate) -> KillSwitchOut:
    """Engage or disengage the global halt; logged as a `risk_event`."""
    await set_engaged(session, update.enabled, update.reason)
    return await _kill_switch_out(session)
