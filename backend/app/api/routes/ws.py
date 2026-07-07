"""/ws endpoints — live portfolio/risk feed for the dashboards (Phase 5).

`/ws/live` pushes a full state frame (portfolio summary, kill switch, recent copy orders and
risk events) on connect and then every `interval` seconds. Polling the DB keeps it correct
across processes — the copy worker writes, the API pushes — without a shared message bus.
"""

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.database.base import async_session_factory
from app.database.models import CopyOrder, RiskEvent
from app.portfolio.valuation import summarize
from app.risk.killswitch import get_state
from app.schemas.copy import CopyOrderOut
from app.schemas.portfolio import PortfolioPositionOut, PortfolioSummaryOut
from app.schemas.risk import KillSwitchOut, RiskEventOut

router = APIRouter(prefix="/ws", tags=["ws"])

MIN_INTERVAL = 0.5
MAX_INTERVAL = 30.0
RECENT = 10


async def _state_frame(mode: str) -> dict:
    async with async_session_factory() as session:
        summary = await summarize(session, mode)
        flag = await get_state(session)
        orders = (
            await session.scalars(
                select(CopyOrder)
                .where(CopyOrder.mode == mode)
                .order_by(CopyOrder.id.desc())
                .limit(RECENT)
            )
        ).all()
        events = (
            await session.scalars(
                select(RiskEvent)
                .order_by(RiskEvent.timestamp.desc(), RiskEvent.id.desc())
                .limit(RECENT)
            )
        ).all()

    portfolio = PortfolioSummaryOut(
        mode=summary["mode"],
        cash=summary["cash"],
        total_value=summary["total_value"],
        realized_pnl=summary["realized_pnl"],
        unrealized_pnl=summary["unrealized_pnl"],
        open_positions=summary["open_positions"],
        last_snapshot_at=summary["last_snapshot_at"],
        positions=[PortfolioPositionOut.model_validate(p) for p in summary["positions"]],
    )
    kill_switch = KillSwitchOut(
        engaged=bool(flag and flag.enabled),
        reason=flag.reason if flag else None,
        updated_at=flag.updated_at if flag else None,
    )
    return {
        "type": "state",
        "portfolio": portfolio.model_dump(mode="json"),
        "kill_switch": kill_switch.model_dump(mode="json"),
        "copy_orders": [CopyOrderOut.model_validate(o).model_dump(mode="json") for o in orders],
        "risk_events": [RiskEventOut.model_validate(e).model_dump(mode="json") for e in events],
    }


@router.websocket("/live")
async def live(websocket: WebSocket, mode: str = "paper", interval: float = 2.0) -> None:
    await websocket.accept()
    interval = min(max(interval, MIN_INTERVAL), MAX_INTERVAL)
    try:
        while True:
            await websocket.send_json(await _state_frame(mode))
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
