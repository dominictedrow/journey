"""System endpoints: health/liveness."""

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.config import get_settings

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    version: str
    execution_mode: str
    live_armed: bool


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        execution_mode=settings.execution_mode,
        # Live only trades when the mode is live AND armed; surfaced so operators can see the gate.
        live_armed=settings.execution_mode == "live" and settings.live_trading_armed,
    )
