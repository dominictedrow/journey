"""Pydantic models for /risk: kill switch, configured limits, and the risk-event feed."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from app.config import Settings


class RiskLimitsOut(BaseModel):
    """The configured caps the gate enforces (Decimals serialize as strings)."""

    max_total_exposure_pct: Decimal
    max_daily_loss: Decimal
    max_weekly_loss: Decimal
    max_monthly_loss: Decimal
    max_open_positions: int
    max_position_size_pct: Decimal
    max_per_market_exposure_pct: Decimal
    max_per_wallet_exposure_pct: Decimal
    min_market_liquidity: Decimal
    max_slippage_pct: Decimal
    max_copy_delay_seconds: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "RiskLimitsOut":
        return cls(
            max_total_exposure_pct=settings.risk_max_total_exposure_pct,
            max_daily_loss=settings.risk_max_daily_loss,
            max_weekly_loss=settings.risk_max_weekly_loss,
            max_monthly_loss=settings.risk_max_monthly_loss,
            max_open_positions=settings.risk_max_open_positions,
            max_position_size_pct=settings.risk_max_position_size_pct,
            max_per_market_exposure_pct=settings.risk_max_per_market_exposure_pct,
            max_per_wallet_exposure_pct=settings.risk_max_per_wallet_exposure_pct,
            min_market_liquidity=settings.risk_min_market_liquidity,
            max_slippage_pct=settings.risk_max_slippage_pct,
            max_copy_delay_seconds=settings.risk_max_copy_delay_seconds,
        )


class KillSwitchOut(BaseModel):
    engaged: bool
    reason: str | None
    updated_at: datetime | None


class KillSwitchUpdate(BaseModel):
    enabled: bool
    reason: str | None = None


class RiskEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    timestamp: datetime
    type: str
    severity: str
    context: dict | None
    action_taken: str | None


class RiskEventListOut(BaseModel):
    items: list[RiskEventOut]
    total: int
    limit: int
    offset: int


class RiskStatusOut(BaseModel):
    kill_switch: KillSwitchOut
    limits: RiskLimitsOut
    execution_mode: str
    live_armed: bool
    recent_events: list[RiskEventOut]
