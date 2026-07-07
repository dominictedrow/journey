"""Module 7 — backtest configuration and result types.

`BacktestConfig` is the single replay parameterization: it validates API input and is
serialized verbatim (JSON mode) into `backtests.params`, so a stored run can always be
re-executed. `BacktestResult` is the pure engine output persisted onto the row.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SizingMode = Literal["fixed", "percentage", "proportional"]


class BacktestConfig(BaseModel):
    """Deterministic replay parameters (same config + same fills = same result).

    Sizing semantics per mode:
    - fixed:        `sizing_value` is USDC notional spent per copied BUY.
    - percentage:   `sizing_value` is the fraction of current equity spent per copied BUY.
    - proportional: `sizing_value` scales the source trade's own notional.
    `weights` multiplies the spend per source wallet (missing wallets default to 1).
    """

    wallets: list[str] = Field(min_length=1)
    start: datetime | None = None
    end: datetime | None = None
    initial_capital: Decimal = Field(default=Decimal("1000"), gt=0)
    sizing_mode: SizingMode = "fixed"
    sizing_value: Decimal = Field(default=Decimal("50"), gt=0)
    weights: dict[str, Decimal] | None = None
    delay_seconds: int = Field(default=5, ge=0)
    slippage_pct: Decimal = Field(default=Decimal("0.005"), ge=0)
    fee_pct: Decimal = Field(default=Decimal("0"), ge=0)
    max_open_positions: int | None = Field(default=None, ge=1)
    max_position_size: Decimal | None = Field(default=None, gt=0)

    @field_validator("wallets")
    @classmethod
    def _lowercase_wallets(cls, v: list[str]) -> list[str]:
        return [w.lower() for w in v]

    @field_validator("weights")
    @classmethod
    def _lowercase_weight_keys(cls, v: dict[str, Decimal] | None) -> dict[str, Decimal] | None:
        return {k.lower(): w for k, w in v.items()} if v else v


@dataclass
class BacktestResult:
    metrics: dict
    equity_curve: list[dict]  # [{"t": iso8601, "equity": float}, ...]
    trade_log: list[dict]
