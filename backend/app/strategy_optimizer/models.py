"""Module 6 — strategy specification and optimizer value types.

A `StrategySpec` is the portable definition of a copy-portfolio strategy: a selection rule
(top-N of a ranking category), a weighting scheme, a sizing model, and limits. It is stored
verbatim in `strategies.params`, so any adopted or recommended strategy can be re-resolved
into a `BacktestConfig` and replayed. `candidates.resolve_config` turns a spec + the latest
rankings into that config; the Phase-4 engine does the rest.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.backtesting.models import BacktestConfig, SizingMode

Weighting = Literal["equal", "score", "volatility"]


class StrategySpec(BaseModel):
    """Portable, re-runnable strategy definition (persisted to `strategies.params`).

    Selection: the top `top_n` wallets of the latest `category` ranking generation.
    Weighting scales each wallet's spend — `equal` (all 1), `score` (by composite score),
    `volatility` (by inverse metric volatility); weights are normalized to mean 1 so total
    capital deployment is comparable across schemes. Sizing/limits mirror `BacktestConfig`.
    """

    category: str = "overall"
    top_n: int = Field(default=10, ge=1)
    weighting: Weighting = "equal"
    sizing_mode: SizingMode = "proportional"
    sizing_value: Decimal = Field(default=Decimal("1.0"), gt=0)
    delay_seconds: int = Field(default=5, ge=0)
    slippage_pct: Decimal = Field(default=Decimal("0.005"), ge=0)
    fee_pct: Decimal = Field(default=Decimal("0"), ge=0)
    max_open_positions: int | None = Field(default=None, ge=1)
    max_position_size: Decimal | None = Field(default=None, gt=0)

    @field_validator("category")
    @classmethod
    def _lower_category(cls, v: str) -> str:
        return v.lower()

    def describe(self) -> str:
        """Human-readable one-liner used as the strategy's name."""
        return (
            f"Top-{self.top_n} {self.category} · {self.weighting}-weighted · "
            f"{self.sizing_mode} sizing"
        )


@dataclass
class CandidateResult:
    """One candidate evaluated against the trailing window."""

    spec: StrategySpec
    config: BacktestConfig
    metrics: dict
    objective_value: float | None
    passes_risk: bool
    wallets: int

    @property
    def name(self) -> str:
        return self.spec.describe()

    def summary(self) -> dict:
        """Compact, JSON-safe view for the rationale's candidate list."""
        return {
            "name": self.name,
            "wallets": self.wallets,
            "objective_value": self.objective_value,
            "passes_risk": self.passes_risk,
            "total_return": self.metrics.get("total_return"),
            "sharpe": self.metrics.get("sharpe"),
            "max_drawdown": self.metrics.get("max_drawdown"),
            "trades_executed": self.metrics.get("trades_executed"),
        }


@dataclass
class OptimizeResult:
    """Outcome of one optimizer run."""

    adopted: bool
    objective: str
    strategy_id: int | None = None
    best: CandidateResult | None = None
    active: CandidateResult | None = None
    candidates: list[CandidateResult] = field(default_factory=list)
    rationale: dict = field(default_factory=dict)
