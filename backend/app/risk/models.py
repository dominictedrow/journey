"""Module 8 — risk gate value types.

A `TradeIntent` is the copy engine's request to trade; `RiskEngine.check` runs it through
the kill switch and the ordered rules and returns a `RiskDecision` (allow / deny / adjust).
Rules are pure functions of a `RiskContext` (the portfolio state the gate needs) plus the
running size, so each rule is unit-testable in isolation against a hand-built context.
`RiskLimits` freezes the configured caps into a plain object the rules read from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal

from app.config import Settings

Side = Literal["BUY", "SELL"]
RuleAction = Literal["allow", "deny", "adjust"]


@dataclass
class TradeIntent:
    """A desire to mirror one source-wallet fill as our own trade."""

    source_wallet: str
    token_id: str
    side: Side
    size: Decimal  # shares we want to trade
    price: Decimal  # reference (source fill) price, in (0, 1)
    market_id: str | None = None
    source_trade_id: int | None = None
    delay_seconds: float | None = None  # how stale this copy is at decision time
    market_liquidity: Decimal | None = None
    mode: str = "paper"

    @property
    def notional(self) -> Decimal:
        return self.size * self.price


@dataclass
class RiskLimits:
    """The configured caps, frozen into a plain object the rules read (USDC unless noted)."""

    # Every exposure cap below is a fraction of current equity, not a flat dollar figure —
    # see config.py's risk limits section for the nesting rationale.
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
    def from_settings(cls, settings: Settings) -> RiskLimits:
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


@dataclass
class RiskContext:
    """The portfolio state the gate evaluates against (built from the DB by the engine)."""

    total_exposure: Decimal = Decimal(0)
    total_equity: Decimal = Decimal(0)  # current portfolio equity — the exposure cap's basis
    open_positions: int = 0
    token_exposure: Decimal = Decimal(0)  # our notional already held in the intent's token
    market_exposure: Decimal = Decimal(0)
    wallet_exposure: Decimal = Decimal(0)
    holds_token: bool = False  # do we already hold this token (so a BUY only increases it)
    wallet_copy_enabled: bool = True
    wallet_disabled_reason: str | None = None
    # Mark-to-market equity change over each trailing window; None when no baseline snapshot.
    period_pnl: Mapping[str, Decimal | None] = field(default_factory=dict)


@dataclass
class RuleResult:
    action: RuleAction
    size: Decimal
    reason: str | None = None

    @classmethod
    def allow(cls, size: Decimal) -> RuleResult:
        return cls("allow", size)

    @classmethod
    def deny(cls, reason: str) -> RuleResult:
        return cls("deny", Decimal(0), reason)

    @classmethod
    def adjust(cls, size: Decimal, reason: str) -> RuleResult:
        return cls("adjust", size, reason)


@dataclass
class RiskDecision:
    allow: bool
    adjusted_size: Decimal
    reason: str | None = None
    triggered_rule: str | None = None
    adjustments: list[dict] = field(default_factory=list)
