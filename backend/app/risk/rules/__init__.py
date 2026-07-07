"""Ordered default rule chain for the risk gate.

Deny-only gates run first (cheap, no point sizing a doomed order), then the size-adjusting
caps from broadest headroom check to tightest. Each successive adjust sees the already-reduced
size, so the final `adjusted_size` respects every cap simultaneously."""

from app.risk.models import RiskLimits
from app.risk.rules.base import Rule
from app.risk.rules.controls import (
    LossLimitRule,
    MarketFilterRule,
    TimingRule,
    WalletControlRule,
)
from app.risk.rules.limits import (
    MaxOpenPositionsRule,
    PerMarketExposureRule,
    PerWalletExposureRule,
    PortfolioExposureRule,
    PositionSizeRule,
)

__all__ = [
    "Rule",
    "WalletControlRule",
    "MarketFilterRule",
    "TimingRule",
    "LossLimitRule",
    "MaxOpenPositionsRule",
    "PositionSizeRule",
    "PerMarketExposureRule",
    "PerWalletExposureRule",
    "PortfolioExposureRule",
    "default_rules",
]


def default_rules(limits: RiskLimits) -> list[Rule]:
    return [
        WalletControlRule(),
        MarketFilterRule(limits),
        TimingRule(limits),
        LossLimitRule(limits),
        MaxOpenPositionsRule(limits),
        PositionSizeRule(limits),
        PerMarketExposureRule(limits),
        PerWalletExposureRule(limits),
        PortfolioExposureRule(limits),
    ]
