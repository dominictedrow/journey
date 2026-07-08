"""Position- and exposure-cap rules — the size-adjusting gates.

Each caps a BUY's notional against a different slice of headroom (per-position, per-market,
per-wallet, whole-portfolio) and shrinks the order to fit, or denies when there is none.
`MaxOpenPositionsRule` is the one hard count limit: it blocks *new* positions past the cap
but lets existing ones grow. SELLs pass every rule here."""

from decimal import Decimal

from app.risk.models import RiskContext, RiskLimits, RuleResult, TradeIntent
from app.risk.rules.base import cap_by_notional


class PositionSizeRule:
    """Cap the total notional held in a single token at `max_position_size_pct` of equity."""

    name = "position_size"

    def __init__(self, limits: RiskLimits) -> None:
        self.pct = limits.max_position_size_pct

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY":
            return RuleResult.allow(size)
        cap = ctx.total_equity * self.pct
        remaining = cap - ctx.token_exposure
        return cap_by_notional(size, intent.price, remaining, "max_position_size_pct")


class MaxOpenPositionsRule:
    """Block opening a *new* position once we already hold the maximum count."""

    name = "max_open_positions"

    def __init__(self, limits: RiskLimits) -> None:
        self.cap = limits.max_open_positions

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY" or ctx.holds_token:
            return RuleResult.allow(size)  # increasing an existing position never opens a slot
        if ctx.open_positions >= self.cap:
            return RuleResult.deny(f"max open positions ({self.cap}) reached")
        return RuleResult.allow(size)


class PerMarketExposureRule:
    """Cap total notional in one market at `max_per_market_exposure_pct` of equity."""

    name = "per_market_exposure"

    def __init__(self, limits: RiskLimits) -> None:
        self.pct = limits.max_per_market_exposure_pct

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY":
            return RuleResult.allow(size)
        cap = ctx.total_equity * self.pct
        remaining = cap - ctx.market_exposure
        return cap_by_notional(size, intent.price, remaining, "max_per_market_exposure_pct")


class PerWalletExposureRule:
    """Cap total notional copied from one source wallet at `max_per_wallet_exposure_pct` of
    equity."""

    name = "per_wallet_exposure"

    def __init__(self, limits: RiskLimits) -> None:
        self.pct = limits.max_per_wallet_exposure_pct

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY":
            return RuleResult.allow(size)
        cap = ctx.total_equity * self.pct
        remaining = cap - ctx.wallet_exposure
        return cap_by_notional(size, intent.price, remaining, "max_per_wallet_exposure_pct")


class PortfolioExposureRule:
    """Cap the whole portfolio's open notional at `max_total_exposure_pct` of current equity."""

    name = "portfolio_exposure"

    def __init__(self, limits: RiskLimits) -> None:
        self.pct = limits.max_total_exposure_pct

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY":
            return RuleResult.allow(size)
        cap = ctx.total_equity * self.pct
        remaining = cap - ctx.total_exposure
        return cap_by_notional(size, intent.price, remaining, "max_total_exposure_pct")
