"""Wallet-control, market-filter, timing, and loss-limit rules — the deny-only gates.

None of these adjust size: they either let the intent through or block it. Exits (SELL) pass
every one of them, so a disabled or losing book can always be wound down."""

from decimal import Decimal

from app.risk.models import RiskContext, RiskLimits, RuleResult, TradeIntent


class WalletControlRule:
    """Only copy wallets that are still copy-enabled; block new entries from disabled ones."""

    name = "wallet_control"

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY":
            return RuleResult.allow(size)
        if not ctx.wallet_copy_enabled:
            reason = ctx.wallet_disabled_reason or "copy disabled"
            return RuleResult.deny(f"wallet copy disabled: {reason}")
        return RuleResult.allow(size)


class MarketFilterRule:
    """Require a minimum market liquidity for new entries (unknown liquidity passes)."""

    name = "market_filter"

    def __init__(self, limits: RiskLimits) -> None:
        self.min_liquidity = limits.min_market_liquidity

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY":
            return RuleResult.allow(size)
        if intent.market_liquidity is not None and intent.market_liquidity < self.min_liquidity:
            return RuleResult.deny(
                f"market liquidity {intent.market_liquidity} < min {self.min_liquidity}"
            )
        return RuleResult.allow(size)


class TimingRule:
    """Drop entries that are too stale to be worth copying (exits are still mirrored)."""

    name = "timing"

    def __init__(self, limits: RiskLimits) -> None:
        self.max_delay = limits.max_copy_delay_seconds

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY":
            return RuleResult.allow(size)
        if intent.delay_seconds is not None and intent.delay_seconds > self.max_delay:
            return RuleResult.deny(
                f"copy delay {intent.delay_seconds:.0f}s > max {self.max_delay}s"
            )
        return RuleResult.allow(size)


class LossLimitRule:
    """Halt new entries once the trailing day/week/month loss breaches its cap."""

    name = "loss_limit"

    def __init__(self, limits: RiskLimits) -> None:
        self.caps = {
            "day": limits.max_daily_loss,
            "week": limits.max_weekly_loss,
            "month": limits.max_monthly_loss,
        }

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult:
        if intent.side != "BUY":
            return RuleResult.allow(size)
        for window, cap in self.caps.items():
            pnl = ctx.period_pnl.get(window)
            if pnl is not None and pnl <= -cap:
                return RuleResult.deny(f"{window} loss {pnl} breached limit {cap}")
        return RuleResult.allow(size)
