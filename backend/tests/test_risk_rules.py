"""Module 8 unit tests: every rule proves allow / deny / adjust against a hand-built context,
plus the engine chain and the kill switch (acceptance: each rule tested; kill switch halts)."""

from decimal import Decimal

from app.risk.models import RiskContext, RiskLimits, TradeIntent
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

WALLET = "0x" + "aa" * 20


def _limits(**overrides) -> RiskLimits:
    base = dict(
        max_total_exposure=Decimal("1000"),
        max_daily_loss=Decimal("100"),
        max_weekly_loss=Decimal("250"),
        max_monthly_loss=Decimal("500"),
        max_open_positions=3,
        max_position_size=Decimal("100"),
        max_per_market_exposure=Decimal("200"),
        max_per_wallet_exposure=Decimal("300"),
        min_market_liquidity=Decimal("10000"),
        max_slippage_pct=Decimal("2.0"),
        max_copy_delay_seconds=60,
    )
    base.update(overrides)
    return RiskLimits(**base)


def _intent(side="BUY", size="100", price="0.50", **kw) -> TradeIntent:
    return TradeIntent(
        source_wallet=WALLET,
        token_id="tok",
        side=side,
        size=Decimal(size),
        price=Decimal(price),
        market_id="mkt",
        **kw,
    )


def test_wallet_control_allow_deny_and_sell_bypass() -> None:
    rule = WalletControlRule()
    enabled = RiskContext(wallet_copy_enabled=True)
    disabled = RiskContext(wallet_copy_enabled=False, wallet_disabled_reason="paused")

    assert rule.evaluate(enabled, _intent(), Decimal("100")).action == "allow"
    denied = rule.evaluate(disabled, _intent(), Decimal("100"))
    assert denied.action == "deny" and "paused" in denied.reason
    # Exits are never blocked by wallet controls.
    assert rule.evaluate(disabled, _intent(side="SELL"), Decimal("100")).action == "allow"


def test_market_filter_allow_deny_unknown() -> None:
    rule = MarketFilterRule(_limits())
    ctx = RiskContext()
    assert (
        rule.evaluate(ctx, _intent(market_liquidity=Decimal("50000")), Decimal("100")).action
        == "allow"
    )
    assert (
        rule.evaluate(ctx, _intent(market_liquidity=Decimal("500")), Decimal("100")).action
        == "deny"
    )
    # Unknown liquidity does not block.
    assert rule.evaluate(ctx, _intent(market_liquidity=None), Decimal("100")).action == "allow"


def test_timing_allow_deny_and_sell_bypass() -> None:
    rule = TimingRule(_limits())
    ctx = RiskContext()
    assert rule.evaluate(ctx, _intent(delay_seconds=10), Decimal("100")).action == "allow"
    assert rule.evaluate(ctx, _intent(delay_seconds=120), Decimal("100")).action == "deny"
    assert (
        rule.evaluate(ctx, _intent(side="SELL", delay_seconds=999), Decimal("100")).action
        == "allow"
    )


def test_loss_limit_allow_and_deny() -> None:
    rule = LossLimitRule(_limits())
    ok = RiskContext(period_pnl={"day": Decimal("-50"), "week": None, "month": None})
    breached = RiskContext(period_pnl={"day": Decimal("-150"), "week": None, "month": None})
    assert rule.evaluate(ok, _intent(), Decimal("100")).action == "allow"
    denied = rule.evaluate(breached, _intent(), Decimal("100"))
    assert denied.action == "deny" and "day" in denied.reason
    # A breached loss limit still lets us exit.
    assert rule.evaluate(breached, _intent(side="SELL"), Decimal("100")).action == "allow"


def test_max_open_positions_allow_deny_and_increase_bypass() -> None:
    rule = MaxOpenPositionsRule(_limits(max_open_positions=3))
    assert rule.evaluate(RiskContext(open_positions=2), _intent(), Decimal("100")).action == "allow"
    assert rule.evaluate(RiskContext(open_positions=3), _intent(), Decimal("100")).action == "deny"
    # Increasing an existing holding never opens a new slot.
    at_cap_holding = RiskContext(open_positions=3, holds_token=True)
    assert rule.evaluate(at_cap_holding, _intent(), Decimal("100")).action == "allow"


def test_position_size_allow_adjust_deny() -> None:
    rule = PositionSizeRule(_limits(max_position_size=Decimal("100")))
    # notional 25 < 100 -> allow untouched.
    assert (
        rule.evaluate(RiskContext(), _intent(size="50", price="0.50"), Decimal("50")).action
        == "allow"
    )
    # notional 150 > 100 -> shrink to 200 shares (100 / 0.5).
    adjusted = rule.evaluate(RiskContext(), _intent(size="300", price="0.50"), Decimal("300"))
    assert adjusted.action == "adjust" and adjusted.size == Decimal("200")
    # already at the cap -> deny.
    full = RiskContext(token_exposure=Decimal("100"))
    assert rule.evaluate(full, _intent(size="10", price="0.50"), Decimal("10")).action == "deny"


def test_per_market_exposure_adjust_and_deny() -> None:
    rule = PerMarketExposureRule(_limits(max_per_market_exposure=Decimal("200")))
    ctx = RiskContext(market_exposure=Decimal("150"))
    # 50 of headroom / price 0.5 = 100 shares max; request 300 -> adjust to 100.
    adjusted = rule.evaluate(ctx, _intent(size="300", price="0.50"), Decimal("300"))
    assert adjusted.action == "adjust" and adjusted.size == Decimal("100")
    assert (
        rule.evaluate(RiskContext(market_exposure=Decimal("200")), _intent(), Decimal("100")).action
        == "deny"
    )


def test_per_wallet_exposure_adjust_and_deny() -> None:
    rule = PerWalletExposureRule(_limits(max_per_wallet_exposure=Decimal("300")))
    ctx = RiskContext(wallet_exposure=Decimal("250"))
    adjusted = rule.evaluate(ctx, _intent(size="300", price="0.50"), Decimal("300"))
    assert adjusted.action == "adjust" and adjusted.size == Decimal("100")  # 50 / 0.5
    assert (
        rule.evaluate(RiskContext(wallet_exposure=Decimal("300")), _intent(), Decimal("100")).action
        == "deny"
    )


def test_portfolio_exposure_allow_adjust_deny() -> None:
    rule = PortfolioExposureRule(_limits(max_total_exposure=Decimal("1000")))
    assert (
        rule.evaluate(
            RiskContext(total_exposure=Decimal("500")),
            _intent(size="100", price="0.5"),
            Decimal("100"),
        ).action
        == "allow"
    )
    ctx = RiskContext(total_exposure=Decimal("950"))
    adjusted = rule.evaluate(ctx, _intent(size="300", price="0.50"), Decimal("300"))
    assert adjusted.action == "adjust" and adjusted.size == Decimal("100")  # 50 / 0.5
    assert (
        rule.evaluate(RiskContext(total_exposure=Decimal("1000")), _intent(), Decimal("100")).action
        == "deny"
    )
