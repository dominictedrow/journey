"""Fix 1 regression — live equity is read from on-chain collateral, not pinned at the config
default of 0.

Before this fix, `risk_live_starting_equity` defaulted to 0, so live `total_equity` was 0, every
%-of-equity exposure cap resolved to 0, and `cap_by_notional` denied every live BUY — live
trading was dead-on-arrival. These tests assert the live equity baseline now comes from the
account's real collateral balance so a live BUY clears the portfolio exposure cap.
"""

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.portfolio.service import current_cash, current_equity
from app.risk.engine import build_context
from app.risk.models import RiskLimits, TradeIntent
from app.risk.rules.limits import PortfolioExposureRule


def _stub_balance(monkeypatch, value: Decimal | None) -> None:
    async def _read(*_a, **_k):
        return value

    monkeypatch.setattr("app.portfolio.service.read_collateral_balance", _read)


async def test_live_equity_reads_onchain_before_any_snapshot(
    session: AsyncSession, monkeypatch
) -> None:
    _stub_balance(monkeypatch, Decimal("2500"))

    assert await current_equity(session, "live") == Decimal("2500.000000")
    assert await current_cash(session, "live") == Decimal("2500.000000")


async def test_live_equity_falls_back_to_config_when_chain_read_fails(
    session: AsyncSession, monkeypatch
) -> None:
    _stub_balance(monkeypatch, None)  # RPC unconfigured / read failed

    # No snapshot yet → config baseline (risk_live_starting_equity default 0). Fail-open, not a
    # crash: the exposure cap is conservative rather than nonsensical.
    assert await current_equity(session, "live") == get_settings().risk_live_starting_equity


async def test_live_exposure_cap_is_nonzero_and_allows_a_buy(
    session: AsyncSession, monkeypatch
) -> None:
    _stub_balance(monkeypatch, Decimal("2500"))
    intent = TradeIntent(
        source_wallet="0xabc",
        token_id="tok",
        side="BUY",
        size=Decimal("10"),
        price=Decimal("0.5"),
        mode="live",
    )

    ctx = await build_context(session, intent, "live")
    assert ctx.total_equity == Decimal("2500.000000")

    rule = PortfolioExposureRule(RiskLimits.from_settings(get_settings()))
    result = rule.evaluate(ctx, intent, intent.size)
    # cap = 2500 * 0.60 = 1500 of headroom; a $5 notional BUY clears it.
    assert result.action == "allow"
    assert result.size == intent.size
