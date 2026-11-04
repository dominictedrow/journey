import datetime as dt
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trader.rebalance.backtest import simulate
from trader.rebalance.config import ConfigError, Holdings, RebalanceConfig, Rules
from trader.rebalance.planner import Quote, plan_rebalance

TODAY = dt.date(2026, 11, 3)
CFG = RebalanceConfig({"SPY": 0.6, "IEF": 0.4}, Rules(min_trade_usd=50))


def quotes(spy=500.0, ief=100.0, date=TODAY, **extra):
    q = {"SPY": Quote(spy, date, spy), "IEF": Quote(ief, date, ief)}
    q.update({k: Quote(v, date, v) for k, v in extra.items()})
    return q


def trades(plan):
    return {(t.symbol, t.side): t for t in plan.trades}


# ---- config -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "targets,rules",
    [
        ({"SPY": 0.6, "IEF": 0.3}, Rules()),  # sums to 0.9
        ({"SPY": 1.2, "IEF": -0.2}, Rules()),  # negative = short
        ({"SPY": 0.5, "IEF": 0.5}, Rules(cash_target=0.1)),  # 1.1 with cash
        ({"SPY": 1.0}, Rules(band_abs=0.0)),
    ],
)
def test_bad_config_rejected(targets, rules):
    with pytest.raises(ConfigError):
        RebalanceConfig(targets, rules).validate()


def test_negative_holdings_rejected():
    with pytest.raises(ConfigError):
        Holdings(cash=100, shares={"SPY": -1}).validate()


# ---- planning ---------------------------------------------------------------------------
def test_within_band_no_trades():
    h = Holdings(cash=0, shares={"SPY": 12, "IEF": 40})  # 6000/4000 = 60/40
    plan = plan_rebalance(CFG, h, quotes(), TODAY)
    assert plan.status == "NO_TRADE_NEEDED" and not plan.trades


def test_drift_rebalances_to_target_and_conserves_cash():
    h = Holdings(cash=0, shares={"SPY": 14, "IEF": 30})  # 7000/3000 = 70/30
    plan = plan_rebalance(CFG, h, quotes(), TODAY)
    t = trades(plan)
    assert plan.status == "OK"
    assert plan.trades[0].side == "SELL"  # sells listed first
    assert t[("SPY", "SELL")].value == pytest.approx(1000)
    assert t[("IEF", "BUY")].value == pytest.approx(1000)
    assert plan.cash_after == pytest.approx(0, abs=1e-6)


def test_new_deposit_is_invested_at_target():
    plan = plan_rebalance(CFG, Holdings(cash=25_000, shares={}), quotes(), TODAY)
    t = trades(plan)
    assert t[("SPY", "BUY")].value == pytest.approx(15_000)
    assert t[("IEF", "BUY")].value == pytest.approx(10_000)


def test_whole_shares_round_down_and_never_overspend():
    cfg = replace(CFG, rules=Rules(fractional=False))
    plan = plan_rebalance(cfg, Holdings(cash=1_234, shares={}), quotes(spy=333.0, ief=97.0), TODAY)
    for tr in plan.trades:
        assert tr.shares == int(tr.shares)
    assert plan.cash_after >= 0
    assert sum(tr.value for tr in plan.trades) <= 1_234


def test_small_trades_dropped():
    cfg = replace(CFG, rules=Rules(band_abs=0.01, min_trade_usd=500))
    h = Holdings(cash=0, shares={"SPY": 12.4, "IEF": 38})  # 6200/3800: $200 off
    plan = plan_rebalance(cfg, h, quotes(), TODAY)
    assert plan.status == "NO_TRADE_NEEDED"
    assert "below min_trade_usd" in plan.messages[-1]


def test_position_outside_allocation_is_sold_fully():
    h = Holdings(cash=0, shares={"SPY": 12, "IEF": 40, "GLD": 0.1})
    plan = plan_rebalance(CFG, h, quotes(GLD=200.0), TODAY)
    sell = trades(plan).get(("GLD", "SELL"))
    assert sell is not None and sell.shares == pytest.approx(0.1)  # even though only $20


def test_no_sell_uses_cash_on_most_underweight_first():
    h = Holdings(cash=1_000, shares={"SPY": 14, "IEF": 30})  # 7000/3000 + 1000 cash
    plan = plan_rebalance(CFG, h, quotes(), TODAY, no_sell=True)
    assert all(t.side == "BUY" for t in plan.trades)
    assert plan.trades[0].symbol == "IEF"
    assert plan.cash_after >= -1e-9


def test_drawdown_alert():
    h = Holdings(cash=0, shares={"SPY": 12, "IEF": 40}, peak_value=20_000)
    plan = plan_rebalance(CFG, h, quotes(), TODAY)
    assert any(m.startswith("ALERT") for m in plan.messages)


# ---- safety: default to NO TRADE ----------------------------------------------------------
def test_missing_price_halts():
    q = quotes()
    del q["IEF"]
    plan = plan_rebalance(CFG, Holdings(cash=1000, shares={}), q, TODAY)
    assert plan.status == "HALT" and not plan.trades


def test_stale_price_halts():
    old = TODAY - dt.timedelta(days=10)
    plan = plan_rebalance(CFG, Holdings(cash=1000, shares={}), quotes(date=old), TODAY)
    assert plan.status == "HALT" and "days old" in plan.messages[0]


def test_suspicious_move_halts():
    q = quotes()
    q["SPY"] = Quote(500.0, TODAY, prev_close=300.0)
    plan = plan_rebalance(CFG, Holdings(cash=1000, shares={}), q, TODAY)
    assert plan.status == "HALT" and "one-day move" in plan.messages[0]


# ---- rule comparison --------------------------------------------------------------------
def test_band_rule_keeps_weights_near_target():
    idx = pd.bdate_range("2020-01-01", periods=500)
    spy = pd.Series(np.linspace(100, 300, 500), index=idx)  # strong drift
    ief = pd.Series(100.0, index=idx)
    px = pd.DataFrame({"SPY": spy, "IEF": ief})
    _, n_none, _ = simulate(px, {"SPY": 0.6, "IEF": 0.4}, "none")
    eq, n_band, turn = simulate(px, {"SPY": 0.6, "IEF": 0.4}, "band", 0.05)
    assert n_none == 0 and n_band > 0 and turn > 0
    assert (eq > 0).all()
