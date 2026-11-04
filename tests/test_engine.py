import math

import pandas as pd
import pytest

from tests.conftest import Scripted, make_bars, random_walk
from trader.backtest.engine import Backtester
from trader.instruments import MES
from trader.modes.base import Enter, Exit, MoveStop, Side
from trader.risk import RiskConfig

FLAT = (100.0, 100.5, 99.5, 100.0)


def run(inst, bars, script, **kw):
    mode = Scripted(script)
    return Backtester(inst, **kw).run(mode, bars), mode


def test_entry_fills_at_next_bar_open(free_mes):
    bars = make_bars([FLAT, FLAT, (101.0, 101.5, 100.5, 101.0), FLAT, FLAT])
    res, _ = run(free_mes, bars, {1: [Enter(Side.LONG, stop=95.0)]})
    t = res.trades.iloc[0]
    assert t.entry_time == bars.index[2]
    assert t.entry_price == 101.0
    assert t.exit_reason == "end_of_data"


def test_mode_never_sees_future_bars(free_mes):
    bars = random_walk(50)
    _, mode = run(free_mes, bars, {})
    for ctx in mode.seen:
        assert ctx.time == bars.index[ctx.i] + pd.Timedelta("1min")
        assert ctx.time <= bars.index[-1]  # last bar is never decided on: no bar left to fill


def test_stop_fill_intrabar_and_costs():
    bars = make_bars([FLAT, FLAT, (100.0, 100.5, 97.0, 98.0), FLAT])
    res, _ = run(MES, bars, {0: [Enter(Side.LONG, stop=98.0)]})
    t = res.trades.iloc[0]
    entry = 100.0 + MES.half_spread + MES.slippage
    exit_ = 98.0 - MES.half_spread - MES.slippage
    assert t.entry_price == pytest.approx(entry)
    assert t.exit_price == pytest.approx(exit_)
    assert t.exit_reason == "stop"
    expected = (
        exit_ - entry
    ) * t.units * MES.point_value - 2 * t.units * MES.commission_per_unit_side
    assert t.pnl == pytest.approx(expected)


def test_gap_through_stop_fills_at_open(free_mes):
    bars = make_bars([FLAT, FLAT, (90.0, 91.0, 89.0, 90.5), FLAT])
    res, _ = run(free_mes, bars, {0: [Enter(Side.LONG, stop=98.0)]})
    assert res.trades.iloc[0].exit_price == 90.0


def test_stop_wins_when_bar_hits_both(free_mes):
    bars = make_bars([FLAT, FLAT, (100.0, 110.0, 90.0, 100.0), FLAT])
    res, _ = run(free_mes, bars, {0: [Enter(Side.LONG, stop=98.0, target=105.0)]})
    t = res.trades.iloc[0]
    assert (t.exit_reason, t.exit_price) == ("stop", 98.0)


def test_short_target(free_mes):
    bars = make_bars([FLAT, FLAT, (100.0, 100.5, 94.0, 95.0), FLAT])
    res, _ = run(free_mes, bars, {0: [Enter(Side.SHORT, stop=102.0, target=96.0)]})
    t = res.trades.iloc[0]
    assert (t.exit_reason, t.exit_price, t.side) == ("target", 96.0, -1)
    assert t.pnl > 0


@pytest.mark.parametrize(
    ("intent", "reason"),
    [
        (Enter(Side.LONG, stop=None), "no_stop"),
        (Enter(Side.LONG, stop=float("nan")), "no_stop"),
        (Enter(Side.LONG, stop=101.0), "stop_wrong_side"),
        (Enter(Side.SHORT, stop=99.0), "stop_wrong_side"),
        (Enter(Side.LONG, stop=99.9), "stop_too_tight"),
    ],
)
def test_risk_layer_vetoes_bad_stops(free_mes, intent, reason):
    bars = make_bars([FLAT] * 4)
    res, _ = run(free_mes, bars, {0: [intent]})
    assert res.trades.empty
    assert res.rejections == {reason: 1}


def test_sizing_matches_policy_formula():
    bars = make_bars([FLAT] * 4)
    res, _ = run(MES, bars, {0: [Enter(Side.LONG, stop=90.0)]}, initial_equity=100_000)
    fill = 100.0 + MES.half_spread + MES.slippage
    per_unit = (fill - 90.0) * MES.point_value + MES.round_trip_cost_per_unit()
    assert res.trades.iloc[0].units == math.floor(100_000 * 0.0025 / per_unit)


def test_below_min_size_is_no_trade():
    bars = make_bars([FLAT] * 4)
    res, _ = run(MES, bars, {0: [Enter(Side.LONG, stop=10.0)]}, initial_equity=1_000)
    assert res.trades.empty and res.rejections == {"below_min_size": 1}


def test_stale_data_blocks_entry_but_not_exit(free_mes):
    idx = pd.DatetimeIndex(
        [
            "2024-01-02 14:30",
            "2024-01-02 14:31",
            "2024-01-02 16:00",
            "2024-01-02 16:01",
            "2024-01-02 16:02",
            "2024-01-02 18:00",
            "2024-01-02 18:01",
        ],
        tz="UTC",
        name="ts",
    )
    bars = make_bars([FLAT] * 7, index=idx)
    script = {1: [Enter(Side.LONG, stop=95.0)], 2: [Enter(Side.LONG, stop=95.0)], 4: [Exit("x")]}
    res, _ = run(free_mes, bars, script)
    assert res.rejections == {"stale_data": 1}
    assert len(res.trades) == 1
    t = res.trades.iloc[0]
    assert (t.entry_time, t.exit_time) == (idx[3], idx[5])  # exit fills at the delayed open


def test_move_stop_and_reversal(free_mes):
    rows = [FLAT, FLAT, FLAT, (100.0, 100.5, 98.5, 99.0), FLAT, FLAT, FLAT]
    bars = make_bars(rows)
    script = {
        0: [Enter(Side.LONG, stop=95.0)],
        1: [MoveStop(98.8), MoveStop(150.0)],  # second is on the wrong side: vetoed
        4: [Enter(Side.SHORT, stop=105.0)],
        5: [Exit("flip"), Enter(Side.LONG, stop=95.0)],
    }
    res, _ = run(free_mes, bars, script)
    assert res.rejections == {"move_stop:stop_wrong_side": 1}
    assert list(res.trades.exit_reason) == ["stop", "flip", "end_of_data"]
    assert res.trades.iloc[0].exit_price == 98.8
    assert list(res.trades.side) == [1, -1, 1]


def test_already_in_position_rejected(free_mes):
    bars = make_bars([FLAT] * 5)
    res, _ = run(
        free_mes, bars, {0: [Enter(Side.LONG, stop=95.0)], 1: [Enter(Side.LONG, stop=95.0)]}
    )
    assert len(res.trades) == 1 and res.rejections == {"already_in_position": 1}


def test_daily_loss_breaker():
    bars = make_bars([FLAT, FLAT, (100.0, 100.0, 50.0, 50.0), FLAT, FLAT, FLAT])
    risk = RiskConfig(risk_per_trade=0.05, daily_loss_limit=0.01)
    res, _ = run(
        MES,
        bars,
        {0: [Enter(Side.LONG, stop=99.0)], 3: [Enter(Side.LONG, stop=45.0)]},
        risk=risk,
        initial_equity=100_000,
    )
    assert len(res.trades) == 1
    assert res.rejections == {"daily_loss_breaker": 1}


def test_daily_equity_reconciles_with_ledger():
    bars = random_walk(4 * 24 * 60, seed=3)
    script = {}
    for i in range(10, len(bars) - 10, 500):
        price = bars.close.iloc[i]
        script[i] = [Enter(Side.LONG, stop=price - 15.0)]
        script[i + 200] = [Exit("time")]
    res, _ = run(MES, bars, script, initial_equity=50_000)
    assert len(res.trades) > 5
    assert res.daily_equity.iloc[-1] == pytest.approx(50_000 + res.trades.pnl.sum())
    assert res.trades.cost.gt(0).all()


def test_mode_can_tolerate_scheduled_halt(free_mes):
    idx = pd.DatetimeIndex(
        ["2024-01-02 14:30", "2024-01-02 14:31", "2024-01-02 15:30", "2024-01-02 15:31"],
        tz="UTC",
        name="ts",
    )
    bars = make_bars([FLAT] * 4, index=idx)

    class Patient(Scripted):
        max_entry_delay = "2h"

    res = Backtester(free_mes).run(Patient({1: [Enter(Side.LONG, stop=95.0)]}), bars)
    assert res.rejections == {} and res.trades.iloc[0].entry_time == idx[2]


def test_observed_half_spread_replaces_constant():
    bars = make_bars([FLAT, FLAT, (100.0, 100.5, 97.0, 98.0), FLAT])
    hs = [0.0, 0.5, 1.0, 0.0]
    mode = Scripted({0: [Enter(Side.LONG, stop=98.0)]})
    res = Backtester(MES, cost_multiplier=2.0).run(mode, bars, half_spread=hs)
    t = res.trades.iloc[0]
    # entry on bar 1 pays its own half-spread (x2), stop on bar 2 pays bar 2's (x2); slippage x2.
    assert t.entry_price == pytest.approx(100.0 + 2 * 0.5 + 2 * MES.slippage)
    assert t.exit_price == pytest.approx(98.0 - 2 * 1.0 - 2 * MES.slippage)


def test_observed_half_spread_validated():
    bars = make_bars([FLAT, FLAT])
    with pytest.raises(ValueError):
        Backtester(MES).run(Scripted({}), bars, half_spread=[0.1])
