"""TREND02 rule tests."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from trader.backtest.engine import Backtester
from trader.instruments import EURUSD
from trader.modes.base import BarContext, Enter, Exit, PositionView, Side
from trader.modes.trend import TrendEnsemble


def ctx_for(mode: TrendEnsemble, i: int, side: Side = Side.FLAT, n: int = 10) -> BarContext:
    arr = np.full(n, 100.0)
    idx = pd.date_range("2024-01-03 15:00", periods=n, freq="15min", tz="UTC")  # a Wednesday
    pos = PositionView(side=side, units=1, entry_price=100.0, stop=90.0 if side == 1 else 110.0)
    return BarContext(
        i=i,
        time=idx[i] + pd.Timedelta("15min"),
        open=arr,
        high=arr,
        low=arr,
        close=arr,
        volume=arr,
        index=idx,
        position=pos,
        equity=1e5,
    )


def set_feat(mode, votes, weekend=None):
    n = len(votes)
    mode.feat = {
        "votes": np.array(votes, dtype=float),
        "stop_dist": np.full(n, 2.0),
        "weekend": np.zeros(n, bool) if weekend is None else np.array(weekend),
    }


def test_unanimous_vote_enters_with_stop():
    m = TrendEnsemble()
    set_feat(m, [3, -3, 2])
    (long,) = m.on_bar(ctx_for(m, 0))
    assert isinstance(long, Enter) and long.side is Side.LONG and long.stop == 98.0
    (short,) = m.on_bar(ctx_for(m, 1))
    assert short.side is Side.SHORT and short.stop == 102.0
    assert m.on_bar(ctx_for(m, 2)) == []  # 2 of 3 is not unanimous


def test_min_agree_two_of_three():
    m = TrendEnsemble(min_agree=2)
    set_feat(m, [1, 2])  # net +1 = a 2-vs-1 split; net +2 needs two up, one flat
    assert m.on_bar(ctx_for(m, 0)) == []
    assert m.on_bar(ctx_for(m, 1))[0].side is Side.LONG


def test_holds_while_vote_favours_then_exits_and_reverses():
    m = TrendEnsemble()
    set_feat(m, [1, 0, -3])
    assert m.on_bar(ctx_for(m, 0, Side.LONG)) == []  # still net long: hold
    (ex,) = m.on_bar(ctx_for(m, 1, Side.LONG))
    assert isinstance(ex, Exit)
    ex, rev = m.on_bar(ctx_for(m, 2, Side.LONG))
    assert isinstance(ex, Exit) and rev.side is Side.SHORT


def test_weekend_flat_and_no_entry():
    m = TrendEnsemble()
    set_feat(m, [3, 3], weekend=[True, True])
    assert isinstance(m.on_bar(ctx_for(m, 0, Side.LONG))[0], Exit)
    assert m.on_bar(ctx_for(m, 1)) == []


def test_weekend_window_is_friday_1545_et():
    idx = pd.date_range("2024-01-05 20:00", "2024-01-05 21:30", freq="15min", tz="UTC")  # Fri EST
    bars = make_bars([(1.1, 1.1, 1.1, 1.1)] * len(idx), index=idx)
    m = TrendEnsemble()
    m.prepare(bars)
    ends = (idx + pd.Timedelta("15min")).tz_convert("America/New_York").strftime("%H:%M")
    flagged = dict(zip(ends, m.feat["weekend"], strict=True))
    assert not flagged["15:30"] and flagged["15:45"] and flagged["16:45"]


def test_votes_are_nan_during_warmup_then_signed():
    n = 32 * 16 + 50
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="15min", tz="UTC")
    px = 1.10 + 0.0001 * np.arange(n)
    bars = make_bars(list(zip(px, px + 2e-5, px - 2e-5, px, strict=True)), index=idx)
    m = TrendEnsemble()
    m.prepare(bars)
    v = m.feat["votes"]
    assert np.isnan(v[: 32 * 16 - 1]).all()
    assert (v[32 * 16 :] == 3).all()


def test_uptrend_backtest_goes_long_and_flattens_friday():
    # Steady uptrend over two weeks of 1-minute EURUSD bars.
    idx = pd.date_range("2024-01-01 00:00", "2024-01-19 21:00", freq="1min", tz="UTC")
    idx = idx[idx.dayofweek < 5]
    px = 1.05 + 0.03 * np.linspace(0, 1, len(idx)) + 2e-5 * (-1.0) ** np.arange(len(idx))
    bars = make_bars(list(zip(px, px + 3e-5, px - 3e-5, px, strict=True)), index=idx)
    res = Backtester(EURUSD, initial_equity=1e6).run(TrendEnsemble(speeds=[1, 4]), bars)
    assert len(res.trades) >= 1
    assert (res.trades.side == 1).all()
    fri = res.trades[res.trades.exit_reason == "weekend"]
    assert len(fri) >= 1
    assert (fri.exit_time.dt.tz_convert("America/New_York").dt.dayofweek == 4).all()
    assert (res.trades.entry_price > res.trades.initial_stop).all()


@pytest.mark.parametrize("speeds", [[1], [4], [16]])
def test_single_speed_variants_build(speeds):
    m = TrendEnsemble(speeds=speeds)
    assert m._need() == 1
