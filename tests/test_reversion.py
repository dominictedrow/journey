"""REV01 rule tests."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars
from trader.backtest.engine import Backtester
from trader.data.resample import resample_bars
from trader.instruments import MES
from trader.modes.base import BarContext, Enter, Exit, PositionView, Side
from trader.modes.reversion import ShortTermReversion


def ctx_for(i: int, side: Side = Side.FLAT, held: int = 0, n: int = 10) -> BarContext:
    arr = np.full(n, 100.0)
    idx = pd.date_range("2024-01-03 15:00", periods=n, freq="5min", tz="UTC")
    pos = PositionView(side=side, units=1, entry_price=100.0, stop=95.0, bars_held=held)
    return BarContext(
        i=i,
        time=idx[i],
        open=arr,
        high=arr,
        low=arr,
        close=arr,
        volume=arr,
        index=idx,
        position=pos,
        equity=1e5,
    )


def set_feat(m, z, er=0.1, entry_ok=True, flat=False, active=True):
    n = len(z)
    m.feat = {
        "z": np.array(z, float),
        "sigma": np.full(n, 2.0),
        "er": np.full(n, er),
        "entry_ok": np.full(n, entry_ok),
        "flat": np.full(n, flat),
        "active": np.full(n, active),
    }


def test_fades_extremes_with_stop_beyond_entry():
    m = ShortTermReversion()
    set_feat(m, [-2.5, 2.5, 1.0])
    (long,) = m.on_bar(ctx_for(0))
    assert isinstance(long, Enter) and long.side is Side.LONG and long.stop == 97.0  # 1.5 x 2.0
    (short,) = m.on_bar(ctx_for(1))
    assert short.side is Side.SHORT and short.stop == 103.0
    assert m.on_bar(ctx_for(2)) == []


def test_trend_filter_and_window_block_entries():
    m = ShortTermReversion()
    set_feat(m, [-3.0], er=0.5)
    assert m.on_bar(ctx_for(0)) == []
    set_feat(m, [-3.0], entry_ok=False)
    assert m.on_bar(ctx_for(0)) == []
    m = ShortTermReversion(er_max=None)
    set_feat(m, [-3.0], er=0.9)
    assert m.on_bar(ctx_for(0))[0].side is Side.LONG


@pytest.mark.parametrize(
    ("side", "z", "held", "flat", "reason"),
    [
        (Side.LONG, 0.1, 1, False, "reverted"),
        (Side.SHORT, -0.1, 1, False, "reverted"),
        (Side.LONG, -1.0, 12, False, "time"),
        (Side.LONG, -1.0, 1, True, "flat_time"),
    ],
)
def test_exits(side, z, held, flat, reason):
    m = ShortTermReversion()
    set_feat(m, [z], flat=flat)
    (ex,) = m.on_bar(ctx_for(0, side, held))
    assert isinstance(ex, Exit) and ex.reason == reason


def test_holds_until_reverted():
    m = ShortTermReversion()
    set_feat(m, [-1.0])
    assert m.on_bar(ctx_for(0, Side.LONG, 3)) == []


def test_zscore_matches_manual():
    bars = make_bars([(p, p + 0.5, p - 0.5, p) for p in np.r_[np.full(19, 100.0), 104.0]])
    m = ShortTermReversion(lookback=20)
    m.prepare(bars)
    c = bars.close.to_numpy()
    assert m.feat["z"][-1] == pytest.approx((c[-1] - c.mean()) / c.std())
    assert np.isnan(m.feat["z"][:19]).all()


def test_session_window_in_new_york_time():
    idx = pd.date_range("2024-01-03 14:30", "2024-01-03 21:00", freq="5min", tz="UTC")  # EST
    bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * len(idx), index=idx)
    m = ShortTermReversion()
    m.prepare(bars)
    end = (idx + pd.Timedelta("5min")).tz_convert("America/New_York").strftime("%H:%M")
    ok = dict(zip(end, m.feat["entry_ok"], strict=True))
    flat = dict(zip(end, m.feat["flat"], strict=True))
    assert not ok["09:40"] and ok["09:45"] and ok["15:30"] and not ok["15:35"]
    assert not flat["15:50"] and flat["15:55"]
    mask = m.decision_mask()
    act = dict(zip(end, mask, strict=True))
    assert act["09:45"] and act["15:55"] and not act["09:40"]
    assert act["16:00"] and not act["16:05"]  # first bar after flat_time kept as a safety net


def test_mask_keeps_first_bar_after_flat_when_bars_missing():
    idx = pd.date_range("2024-01-03 14:30", "2024-01-03 20:40", freq="5min", tz="UTC")
    idx = idx.append(pd.DatetimeIndex(["2024-01-03 21:30", "2024-01-03 21:35"], tz="UTC"))
    bars = make_bars([(100.0, 100.5, 99.5, 100.0)] * len(idx), index=idx)
    m = ShortTermReversion()
    m.prepare(bars)
    assert m.decision_mask()[-2] and not m.decision_mask()[-1]


def test_timeframe_and_anchor_params():
    m = ShortTermReversion(timeframe="15min", anchor="vwap")
    assert m.timeframe == "15min" and m.describe()["timeframe"] == "15min"
    with pytest.raises(ValueError):
        ShortTermReversion(anchor="median").prepare(make_bars([(1.0, 1.1, 0.9, 1.0)] * 30))


def test_backtest_oscillation_trades_intraday_only():
    # One RTH week of 1-minute bars oscillating around 4000 with occasional spikes.
    days = pd.bdate_range("2024-01-08", "2024-01-12")
    idx = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(
                    f"{d.date()} 14:30",
                    f"{d.date()} 21:00",
                    freq="1min",
                    tz="UTC",
                    inclusive="left",
                ).asi8
                for d in days
            ]
        )
    ).tz_localize("UTC")
    t = np.arange(len(idx))
    px = 4000 + 3 * np.sin(t / 7.0) + np.where(t % 97 == 50, -12.0, 0.0)
    bars = make_bars(list(zip(px, px + 0.5, px - 0.5, px, strict=True)), index=idx)
    res = Backtester(MES, initial_equity=1e6).run(ShortTermReversion(er_max=None), bars)
    assert len(res.trades) > 3
    exit_et = res.trades.exit_time.dt.tz_convert("America/New_York")
    assert (exit_et.dt.hour * 60 + exit_et.dt.minute <= 16 * 60).all()
    assert set(res.trades.exit_reason) <= {"reverted", "time", "flat_time", "stop"}
    five = resample_bars(bars, "5min")
    assert len(five) > 0
