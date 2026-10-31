"""J-T01..J-T03 (Joovier INTERPRETATION modes) and their indicators."""

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_bars, random_walk
from trader.backtest.engine import Backtester
from trader.data.resample import resample_bars
from trader.instruments import MES
from trader.modes.base import BarContext, Enter, Exit, PositionView, Provenance, Side
from trader.modes.indicators import (
    anchored_vwap,
    heikin_ashi,
    htf_position,
    recent_pivots,
    swing_pivots,
)
from trader.modes.joovier import SqueezeBreakout, SupportResistanceRetest, VwapHeikinAshi


def bars_from_closes(closes, start="2024-01-03 14:00", freq="1min"):
    rows = [(c, c + 0.5, c - 0.5, c) for c in closes]
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz="UTC", name="ts")
    return make_bars(rows, index=idx)


def ctx_for(i, close, side=Side.FLAT, held=0):
    arr = np.asarray(close, float)
    idx = pd.date_range("2024-01-03 15:00", periods=len(arr), freq="1min", tz="UTC")
    pos = PositionView(side=side, units=1, entry_price=100.0, stop=95.0, bars_held=held)
    return BarContext(i=i, time=idx[i], open=arr, high=arr, low=arr, close=arr, volume=arr,
                      index=idx, position=pos, equity=1e5)  # fmt: skip


def session(n, entry_ok=True, flat=False, active=True):
    return {
        "entry_ok": np.full(n, entry_ok),
        "flat": np.full(n, flat),
        "active": np.full(n, active),
    }


# ---- indicators --------------------------------------------------------------------------


def test_swing_pivots_confirm_late_and_count_flat_tops_once():
    bars = bars_from_closes([1, 2, 5, 5, 3, 2, 1, 4, 6])
    sh, sl = swing_pivots(bars, 2)
    # Flat top at rows 2-3: only row 2 qualifies (strictly above the bars before it),
    # confirmed two bars later at row 4.
    assert np.flatnonzero(~np.isnan(sh)).tolist() == [4]
    assert sh[4] == 5.5
    assert np.flatnonzero(~np.isnan(sl)).tolist() == [8]  # low at row 6, known at row 8
    assert sl[8] == 0.5


def test_recent_pivots_oldest_first():
    lv = np.array([np.nan, 3.0, np.nan, 5.0, np.nan, 4.0])
    vals, rows = recent_pivots(lv, 2)
    assert np.isnan(vals[0]).all() and rows[0].tolist() == [-1, -1]
    assert np.isnan(vals[2, 0]) and vals[2, 1] == 3.0
    assert vals[4].tolist() == [3.0, 5.0] and rows[4].tolist() == [1, 3]
    assert vals[5].tolist() == [5.0, 4.0]


def test_heikin_ashi_matches_recursion():
    bars = random_walk(50, seed=3)
    ha_open, ha_close = heikin_ashi(bars)
    o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    exp_open = [(o[0] + c[0]) / 2]
    for t in range(1, 50):
        exp_open.append((exp_open[-1] + (o[t - 1] + h[t - 1] + lo[t - 1] + c[t - 1]) / 4) / 2)
    assert ha_close == pytest.approx((o + h + lo + c) / 4)
    assert ha_open == pytest.approx(exp_open)


def test_anchored_vwap_resets_at_anchor():
    # 09:28-09:32 ET (EST): two bars before the 09:30 anchor, three after.
    bars = bars_from_closes([10, 20, 100, 102, 104], start="2024-01-03 14:28")
    bars["volume"] = [1.0, 1.0, 1.0, 3.0, 0.0]
    v = anchored_vwap(bars, "09:30")
    assert v[1] == pytest.approx(15.0)
    assert v[2] == pytest.approx(100.0)  # new session: earlier bars forgotten
    assert v[3] == pytest.approx((100 + 3 * 102) / 4)
    assert v[4] == v[3]  # zero volume leaves VWAP unchanged


def test_htf_bar_used_only_after_it_closes():
    bars = bars_from_closes(np.arange(130.0), start="2024-01-03 14:00")
    htf = resample_bars(bars, "1h")
    pos = htf_position(bars.index, pd.Timedelta("1min"), htf["end"])
    assert pos[58] == -1  # 14:58 bar ends 14:59: the 14:00 hour is still forming
    assert pos[59] == 0  # 14:59 bar ends 15:00: hour complete
    assert pos[119] == 1


# ---- J-T01 -------------------------------------------------------------------------------


def jt01_feat(m, n, touch_at=(), floor=95.0, brk=101.0, tgt=110.0, low=None, **sess):
    low = np.full(n, 98.0) if low is None else np.asarray(low, float)
    touch = np.zeros(n, bool)
    touch[list(touch_at)] = True
    m.feat = {
        **session(n, **sess),
        "touchL": touch, "touchS": np.zeros(n, bool),
        "floorL": np.full(n, floor), "floorS": np.full(n, np.nan),
        "tgtL": np.full(n, tgt), "tgtS": np.full(n, np.nan),
        "brkL": np.full(n, brk), "brkS": np.full(n, np.nan),
        "advL": low, "advS": np.full(n, -200.0),
        "closeL": np.zeros(n), "closeS": np.zeros(n),
    }  # fmt: skip
    m._arm = {Side.LONG: None, Side.SHORT: None}


def run_jt01(m, closes):
    m.feat["closeL"], m.feat["closeS"] = np.asarray(closes, float), -np.asarray(closes, float)
    return [m.on_bar(ctx_for(i, closes)) for i in range(len(closes))]


def test_jt01_needs_touch_then_later_break():
    m = SupportResistanceRetest()
    jt01_feat(m, 4, touch_at=[0], low=[96.0, 97.0, 99.0, 99.0])
    out = run_jt01(m, [102.0, 100.0, 102.0, 103.0])  # bar 0 already closes above brk: no entry
    assert out[0] == [] and out[1] == []
    (e,) = out[2]
    assert isinstance(e, Enter) and e.side is Side.LONG
    assert e.stop == 96.0 and e.target == 110.0  # stop at the retest low
    assert out[3] == []  # disarmed after entering


def test_jt01_no_trade_without_touch_or_after_zone_breaks():
    m = SupportResistanceRetest()
    jt01_feat(m, 3)
    assert run_jt01(m, [102.0, 102.0, 102.0]) == [[], [], []]
    m = SupportResistanceRetest()
    jt01_feat(m, 3, touch_at=[0])
    assert run_jt01(m, [99.0, 94.0, 102.0]) == [[], [], []]  # closed below the zone floor


def test_jt01_arm_expires_and_min_rr_filters():
    m = SupportResistanceRetest(arm_bars=2)
    jt01_feat(m, 4, touch_at=[0])
    assert run_jt01(m, [99.0, 99.0, 99.0, 102.0]) == [[], [], [], []]
    m = SupportResistanceRetest()
    jt01_feat(m, 2, touch_at=[0], tgt=104.0)  # reward 2 < risk 4
    assert run_jt01(m, [99.0, 102.0]) == [[], []]


def test_jt01_exits_only_at_day_flat():
    m = SupportResistanceRetest()
    jt01_feat(m, 1, flat=True)
    (ex,) = m.on_bar(ctx_for(0, [100.0], Side.LONG))
    assert isinstance(ex, Exit) and ex.reason == "flat_time"
    jt01_feat(m, 1)
    assert m.on_bar(ctx_for(0, [100.0], Side.LONG)) == []


def test_jt01_prepare_builds_zone_touch_from_closed_hours():
    # Hours 0-6 form an hourly swing low at hour 3 (k=3, known at the end of hour 6); price
    # then returns to it in hour 8.
    hourly = [110, 108, 106, 100, 106, 108, 110, 112, 100.2]
    closes = np.repeat(np.asarray(hourly, float), 60)
    bars = bars_from_closes(closes, start="2024-01-03 05:00")
    m = SupportResistanceRetest(zone_atr=0.25, atr_period=2)
    m.prepare(bars)
    hour = np.arange(len(closes)) // 60
    assert not m.feat["touchL"][hour < 8].any()
    assert m.feat["touchL"][hour == 8].all()
    assert np.isnan(m.feat["tgtL"][hour == 8]).all()  # no resistance pivot above yet


def test_jt01_short_side_mirrors_long():
    m = SupportResistanceRetest()
    n = 3
    m.feat = {
        **session(n),
        "touchL": np.zeros(n, bool), "touchS": np.array([True, False, False]),
        "floorL": np.full(n, np.nan), "floorS": np.full(n, -105.0),
        "tgtL": np.full(n, np.nan), "tgtS": np.full(n, -90.0),
        "brkL": np.full(n, np.nan), "brkS": np.full(n, -99.0),
        "advL": np.zeros(n), "advS": -np.array([104.0, 102.0, 101.0]),
    }  # fmt: skip
    m._arm = {Side.LONG: None, Side.SHORT: None}
    closes = [101.0, 100.0, 98.0]
    m.feat["closeL"], m.feat["closeS"] = np.array(closes), -np.array(closes)
    out = [m.on_bar(ctx_for(i, closes)) for i in range(n)]
    (e,) = out[2]
    assert e.side is Side.SHORT and e.stop == 104.0 and e.target == 90.0


# ---- J-T02 -------------------------------------------------------------------------------


def test_jt02_prepare_finds_triangle_break():
    # Hourly zigzag: highs 110.5 -> 106.5 (falling), lows 89.5 -> 93.5 (rising), then a break.
    closes = [100, 110, 100, 90, 100, 106, 100, 94, 98, 104]
    bars = bars_from_closes(closes, start="2024-01-03 05:00", freq="1h")
    m = SqueezeBreakout(timeframe="1h", pivot_k=1, swing_k=1, min_rr=0.0)
    m.prepare(bars)
    # At hour 9 the lines extrapolate to the next bar (x=10): upper 101.5, lower 96.5.
    assert not m.feat["sigL"].any()  # 104 is below the last swing high, 106.5
    bars.iloc[9] = [107.0, 107.5, 106.5, 107.0, 1.0]
    m.prepare(bars)
    assert m.feat["sigL"][9] and not m.feat["sigL"][:9].any()
    assert m.feat["stopL"][9] == 93.5 and m.feat["tgtL"][9] == 110.5


def jt02_feat(m, n, sig, pid, stop=95.0, tgt=110.0):
    m.feat = {
        **session(n),
        "sigL": np.asarray(sig, bool), "sigS": np.zeros(n, bool),
        "stopL": np.full(n, stop), "stopS": np.full(n, np.nan),
        "tgtL": np.full(n, tgt), "tgtS": np.full(n, np.nan),
        "pid": np.asarray(pid, float),
    }  # fmt: skip
    m._used = -1.0


def test_jt02_one_trade_per_pattern_and_rr_filter():
    m = SqueezeBreakout()
    jt02_feat(m, 3, [True, True, True], [5, 5, 6])
    out = [m.on_bar(ctx_for(i, [100.0] * 3)) for i in range(3)]
    assert isinstance(out[0][0], Enter) and out[0][0].stop == 95.0 and out[0][0].target == 110.0
    assert out[1] == []  # same pattern
    assert isinstance(out[2][0], Enter)  # new pattern
    m = SqueezeBreakout()
    jt02_feat(m, 1, [True], [5], tgt=103.0)  # reward 3 < risk 5
    assert m.on_bar(ctx_for(0, [100.0])) == []


# ---- J-T03 -------------------------------------------------------------------------------


def jt03_feat(m, n, **over):
    base = {
        **session(n),
        "sigL": np.zeros(n, bool), "sigS": np.zeros(n, bool),
        "stopL": np.full(n, 99.0), "stopS": np.full(n, 101.0),
        "ha_exitL": np.zeros(n, bool), "ha_exitS": np.zeros(n, bool),
        "vwap_exitL": np.zeros(n, bool), "vwap_exitS": np.zeros(n, bool),
    }  # fmt: skip
    m.feat = {**base, **{k: np.asarray(v) for k, v in over.items()}}


def test_jt03_entry_needs_signal_and_stop_below():
    m = VwapHeikinAshi()
    jt03_feat(m, 2, sigL=[True, True], stopL=[99.0, 100.0])
    (e,) = m.on_bar(ctx_for(0, [100.0, 100.0]))
    assert e.side is Side.LONG and e.stop == 99.0 and e.target is None
    assert m.on_bar(ctx_for(1, [100.0, 100.0])) == []  # stop not below price


@pytest.mark.parametrize(
    ("over", "held", "reason"),
    [
        ({"ha_exitL": [True]}, 1, "ha_flip"),
        ({"vwap_exitL": [True]}, 1, "vwap"),
        ({}, 30, "time"),
        ({"flat": [True]}, 1, "flat_time"),
    ],
)
def test_jt03_exits(over, held, reason):
    m = VwapHeikinAshi()
    jt03_feat(m, 1, **over)
    (ex,) = m.on_bar(ctx_for(0, [100.0], Side.LONG, held))
    assert isinstance(ex, Exit) and ex.reason == reason


def test_jt03_short_exit_uses_short_flags():
    m = VwapHeikinAshi()
    jt03_feat(m, 1, ha_exitL=[True])
    assert m.on_bar(ctx_for(0, [100.0], Side.SHORT, 1)) == []


def test_jt03_prepare_long_signal_after_clean_pullback():
    # Zigzag uptrend above VWAP (rising swing highs and lows), two clean bearish HA bars,
    # then a bullish HA bar whose low stays above its HA open (no lower wick).
    zig = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106]
    rows = [(c, c + 0.5, c - 0.5, c) for c in zig]
    rows += [(105.0, 105.0, 104.0, 104.2), (104.2, 104.2, 103.6, 103.8)]
    idx = pd.date_range("2024-01-03 14:30", periods=13, freq="1min", tz="UTC", name="ts")
    m = VwapHeikinAshi(swing_k=1, min_side_bars=5, min_clean=2)
    m.prepare(make_bars([*rows, (104.4, 106.5, 104.4, 106.3)], index=idx))
    assert m.feat["sigL"].tolist() == [False] * 12 + [True]
    assert m.feat["stopL"][-1] == 100.5 and not m.feat["sigS"].any()
    m.prepare(make_bars([*rows, (104.3, 106.5, 104.3, 106.3)], index=idx))  # lower wick
    assert not m.feat["sigL"].any()
    m = VwapHeikinAshi(swing_k=1, min_side_bars=5, min_clean=3)  # only two clean bears
    m.prepare(make_bars([*rows, (104.4, 106.5, 104.4, 106.3)], index=idx))
    assert not m.feat["sigL"].any()


# ---- whole-mode backtests ----------------------------------------------------------------


@pytest.mark.parametrize("cls", [SupportResistanceRetest, SqueezeBreakout, VwapHeikinAshi])
def test_modes_trade_intraday_with_stops_on_the_right_side(cls):
    bars = random_walk(60 * 24 * 30, seed=21, step=1.5)
    mode = cls()
    assert mode.provenance is Provenance.INTERPRETATION
    res = Backtester(MES, initial_equity=1e6).run(mode, bars)
    t = res.trades
    assert len(t) > 0
    exit_et = t.exit_time.dt.tz_convert("America/New_York")
    assert (exit_et.dt.hour * 60 + exit_et.dt.minute <= 16 * 60).all()
    risk = (t.entry_price - t.initial_stop) * t.side
    assert (risk > 0).all()
