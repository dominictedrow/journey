"""IMOM01 rule tests on hand-built sessions (times in UTC; Jan = EST, so 16:00 ET = 21:00 UTC)."""

import numpy as np
import pandas as pd
import pytest

from trader.backtest.engine import Backtester
from trader.modes.base import Side
from trader.modes.imom import IntradayMomentum


def session_bars(day_paths: dict[str, list[tuple[str, float]]], tz="UTC") -> pd.DataFrame:
    """Continuous 1-min bars 14:00-21:30 UTC each day; price linearly interpolated via anchors."""
    frames = []
    for day, anchors in day_paths.items():
        idx = pd.date_range(f"{day} 14:00", f"{day} 21:30", freq="1min", tz=tz, inclusive="left")
        at = pd.DatetimeIndex([pd.Timestamp(f"{day} {t}", tz=tz) for t, _ in anchors])
        px = np.interp(idx.asi8, at.asi8, [p for _, p in anchors])
        px = px + 0.5 * (-1.0) ** np.arange(len(px))  # zero-mean wiggle so volatility > 0
        frames.append(
            pd.DataFrame(
                {"open": px, "high": px + 0.25, "low": px - 0.25, "close": px, "volume": 1.0},
                index=idx,
            )
        )
    out = pd.concat(frames)
    out.index.name = "ts"
    return out


def up_day(day, start, end):
    return [("14:00", start), ("20:59", end), ("21:29", end)]


@pytest.fixture
def two_days():
    # Day 1 ends at 4000 (16:00 ET close). Day 2 rallies to 4040 by 15:30 ET, then 4060 at 16:00.
    return session_bars(
        {
            "2024-01-09": up_day("2024-01-09", 3990, 4000),
            "2024-01-10": [("14:00", 4000), ("20:29", 4040), ("20:59", 4060), ("21:29", 4060)],
        }
    )


def test_signal_uses_prev_1600_close_and_1530_price(two_days):
    mode = IntradayMomentum()
    mode.prepare(two_days)
    i = np.flatnonzero(~np.isnan(mode.feat["signal"]))
    assert len(i) == 1
    assert two_days.index[i[0]] == pd.Timestamp("2024-01-10 20:29", tz="UTC")  # bar ends 15:30 ET
    c = two_days["close"]
    expected = c["2024-01-10 20:29"] / c["2024-01-09 20:59"] - 1  # 15:30 ET / prev 16:00 ET
    assert mode.feat["signal"][i[0]] == pytest.approx(expected)
    assert expected == pytest.approx(0.01, abs=5e-4)


def test_long_entry_after_1530_exit_at_1600(two_days):
    res = Backtester(_inst(), initial_equity=1e6).run(IntradayMomentum(), two_days)
    t = res.trades.iloc[0]
    assert t.side == Side.LONG
    assert t.entry_time == pd.Timestamp("2024-01-10 20:30", tz="UTC")
    assert t.exit_time == pd.Timestamp("2024-01-10 21:00", tz="UTC")
    assert t.exit_reason == "close_1600"
    assert t.pnl > 0


def test_negative_signal_goes_short():
    bars = session_bars(
        {
            "2024-01-09": up_day("2024-01-09", 3990, 4000),
            "2024-01-10": [("14:00", 4000), ("20:29", 3960), ("20:59", 3950), ("21:29", 3950)],
        }
    )
    res = Backtester(_inst(), initial_equity=1e6).run(IntradayMomentum(), bars)
    assert res.trades.iloc[0].side == Side.SHORT and res.trades.iloc[0].pnl > 0


def test_entry_lag_delays_fill(two_days):
    res = Backtester(_inst(), initial_equity=1e6).run(IntradayMomentum(entry_lag_min=5), two_days)
    assert res.trades.iloc[0].entry_time == pd.Timestamp("2024-01-10 20:35", tz="UTC")


def test_summer_dst_shifts_utc_times():
    bars = session_bars(
        {
            "2024-07-09": [("13:00", 5000), ("19:59", 5010), ("21:29", 5010)],
            "2024-07-10": [("13:00", 5010), ("19:29", 5050), ("19:59", 5060), ("21:29", 5060)],
        }
    )
    res = Backtester(_inst(), initial_equity=1e6).run(IntradayMomentum(), bars)
    t = res.trades.iloc[0]
    assert (t.entry_time.hour, t.entry_time.minute) == (19, 30)  # 15:30 EDT
    assert (t.exit_time.hour, t.exit_time.minute) == (20, 0)


def test_early_close_day_is_skipped():
    # 2024-11-29 (day after Thanksgiving) closes 13:00 ET.
    bars = session_bars(
        {
            "2024-11-27": up_day("2024-11-27", 5990, 6000),
            "2024-11-29": [("14:00", 6000), ("17:29", 6050), ("21:29", 6050)],
        }
    )
    mode = IntradayMomentum()
    mode.prepare(bars)
    assert np.isnan(mode.feat["signal"]).all()


def test_missing_signal_bar_means_no_trade(two_days):
    bars = two_days.drop(pd.Timestamp("2024-01-10 20:29", tz="UTC"))
    res = Backtester(_inst(), initial_equity=1e6).run(IntradayMomentum(), bars)
    assert res.trades.empty


def test_missing_close_bar_still_flattens(two_days):
    gap = (two_days.index >= "2024-01-10 20:55") & (two_days.index < "2024-01-10 21:10")
    res = Backtester(_inst(), initial_equity=1e6).run(IntradayMomentum(), two_days[~gap])
    t = res.trades.iloc[0]
    assert t.exit_reason == "close_1600"
    assert t.exit_time == pd.Timestamp("2024-01-10 21:11", tz="UTC")


def test_every_entry_has_stop_on_correct_side(two_days):
    res = Backtester(_inst(), initial_equity=1e6).run(IntradayMomentum(), two_days)
    t = res.trades.iloc[0]
    assert (t.entry_price - t.initial_stop) * t.side > 0


def test_onfh_agreement_filter():
    # ONFH (prev close -> 10:00 ET) negative, ROD positive: filtered out.
    bars = session_bars(
        {
            "2024-01-09": up_day("2024-01-09", 3990, 4000),
            "2024-01-10": [("14:00", 4000), ("14:59", 3980), ("20:29", 4040), ("21:29", 4040)],
        }
    )
    res = Backtester(_inst(), initial_equity=1e6).run(
        IntradayMomentum(require_onfh_agree=True), bars
    )
    assert res.trades.empty


def _inst():
    from trader.instruments import MES

    return MES
