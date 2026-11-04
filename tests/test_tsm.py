import numpy as np
import pandas as pd

from trader.modes.base import BarContext, Enter, Exit, MoveStop, PositionView, Side
from trader.modes.tsm import TimeSeriesMomentum, month_end_mask, session_dates


def daily_index(dates):
    # Daily bars start 17:00 ET the evening before the session date.
    starts = [pd.Timestamp(d) - pd.Timedelta(days=1) + pd.Timedelta(hours=17) for d in dates]
    return pd.DatetimeIndex(starts).tz_localize("America/New_York").tz_convert("UTC")


def test_session_dates_roll_at_17_et():
    idx = daily_index(["2024-03-01"])
    assert session_dates(idx)[0] == pd.Timestamp("2024-03-01")


def test_month_end_is_last_weekday():
    dates = pd.bdate_range("2024-01-25", "2024-02-06")
    mask = month_end_mask(daily_index(dates))
    picked = [d for d, m in zip(dates, mask, strict=True) if m]
    assert picked == [pd.Timestamp("2024-01-31")]


def test_missing_month_end_falls_back_to_first_session_of_next_month():
    dates = [
        d for d in pd.bdate_range("2024-05-27", "2024-06-05") if d != pd.Timestamp("2024-05-31")
    ]
    mask = month_end_mask(daily_index(dates))
    picked = [d for d, m in zip(dates, mask, strict=True) if m]
    assert picked == [pd.Timestamp("2024-06-03")]


def make_mode(ret, sigma_m=0.02):
    mode = TimeSeriesMomentum()
    n = len(ret)
    mode.feat = {"ret": np.array(ret, float), "sigma_m": np.full(n, sigma_m),
                 "decide": np.ones(n, bool)}  # fmt: skip
    return mode


FLAT_POS = PositionView()


def ctx(i, close, position=FLAT_POS):
    n = len(close)
    arr = np.array(close, float)
    return BarContext(
        i=i, time=pd.Timestamp("2024-01-31", tz="UTC"),
        open=arr, high=arr, low=arr, close=arr, volume=np.ones(n),
        index=pd.date_range("2024-01-01", periods=n, tz="UTC"), position=position, equity=1e5,
    )  # fmt: skip


def test_enters_on_sign_with_3_sigma_stop():
    intents = make_mode([0.05]).on_bar(ctx(0, [100.0]))
    assert len(intents) == 1 and isinstance(intents[0], Enter)
    assert intents[0].side is Side.LONG
    assert intents[0].stop == 100.0 - 3 * 0.02 * 100.0


def test_flip_exits_then_reverses():
    pos = PositionView(side=Side.LONG, stop=90.0)
    intents = make_mode([-0.05]).on_bar(ctx(0, [100.0], pos))
    assert isinstance(intents[0], Exit) and intents[0].reason == "flip"
    assert isinstance(intents[1], Enter) and intents[1].side is Side.SHORT
    assert intents[1].stop == 100.0 + 6.0


def test_unchanged_signal_only_tightens_stop():
    tight = PositionView(side=Side.LONG, stop=95.0)
    assert make_mode([0.05]).on_bar(ctx(0, [100.0], tight)) == []  # new stop 94 is looser
    loose = PositionView(side=Side.LONG, stop=90.0)
    intents = make_mode([0.05]).on_bar(ctx(0, [100.0], loose))
    assert intents == [MoveStop(94.0, "monthly")]


def test_no_signal_goes_flat():
    pos = PositionView(side=Side.SHORT, stop=110.0)
    assert make_mode([np.nan]).on_bar(ctx(0, [100.0], pos)) == [Exit("no_signal")]
    assert make_mode([np.nan]).on_bar(ctx(0, [100.0])) == []
