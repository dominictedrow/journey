"""Exchange sessions in UTC: NYSE regular hours, holidays and early closes (exchange_calendars)."""

from __future__ import annotations

from functools import lru_cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

ET = "America/New_York"


@lru_cache(maxsize=1)
def _nyse() -> xcals.ExchangeCalendar:
    return xcals.get_calendar("XNYS")


def nyse_sessions(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """NYSE regular sessions overlapping [start, end]; columns open/close as UTC timestamps."""
    cal = _nyse()
    lo = max(pd.Timestamp(start).tz_convert(None).normalize(), cal.first_session)
    hi = min(pd.Timestamp(end).tz_convert(None).normalize(), cal.last_session)
    sched = cal.schedule.loc[lo:hi, ["open", "close"]]
    return pd.DataFrame(
        {"open": sched["open"].dt.tz_convert("UTC"), "close": sched["close"].dt.tz_convert("UTC")}
    )


def rth_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True for 1-minute bars whose start lies inside an NYSE regular session."""
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    sessions = nyse_sessions(index[0], index[-1])
    opens = sessions["open"].to_numpy(dtype="datetime64[ns]")
    closes = sessions["close"].to_numpy(dtype="datetime64[ns]")
    ts = index.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    k = np.searchsorted(opens, ts, side="right") - 1
    valid = k >= 0
    mask = np.zeros(len(ts), dtype=bool)
    mask[valid] = ts[valid] < closes[k[valid]]
    return mask
