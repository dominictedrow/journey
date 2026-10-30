"""Causal resampling of 1-minute bars to higher timeframes.

Each output bar is labelled by its bucket START and carries ``end`` = bucket end. A bar is only
"known" once wall-clock time reaches ``end``; the engine enforces that. Empty buckets are dropped.

Daily bars use the 17:00 America/New_York roll shared by CME equity futures and spot FX, so a
daily bar starting 17:00 ET Sunday covers the Monday session.
"""

from __future__ import annotations

import pandas as pd

from trader.data.loaders import BAR_COLUMNS

ET = "America/New_York"
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def timeframe_delta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(timeframe)


def resample_bars(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Return HTF bars with columns BAR_COLUMNS + ``end`` (UTC bucket end)."""
    delta = timeframe_delta(timeframe)
    if delta == pd.Timedelta("1min"):
        out = bars[BAR_COLUMNS].copy()
        out["end"] = out.index + delta
        return out
    if delta == pd.Timedelta("1D"):
        # Shift the ET wall clock by +7h so the 17:00 ET roll lands on midnight, bucket by date,
        # then shift back. Wall-clock (naive) arithmetic keeps this correct across DST.
        wall = bars.index.tz_convert(ET).tz_localize(None)
        starts_wall = (wall + pd.Timedelta(hours=7)).normalize() - pd.Timedelta(hours=7)
        starts = starts_wall.tz_localize(ET)  # 17:00 is never ambiguous or nonexistent
        grouped = bars[BAR_COLUMNS].groupby(starts.tz_convert("UTC"))
        out = grouped.agg(_AGG)
        out.index = pd.DatetimeIndex(out.index, name="ts")
        # Bucket end = next 17:00 ET, which is 23 or 25 hours later across DST changes.
        start_et = out.index.tz_convert(ET)
        next_day = (start_et.tz_localize(None) + pd.Timedelta(days=1)).tz_localize(ET)
        out["end"] = next_day.tz_convert("UTC")
        return out
    if delta > pd.Timedelta("1D") or pd.Timedelta("1D") % delta != pd.Timedelta(0):
        raise ValueError(f"unsupported timeframe {timeframe!r}; must divide one day or equal 1D")
    out = bars[BAR_COLUMNS].resample(delta, label="left", closed="left").agg(_AGG)
    out = out.dropna(subset=["open"])
    out.index = pd.DatetimeIndex(out.index, name="ts")
    out["end"] = out.index + delta
    return out
