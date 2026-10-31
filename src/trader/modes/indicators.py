"""Causal indicators: value at row i uses rows <= i only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def atr(bars: pd.DataFrame, period: int) -> np.ndarray:
    """Wilder ATR. NaN until ``period`` true ranges are available."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    out = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return out.to_numpy()


def swing_pivots(bars: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Swing highs/lows as known in real time.

    A swing high at row j is above the k highs before it (strictly, so flat tops count once) and
    at least as high as the k after it. It only becomes known at row j + k, so the level is
    reported on that confirming row and is NaN elsewhere. Swing lows mirror this.
    """
    hi, lo = bars["high"], bars["low"]
    cand_h, cand_l = hi.shift(k), lo.shift(k)
    sh = cand_h.where((cand_h > hi.shift(k + 1).rolling(k).max()) & (cand_h >= hi.rolling(k).max()))
    sl = cand_l.where((cand_l < lo.shift(k + 1).rolling(k).min()) & (cand_l <= lo.rolling(k).min()))
    return sh.to_numpy(), sl.to_numpy()


def recent_pivots(levels: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the last ``n`` non-NaN ``levels`` up to and including that row.

    Returns (values, rows), each shaped (len, n), oldest first; NaN / -1 where fewer are known.
    """
    known = ~np.isnan(levels)
    pos = np.flatnonzero(known)
    count = np.cumsum(known)
    values = np.full((len(levels), n), np.nan)
    rows = np.full((len(levels), n), -1, dtype=np.int64)
    for c in range(n):
        idx = count - n + c
        ok = idx >= 0
        rows[ok, c] = pos[idx[ok]]
        values[ok, c] = levels[rows[ok, c]]
    return values, rows


def heikin_ashi(bars: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Heikin-Ashi (open, close). HA open is seeded with the first bar's (open + close) / 2."""
    o, h, lo, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    ha_close = (o + h + lo + c) / 4
    if len(o) == 0:
        return ha_close.copy(), ha_close
    # ha_open[t] = (ha_open[t-1] + ha_close[t-1]) / 2 is an EWM (alpha 0.5) of the lagged HA close.
    seed = np.r_[(o[0] + c[0]) / 2, ha_close[:-1]]
    ha_open = pd.Series(seed).ewm(alpha=0.5, adjust=False).mean().to_numpy()
    return ha_open, ha_close


def anchored_vwap(bars: pd.DataFrame, anchor: str, tz: str = "America/New_York") -> np.ndarray:
    """VWAP of typical price that resets daily at ``anchor`` (HH:MM, ``tz`` wall clock).

    Bars are labelled by their start, so the bar starting at the anchor opens a new session.
    NaN while the session has no volume.
    """
    h, m = (int(x) for x in anchor.split(":"))
    wall = bars.index.tz_convert(tz).tz_localize(None)
    session = (wall - pd.Timedelta(hours=h, minutes=m)).normalize()
    tp = (bars["high"] + bars["low"] + bars["close"]) / 3
    vol = bars["volume"]
    num = (tp * vol).groupby(session).cumsum()
    den = vol.groupby(session).cumsum()
    return (num / den.where(den > 0)).to_numpy()


def last_known(levels: np.ndarray) -> np.ndarray:
    """Forward-fill: the most recent non-NaN value up to each row."""
    return pd.Series(levels).ffill().to_numpy()


def htf_position(
    ltf_index: pd.DatetimeIndex, ltf_delta: pd.Timedelta, htf_end: pd.Series
) -> np.ndarray:
    """Row of the latest higher-timeframe bar complete by the end of each lower-timeframe bar."""
    ltf_end = (ltf_index + ltf_delta).asi8
    return np.searchsorted(pd.DatetimeIndex(htf_end).asi8, ltf_end, side="right") - 1


def take(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """``arr[pos]`` with NaN where ``pos`` is -1 (no higher-timeframe bar yet)."""
    out = arr[np.maximum(pos, 0)].astype(float)
    out[pos < 0] = np.nan
    return out
