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
