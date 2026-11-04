"""REV01: short-horizon mean reversion (research/sources/INSTITUTIONAL_STRATEGY_CANDIDATES.md).

Source #18 (Robeco, "Reversing the trend of short-term reversal") concerns monthly
cross-sectional stock reversal. It motivates the idea only. Every rule here is our
pre-registered ADAPTATION, fixed before the first run. REV01 was also queued because TREND02
lost gross of costs, so count it in the multiple-testing log as informed by that result.

- z = (close - anchor) / sigma over ``lookback`` bars; anchor = SMA, or a rolling-VWAP proxy
  (CFD/FX tick volume only, D-020).
- Fade: long when z <= -entry_z, short when z >= entry_z.
- Exit when z reverts through +/-exit_z, after ``max_bars``, or at the stop: ``stop_sigma``
  sigmas beyond the entry price.
- Trend filter: skip entries when Kaufman's efficiency ratio over ``er_mult`` x lookback bars
  exceeds ``er_max`` (directional, not choppy, market).
- Session: entries only between ``entry_start`` and ``entry_end`` ET, flat at ``flat_time`` ET,
  weekdays only. Times apply to the decision time (bar end).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trader.data.resample import ET
from trader.modes.base import BarContext, Enter, Exit, Intent, Mode, Provenance, Side


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


class ShortTermReversion(Mode):
    name = "rev01"
    version = "0.1.0"
    provenance = Provenance.ADAPTATION
    instruments = ("USA500IDXUSD", "EURUSD")
    timeframe = "5min"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.timeframe = self.params["timeframe"]  # per-variant timeframe

    @classmethod
    def default_params(cls) -> dict:
        return {
            "timeframe": "5min",
            "lookback": 20,
            "anchor": "sma",
            "entry_z": 2.0,
            "exit_z": 0.0,
            "stop_sigma": 1.5,
            "max_bars": 12,
            "er_mult": 4,
            "er_max": 0.3,
            "entry_start": "09:45",
            "entry_end": "15:30",
            "flat_time": "15:55",
            "allow_short": True,
        }

    def prepare(self, bars: pd.DataFrame) -> None:
        p = self.params
        n = p["lookback"]
        close = bars["close"]
        if p["anchor"] == "sma":
            anchor = close.rolling(n).mean()
        elif p["anchor"] == "vwap":
            vol = bars["volume"]
            den = vol.rolling(n).sum()
            anchor = (close * vol).rolling(n).sum() / den.where(den > 0)
        else:
            raise ValueError(f"anchor must be 'sma' or 'vwap', got {p['anchor']!r}")
        sigma = close.rolling(n).std(ddof=0)
        self.feat["z"] = ((close - anchor) / sigma.where(sigma > 0)).to_numpy()
        self.feat["sigma"] = sigma.to_numpy()

        m = n * p["er_mult"]
        travel = close.diff().abs().rolling(m).sum()
        er = (close - close.shift(m)).abs() / travel.where(travel > 0)
        self.feat["er"] = er.to_numpy()

        end_et = (bars.index + pd.Timedelta(self.timeframe)).tz_convert(ET)
        mins = np.asarray(end_et.hour * 60 + end_et.minute)
        weekday = np.asarray(end_et.dayofweek < 5)
        start, stop, flat = (_minutes(p[k]) for k in ("entry_start", "entry_end", "flat_time"))
        self.feat["entry_ok"] = weekday & (mins >= start) & (mins <= stop)
        self.feat["flat"] = weekday & (mins >= flat)
        # Decisions only while a position may exist: entry window through the flat bar.
        self.feat["active"] = weekday & (mins >= start) & (mins <= flat)

    def decision_mask(self) -> np.ndarray:
        # Timestamp-only. A position opened in the window is flattened by ``flat_time`` the
        # same day, so no decision outside [entry_start, flat_time] is ever needed. The
        # first bar at/after flat_time is included even if bars around it are missing.
        active = self.feat["active"].copy()
        after_flat = self.feat["flat"] & ~active
        first_after = after_flat & ~np.r_[False, after_flat[:-1]]
        return active | first_after

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        p = self.params
        i = ctx.i
        pos = ctx.position.side
        z = self.feat["z"][i]
        if pos is not Side.FLAT:
            if self.feat["flat"][i] or not self.feat["active"][i]:
                return [Exit("flat_time")]
            if ctx.position.bars_held >= p["max_bars"]:
                return [Exit("time")]
            # Long exits once z climbs back to +exit_z, short once it falls to -exit_z.
            if math.isfinite(z) and z * pos.value >= p["exit_z"]:
                return [Exit("reverted")]
            return []
        if not self.feat["entry_ok"][i] or not math.isfinite(z):
            return []
        er = self.feat["er"][i]
        if p["er_max"] is not None and (not math.isfinite(er) or er > p["er_max"]):
            return []
        if z <= -p["entry_z"]:
            side = Side.LONG
        elif z >= p["entry_z"] and p["allow_short"]:
            side = Side.SHORT
        else:
            return []
        dist = p["stop_sigma"] * self.feat["sigma"][i]
        stop = ctx.close[i] - side.value * dist
        return [Enter(side, stop=stop, reason=f"z={z:+.2f}")]
