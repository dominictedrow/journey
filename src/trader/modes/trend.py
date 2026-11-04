"""TREND02: multi-speed EMA trend ensemble (research/sources/INSTITUTIONAL_STRATEGY_CANDIDATES.md).

Sources #6 (Man AHL, fast vs slow EMA crossovers), #1/#2 (AQR) and #10 (CFM) document trend
following at daily-to-monthly horizons. Their full methods are NOT yet reviewed, and none of them
shows an intraday edge, so every parameter below is our pre-registered ADAPTATION:

- One vote per speed: sign(EMA_fast - EMA_slow) of the close, fast/slow = 8/32 bars of that
  speed. Speeds are multiples of the 15-minute base bar (1 = 15m, 4 = 60m, 16 = 4h), done as
  EMAs on 15-minute bars with spans multiplied by the speed.
- Speeds are equal-weighted votes, not an optimised blend (candidate definition).
- Enter when at least ``min_agree`` votes point the same way (default: all of them); exit when
  the net vote no longer favours the position, reversing if the opposite side qualifies.
- Catastrophe stop: ``stop_atr`` x ATR(15m) x sqrt(slowest speed), since a bar's range grows
  ~sqrt(time). Exits are signal-driven; the stop is protection (D-015).
- Flat for the weekend from Friday 15:45 ET: overnight/weekend risk limits are not approved
  (D-007), and spot FX closes Friday 17:00 ET.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trader.data.resample import ET
from trader.modes.base import BarContext, Enter, Exit, Intent, Mode, Provenance, Side
from trader.modes.indicators import atr


class TrendEnsemble(Mode):
    name = "trend02"
    version = "0.1.0"
    provenance = Provenance.ADAPTATION
    instruments = ("USA500IDXUSD", "EURUSD")
    timeframe = "15min"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "speeds": [1, 4, 16],
            "fast": 8,
            "slow": 32,
            "min_agree": None,  # None = unanimous
            "stop_atr": 3.0,
            "atr_period": 14,
            "flat_weekend": True,
            "allow_short": True,
        }

    def prepare(self, bars: pd.DataFrame) -> None:
        p = self.params
        close = bars["close"]
        votes = np.zeros(len(bars))
        for m in p["speeds"]:
            slow_span = p["slow"] * m
            fast = close.ewm(span=p["fast"] * m, adjust=False, min_periods=slow_span).mean()
            slow = close.ewm(span=slow_span, adjust=False, min_periods=slow_span).mean()
            votes = votes + np.sign((fast - slow).to_numpy())  # NaN during warm-up propagates
        self.feat["votes"] = votes
        self.feat["stop_dist"] = (
            p["stop_atr"] * atr(bars, p["atr_period"]) * math.sqrt(max(p["speeds"]))
        )
        # Decision time is the bar END; flatten from Friday 15:45 ET until the Sunday reopen.
        end_et = (bars.index + pd.Timedelta(self.timeframe)).tz_convert(ET)
        minutes = end_et.hour * 60 + end_et.minute
        friday_late = (end_et.dayofweek == 4) & (minutes >= 15 * 60 + 45)
        self.feat["weekend"] = np.asarray(friday_late | (end_et.dayofweek == 5))

    def _need(self) -> int:
        n = len(self.params["speeds"])
        return n if self.params["min_agree"] is None else int(self.params["min_agree"])

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        p = self.params
        i = ctx.i
        pos = ctx.position.side
        if p["flat_weekend"] and self.feat["weekend"][i]:
            return [Exit("weekend")] if pos is not Side.FLAT else []
        v = self.feat["votes"][i]
        if not math.isfinite(v):
            return []

        need = self._need()
        want = Side.FLAT
        if v >= need:
            want = Side.LONG
        elif v <= -need and p["allow_short"]:
            want = Side.SHORT

        out: list[Intent] = []
        if pos is not Side.FLAT:
            if v * pos.value > 0:
                return []  # net vote still favours the position
            out.append(Exit("trend_flip"))
        if want is not Side.FLAT and want is not pos:
            dist = self.feat["stop_dist"][i]
            if math.isfinite(dist) and dist > 0:
                stop = ctx.close[i] - want.value * dist
                out.append(Enter(want, stop=stop, reason=f"votes={v:+.0f}"))
        return out
