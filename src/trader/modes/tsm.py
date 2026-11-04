"""TSM01: monthly time-series momentum (research/experiments/TSM01_DESIGN.md,
pre-registered in 3b3bf52).

ADAPTATION of Moskowitz, Ooi & Pedersen (2012) to one instrument at a time:
- Daily bars (17:00 ET roll). Decide on the last session of each calendar month.
- Signal: sign of close[i] / close[i - 21*L] - 1 (L months, 21 sessions a month).
- Vol: EWMA std of daily log returns, centre of mass 60; monthly sigma = daily * sqrt(21).
- Signal flips -> exit and reverse. Unchanged -> hold and move the stop tighter-only to
  close -/+ stop_sigma * sigma_m * close. Flat (e.g. after a stop-out) -> enter on the signal.
- Protective stop (D-015): entry -/+ stop_sigma * sigma_m * price.

Month-end detection is calendar-only (no look-ahead): a session is a decision session if the
next weekday falls in a new month. If that session is missing (holiday), the first session of
the new month decides instead.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trader.data.resample import ET
from trader.modes.base import BarContext, Enter, Exit, Intent, Mode, MoveStop, Provenance, Side


def session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Session date of each daily bar (bar start 17:00 ET belongs to the next day)."""
    wall = index.tz_convert(ET).tz_localize(None)
    return (wall + pd.Timedelta(hours=7)).normalize()


def month_end_mask(index: pd.DatetimeIndex) -> np.ndarray:
    d = session_dates(index)
    nxt = d + pd.offsets.BDay(1)
    last_weekday = np.asarray(nxt.month != d.month)
    months = np.asarray(d.year * 12 + d.month)
    out = last_weekday.copy()
    for i in range(1, len(d)):
        # Previous month ended without a decision (its last weekday had no bar): decide now.
        if months[i] != months[i - 1] and not out[i - 1]:
            out[i] = True
    return out


class TimeSeriesMomentum(Mode):
    name = "tsm01"
    version = "0.1.0"
    provenance = Provenance.ADAPTATION
    instruments = ("USA500IDXUSD", "EURUSD")
    timeframe = "1D"
    max_entry_delay = "2h"  # spans the 17:00-18:00 ET futures halt

    @classmethod
    def default_params(cls) -> dict:
        return {"lookback_months": 12, "vol_com": 60, "stop_sigma": 3.0, "allow_short": True}

    def prepare(self, bars: pd.DataFrame) -> None:
        p = self.params
        close = bars["close"]
        n = 21 * p["lookback_months"]
        self.feat["ret"] = (close / close.shift(n) - 1).to_numpy()
        logret = np.log(close).diff()
        sd = logret.ewm(com=p["vol_com"], min_periods=p["vol_com"]).std()
        self.feat["sigma_m"] = (sd * math.sqrt(21)).to_numpy()
        self.feat["decide"] = month_end_mask(bars.index)

    def decision_mask(self) -> np.ndarray:
        # Calendar-only: decisions happen at month ends by design. Stops are checked by the
        # engine on 1-minute bars in between.
        return self.feat["decide"]

    def signal(self, i: int) -> Side:
        r = self.feat["ret"][i]
        if not math.isfinite(r) or r == 0:
            return Side.FLAT
        if r < 0 and not self.params["allow_short"]:
            return Side.FLAT
        return Side.LONG if r > 0 else Side.SHORT

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        i = ctx.i
        sm = self.feat["sigma_m"][i]
        want = self.signal(i)
        pos = ctx.position.side
        price = ctx.close[i]
        if not math.isfinite(sm) or sm <= 0:
            return [Exit("no_vol")] if pos is not Side.FLAT else []
        dist = self.params["stop_sigma"] * sm * price
        if want is Side.FLAT:
            return [Exit("no_signal")] if pos is not Side.FLAT else []
        stop = price - want.value * dist
        if pos is want:
            tighter = (stop - ctx.position.stop) * want.value > 0
            return [MoveStop(stop, "monthly")] if tighter else []
        intents: list[Intent] = [Exit("flip")] if pos is not Side.FLAT else []
        reason = f"ret{self.params['lookback_months']}m={self.feat['ret'][i]:+.3f}"
        return [*intents, Enter(want, stop=stop, reason=reason)]
