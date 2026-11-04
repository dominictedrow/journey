"""IMOM01: market intraday momentum (Baltussen, Da, Lammers, Martens 2021, JFE 142:377-403).

Rules and page references: research/sources/PAPER_19_IMOM_METHODS_REVIEW.md.

Source rule (Eq. 12, p. 385): long the last 30 minutes of the cash session if the return from the
previous 16:00 ET close to 15:30 ET (r_ROD) is positive, short otherwise, flat at the close.

Our ADAPTATIONS, each a deviation from the paper:
- Execution: the paper earns r_LH from the same 15:30 price that forms the signal. We fill at the
  open of the first bar after the decision, plus optional ``entry_lag_min`` minutes, net of costs.
- Exit at 16:00 ET (not the 16:15 futures close; Table 12) via a market order at the 16:00 open.
- Mandatory protective stop (D-015), which the paper does not have: ``stop_sigma`` x the expected
  30-minute move, from the trailing ``vol_window`` 1-minute returns.
- Instrument: S&P 500 only. The paper finds no effect in Euro FX (Table B4).
- Early-close days are skipped, as in the paper (p. 381).
- Optional ``require_onfh_agree``: trade only when r_ONFH (prev close -> 10:00 ET) has the same
  sign as r_ROD (Table 6 variant).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trader.data.resample import ET
from trader.modes.base import BarContext, Enter, Exit, Intent, Mode, Provenance, Side
from trader.sessions import nyse_sessions

MIN = pd.Timedelta("1min")


class IntradayMomentum(Mode):
    name = "imom01"
    version = "0.1.0"
    provenance = Provenance.ADAPTATION
    instruments = ("USA500IDXUSD",)
    timeframe = "1min"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "entry_lag_min": 0,
            "stop_sigma": 2.0,
            "vol_window": 60,
            "min_abs_signal": 0.0,
            "require_onfh_agree": False,
            "allow_short": True,
        }

    def prepare(self, bars: pd.DataFrame) -> None:
        p = self.params
        n = len(bars)
        idx = bars.index
        close = bars["close"].to_numpy()
        self.feat = {
            "signal": np.full(n, np.nan),
            "onfh": np.full(n, np.nan),
            "stop_dist": np.full(n, np.nan),
            "exit": np.zeros(n, dtype=bool),
        }
        if n == 0:
            return

        logret = np.log(bars["close"]).diff()
        # Expected 30-minute move in price units, from trailing 1-minute returns only.
        sigma30 = (logret.rolling(p["vol_window"]).std() * math.sqrt(30)).to_numpy() * close

        sessions = nyse_sessions(idx[0], idx[-1])
        close_et = sessions["close"].dt.tz_convert(ET)
        full_day = (close_et.dt.hour == 16) & (close_et.dt.minute == 0)

        def exact(ts: pd.Series) -> np.ndarray:
            """Position of the bar starting exactly at ``ts``, or -1 if missing."""
            return idx.get_indexer(pd.DatetimeIndex(ts))

        # Bar starting 15:59 closes at 16:00: its close is P(close). Bar starting 15:29 -> P(15:30).
        close_bar = exact(sessions["close"] - MIN)
        signal_bar = exact(sessions["close"] - 31 * MIN)
        onfh_bar = exact(sessions["open"] + 29 * MIN)
        # Exit decision: first bar at or after the 15:59 bar, so a data hole still flattens at
        # the earliest bar available rather than carrying the position on.
        exit_bar = np.searchsorted(idx.asi8, pd.DatetimeIndex(sessions["close"] - MIN).asi8)
        lag = int(p["entry_lag_min"])

        for s in range(len(sessions)):
            if exit_bar[s] < n:
                self.feat["exit"][exit_bar[s]] = True
            if s == 0 or not full_day.iloc[s]:
                continue
            ref, sig = close_bar[s - 1], signal_bar[s]
            if ref < 0 or sig < 0:
                continue  # missing reference or signal bar: no trade
            entry = exact(pd.Series([idx[sig] + lag * MIN]))[0] if lag else sig
            if entry < 0 or entry >= exit_bar[s]:
                continue
            self.feat["signal"][entry] = close[sig] / close[ref] - 1
            self.feat["stop_dist"][entry] = sigma30[entry]
            if onfh_bar[s] >= 0:
                self.feat["onfh"][entry] = close[onfh_bar[s]] / close[ref] - 1

    def decision_mask(self) -> np.ndarray:
        return ~np.isnan(self.feat["signal"]) | self.feat["exit"]

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        p = self.params
        i = ctx.i
        if ctx.position.side is not Side.FLAT:
            return [Exit("close_1600")] if self.feat["exit"][i] else []
        sig = self.feat["signal"][i]
        dist = self.feat["stop_dist"][i]
        if not math.isfinite(sig) or not math.isfinite(dist) or abs(sig) <= p["min_abs_signal"]:
            return []
        if p["require_onfh_agree"] and np.sign(self.feat["onfh"][i]) != np.sign(sig):
            return []
        side = Side.LONG if sig > 0 else Side.SHORT
        if side is Side.SHORT and not p["allow_short"]:
            return []
        price = ctx.close[i]
        stop = price - side.value * p["stop_sigma"] * dist
        return [Enter(side, stop=stop, reason=f"r_rod={sig:+.5f}")]
