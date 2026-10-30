"""Control modes. A real strategy has to beat these after costs to be worth anything."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trader.modes.base import BarContext, Enter, Exit, Intent, Mode, MoveStop, Provenance, Side
from trader.modes.indicators import atr


class NoTrade(Mode):
    """Flat forever. The floor: zero return, zero risk."""

    name = "no_trade"
    provenance = Provenance.BASELINE
    timeframe = "1D"

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        return []


class AlwaysLong(Mode):
    """Stay long with a trailing ATR stop; re-enter the day after a stop-out. Long-bias control."""

    name = "always_long"
    provenance = Provenance.BASELINE
    timeframe = "1D"
    max_entry_delay = "2h"

    @classmethod
    def default_params(cls) -> dict:
        return {"atr_period": 14, "stop_atr": 2.0}

    def prepare(self, bars: pd.DataFrame) -> None:
        self.feat["atr"] = atr(bars, self.params["atr_period"])

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        a = self.feat["atr"][ctx.i]
        if not math.isfinite(a):
            return []
        stop = ctx.close[ctx.i] - self.params["stop_atr"] * a
        if ctx.position.side is Side.FLAT:
            return [Enter(Side.LONG, stop=stop, reason="always_long")]
        return [MoveStop(stop, "trail")] if stop > ctx.position.stop else []


class RandomEntry(Mode):
    """Coin-flip entries with ATR stop/target and a time exit. Tests whether a mode beats luck.

    Draws are seeded per bar timestamp, so results do not depend on where the data window starts.
    """

    name = "random"
    provenance = Provenance.BASELINE
    timeframe = "1h"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "seed": 7, "entry_prob": 0.05, "atr_period": 14, "stop_atr": 1.5,
            "target_atr": 2.0, "max_bars": 8, "allow_short": True,
        }  # fmt: skip

    def prepare(self, bars: pd.DataFrame) -> None:
        self.feat["atr"] = atr(bars, self.params["atr_period"])

    def _rng(self, ts: pd.Timestamp) -> np.random.Generator:
        return np.random.default_rng([self.params["seed"], int(ts.value // 60_000_000_000)])

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        p = self.params
        if ctx.position.side is not Side.FLAT:
            return [Exit("time")] if ctx.position.bars_held >= p["max_bars"] else []
        a = self.feat["atr"][ctx.i]
        if not math.isfinite(a) or a <= 0:
            return []
        rng = self._rng(ctx.index[ctx.i])
        if rng.random() >= p["entry_prob"]:
            return []
        side = Side.SHORT if p["allow_short"] and rng.random() < 0.5 else Side.LONG
        price = ctx.close[ctx.i]
        s = side.value
        return [
            Enter(
                side,
                stop=price - s * p["stop_atr"] * a,
                target=price + s * p["target_atr"] * a,
                reason="coin_flip",
            )
        ]
