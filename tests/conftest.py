from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from trader.instruments import MES
from trader.modes.base import BarContext, Intent, Mode, Provenance


def make_bars(
    rows: list[tuple[float, float, float, float]],
    start: str = "2024-01-02 14:30",
    freq: str = "1min",
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a 1-minute bar frame from (open, high, low, close) tuples."""
    idx = (
        index
        if index is not None
        else pd.date_range(start, periods=len(rows), freq=freq, tz="UTC", name="ts")
    )
    arr = np.array(rows, dtype=float)
    return pd.DataFrame(
        {"open": arr[:, 0], "high": arr[:, 1], "low": arr[:, 2], "close": arr[:, 3], "volume": 1.0},
        index=idx,
    )


def random_walk(
    n: int, seed: int = 0, start: str = "2024-01-02 00:00", price: float = 4000.0, step: float = 1.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = price + np.cumsum(rng.normal(0, step, n))
    open_ = np.r_[price, close[:-1]]
    wiggle = np.abs(rng.normal(0, step / 2, n))
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    return make_bars(list(zip(open_, high, low, close, strict=True)), start=start)


class Scripted(Mode):
    """Emits pre-scripted intents at given 1-minute bar indices (decision at bar close)."""

    name = "scripted"
    provenance = Provenance.BASELINE
    timeframe = "1min"

    def __init__(self, script: dict[int, list[Intent]]) -> None:
        super().__init__()
        self.script = script
        self.seen: list[BarContext] = []

    def on_bar(self, ctx: BarContext) -> list[Intent]:
        self.seen.append(ctx)
        return self.script.get(ctx.i, [])


@pytest.fixture
def free_mes():
    """MES specs with every cost set to zero, for exact arithmetic."""
    return replace(MES, half_spread=0.0, slippage=0.0, commission_per_unit_side=0.0)
