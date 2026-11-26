"""Stress score (component doc 03 section 6.3).

Mean of z(VIX), z(BAA10Y), z(-(DGS10 - DGS3MO)) and the VIX/VIX3M > 1 flag (1 or 0). Each z is the
latest value against the last ``z_window`` session samples known at T, this one included (the same
past-only z as the market features' ``*_z252``). Any missing component -> NaN.
"""

from __future__ import annotations

import math

import numpy as np

from qt.common.config import MarketCfg, StressCfg
from qt.features.market import MarketState, zlast


def components(m: MarketState, mcfg: MarketCfg, cfg: StressCfg) -> dict[str, float]:
    n = cfg.z_window
    vix, vix3m = f"px:{mcfg.tickers['vix']}", f"px:{mcfg.tickers['vix3m']}"
    with np.errstate(all="ignore"):
        slope = m.sampled("DGS10", n) - m.sampled("DGS3MO", n)
        ratio = m.latest(vix) / m.latest(vix3m)
    return {
        "z_vix": zlast(m.sampled(vix, n)),
        "z_baa10y": zlast(m.sampled("BAA10Y", n)),
        "z_neg_slope": zlast(-slope),
        "vix_inverted": float(ratio > 1) if math.isfinite(ratio) else math.nan,
    }


def stress_score(m: MarketState, mcfg: MarketCfg, cfg: StressCfg) -> float:
    key = ("stress_score",)
    if key not in m._cache:
        c = list(components(m, mcfg, cfg).values())
        m._cache[key] = float(np.mean(c)) if all(math.isfinite(x) for x in c) else math.nan
    return m._cache[key]
