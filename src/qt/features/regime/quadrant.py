"""Growth/inflation quadrant (component doc 03 section 6.1, D-033 Q6).

Every input is the ALFRED vintage known at T (through :class:`qt.features.market.MarketState`):

- **Growth up** = Sahm not rising (current Sahm <= the mean of its last ``sahm_avg_months``
  monthly values, this one included, + ``sahm_flat_tol``; flat counts as falling) **and** real GDP
  QoQ annualized > the mean of its last ``gdp_avg_quarters`` values (this one included).
- **Inflation up** = core CPI momentum (3-month annualized - YoY, as in ``infl_mom_core_cpi``) > 0.

Goldilocks (growth up, inflation down), Reflation (up, up), Stagflation (down, up), Deflation
(down, down). Unknown (any input missing) -> ``None`` and NaN one-hots.
"""

from __future__ import annotations

import math

import numpy as np

from qt.common.config import QuadrantCfg
from qt.features import market
from qt.features.market import MarketState

GOLDILOCKS, REFLATION = "goldilocks", "reflation"
STAGFLATION, DEFLATION = "stagflation", "deflation"
QUADRANTS = (GOLDILOCKS, REFLATION, STAGFLATION, DEFLATION)


def _mean_last(v: np.ndarray, k: int, need: int) -> float:
    last = v[-k:] if len(v) >= k else np.array([])
    if len(last) < k or not math.isfinite(last[-1]) or np.isfinite(last).sum() < need:
        return math.nan
    return float(np.nanmean(last))


def sahm_not_rising(m: MarketState, cfg: QuadrantCfg) -> bool | None:
    v = market.sahm_values(m).to_numpy(dtype=float)
    # one missing month inside the average is tolerated, like the Sahm averages themselves
    avg = _mean_last(v, cfg.sahm_avg_months, max(cfg.sahm_avg_months - 1, 1))
    if not math.isfinite(avg):
        return None
    return bool(v[-1] <= avg + cfg.sahm_flat_tol)


def gdp_above_avg(m: MarketState, cfg: QuadrantCfg) -> bool | None:
    v = market.gdp_qoq_values(m).to_numpy(dtype=float)
    avg = _mean_last(v, cfg.gdp_avg_quarters, cfg.gdp_avg_quarters)
    if not math.isfinite(avg):
        return None
    return bool(v[-1] > avg)


def growth_up(m: MarketState, cfg: QuadrantCfg) -> bool | None:
    a, b = sahm_not_rising(m, cfg), gdp_above_avg(m, cfg)
    return None if a is None or b is None else (a and b)


def inflation_up(m: MarketState, cfg: QuadrantCfg) -> bool | None:
    x = market.infl_momentum(m, cfg.inflation_series, bridge=True)  # infl_mom_* v2 (D-041)
    return None if not math.isfinite(x) else bool(x > 0)


def quadrant(m: MarketState, cfg: QuadrantCfg) -> str | None:
    key = ("quadrant",)
    if key not in m._cache:
        g, i = growth_up(m, cfg), inflation_up(m, cfg)
        if g is None or i is None:
            q = None
        elif g:
            q = REFLATION if i else GOLDILOCKS
        else:
            q = STAGFLATION if i else DEFLATION
        m._cache[key] = q
    return m._cache[key]


def one_hot(m: MarketState, cfg: QuadrantCfg, name: str) -> float:
    q = quadrant(m, cfg)
    return math.nan if q is None else float(q == name)
