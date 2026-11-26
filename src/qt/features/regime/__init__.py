"""Regime models (component doc 03 section 6): quadrant, HMM (filtered only), stress score.

Each is a market feature (a function of :class:`qt.features.market.MarketState` at T) and also a
column of the ``regime`` table (:func:`regime_row`).
"""

from __future__ import annotations

import math

from qt.common.config import MarketCfg, RegimeCfg
from qt.features import registry
from qt.features.market import MarketState
from qt.features.regime import hmm, quadrant, stress

G_QUADRANT = "regime_quadrant"
G_STRESS = "regime_stress"
G_HMM = "regime_hmm"
GROUPS = (G_QUADRANT, G_STRESS, G_HMM)


def register(mcfg: MarketCfg, cfg: RegimeCfg) -> list[registry.FeatureSpec]:
    """Register the regime features (idempotent). Returns the specs."""
    keys: list[tuple[str, int]] = []
    v = cfg.version

    def reg(name, group, formula, sources, fn, window=None):
        registry.feature(name, v, kind="market", group=group, formula=formula, window=window,
                         sources=tuple(sources))(fn)  # fmt: skip
        keys.append((name, v))

    q = cfg.quadrant
    qrule = (f"growth up = Sahm <= mean of its last {q.sahm_avg_months} months + "
             f"{q.sahm_flat_tol:g} and GDPC1 QoQ ann > mean of its last {q.gdp_avg_quarters} "
             f"quarters; inflation up = {q.inflation_series} momentum > 0")  # fmt: skip
    qsrc = ["UNRATE", "GDPC1", q.inflation_series]
    for name in quadrant.QUADRANTS:
        reg(f"quad_{name}", G_QUADRANT, f"1 if quadrant is {name} ({qrule})", qsrc,
            lambda m, name=name: quadrant.one_hot(m, q, name))  # fmt: skip

    s = cfg.stress
    tk = mcfg.tickers
    reg("stress_score", G_STRESS,
        f"mean of z{s.z_window}(VIX), z{s.z_window}(BAA10Y), z{s.z_window}(-(DGS10 - DGS3MO)), "
        "1[VIX / VIX3M > 1]",
        [f"px:{tk['vix']}", f"px:{tk['vix3m']}", "BAA10Y", "DGS10", "DGS3MO"],
        lambda m: stress.stress_score(m, mcfg, s), s.z_window)  # fmt: skip

    h = cfg.hmm
    hrule = (f"{h.states}-state GaussianHMM ({h.covariance}) on weekly SPY {h.ret_window}d log "
             f"return, SPY {h.vol_window}d rvol x sqrt {h.annualization}, BAA10Y "
             f"{h.credit_chg_window}d change, VIX/VIX3M from {h.first_obs}; standardized on the "
             f"training window; seeds 0-{h.n_seeds - 1}, n_iter {h.n_iter}, tol {h.tol:g}, "
             f"min_covar {h.min_covar:g}; monthly expanding refit >= {h.min_weeks} weeks; "
             "states ordered by SPY vol; forward-filtered probability at T")  # fmt: skip
    hsrc = [f"px:{tk['spy']}", f"px:{tk['vix']}", f"px:{tk['vix3m']}", "BAA10Y", "hmm_fits"]
    for i, st in enumerate(hmm.STATES):
        reg(f"hmm_p_{st}", G_HMM, f"P({st}) from {hrule}", hsrc,
            lambda m, i=i: float(hmm.probabilities(m)[i]), "weekly")  # fmt: skip
    return [registry.get(*k) for k in keys]


def regime_row(m: MarketState, mcfg: MarketCfg, cfg: RegimeCfg) -> dict:
    """One ``regime`` table row at T (NaN -> None)."""
    p = hmm.probabilities(m)

    def nn(x: float) -> float | None:
        return float(x) if math.isfinite(x) else None

    return {
        "asof": m.asof,
        "regime_version": cfg.version,
        "quadrant": quadrant.quadrant(m, cfg.quadrant),
        "hmm_calm": nn(p[0]),
        "hmm_normal": nn(p[1]),
        "hmm_stress": nn(p[2]),
        "stress_score": nn(stress.stress_score(m, mcfg, cfg.stress)),
    }
