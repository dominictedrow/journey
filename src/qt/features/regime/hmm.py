"""3-state Gaussian HMM on weekly market observations (component doc 03 section 6.2, D-033 Q5).

Observations (one row per week, sampled at the week's last XNYS session):
SPY 21-session log return, SPY 21-session realized vol (annualized), BAA10Y 21-session change and
VIX / VIX3M. Each series is sampled at sessions as its newest observation on or before the session
(as ``MarketState.sampled`` does), from the data known at the time the history is read.

Point in time:
- **Fits.** A fit dated F (first Sunday of a month, usable from ``refit_et`` ET) is trained on the
  weekly history *as known at F* (the ALFRED vintage at F carries the whole revised history;
  BAA10Y's first vintage is 2014-01-27, so no fit exists before February 2014). Inputs are
  standardized with the training window's mean and std. 20 restarts (seeds 0..19), the best
  log-likelihood wins, and states are then sorted by mean SPY vol into calm < normal < stress
  (locked rule).
- **Probabilities at T.** The newest fit usable at T runs the *forward* filter over the weekly
  history as known at T, up to the newest weekly sample whose decision time is <= T. Only the
  forward pass is used (never the smoothed ``predict_proba``), so appending later weeks cannot
  change an earlier probability. Between weekly samples the latest weekly step carries forward.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from qt.common.config import HmmCfg, MarketCfg
from qt.features.inputs import ET
from qt.features.market import MarketState

log = logging.getLogger("qt.features.hmm")

OBS_COLS = ("spy_ret", "spy_rvol", "baa_chg", "vix_ratio")
VOL_COL = 1  # states are ordered by the mean of this column
STATES = ("calm", "normal", "stress")


# ---------------------------------------------------------------- observations
def _sample(s: pd.Series, days: np.ndarray) -> np.ndarray:
    """Newest observation dated on or before each day (NaN before the first)."""
    if not len(s):
        return np.full(len(days), np.nan)
    idx = s.index.to_numpy()
    pos = np.searchsorted(idx, days, side="right") - 1
    v = s.to_numpy(dtype=float)
    return np.where(pos >= 0, v[np.maximum(pos, 0)], np.nan)


def _at_et(d: date, hhmm: str) -> pd.Timestamp:
    h, mi = (int(x) for x in hhmm.split(":")[:2])
    return pd.Timestamp(datetime(d.year, d.month, d.day, h, mi, tzinfo=ET))


def week_end_sessions(sessions: np.ndarray) -> np.ndarray:
    """The last session of each Monday-Sunday week, from the full calendar (datetime64[D])."""
    s = np.asarray(sessions, dtype="datetime64[D]")
    # numpy weeks start on Thursday (epoch); shift so a week is Monday..Sunday
    wk = (s - np.datetime64("1970-01-05", "D")).astype(int) // 7
    last = np.r_[wk[1:] != wk[:-1], True]
    return s[last]


def session_obs(m: MarketState, mcfg: MarketCfg, cfg: HmmCfg) -> pd.DataFrame:
    """The four inputs at every session from ``first_obs`` - a buffer to T, as known at T."""
    key = ("hmm_session_obs",)
    if key in m._cache:
        return m._cache[key]
    w = max(cfg.ret_window, cfg.vol_window, cfg.credit_chg_window)
    start = np.datetime64(cfg.first_obs, "D")
    sess = m.sess
    i0 = max(int(np.searchsorted(sess, start)) - w, 0)
    days = sess[i0:]
    tk = mcfg.tickers
    with np.errstate(all="ignore"):
        lp = np.log(_sample(m.px(tk["spy"]), days))
        baa = _sample(m.mac("BAA10Y"), days)
        vix = _sample(m.px(tk["vix"]), days)
        vix3m = _sample(m.px(tk["vix3m"]), days)
        r = pd.Series(np.r_[np.nan, np.diff(lp)])
        rvol = r.rolling(cfg.vol_window, min_periods=cfg.vol_window).std(ddof=1).to_numpy()
        out = pd.DataFrame(
            {
                "spy_ret": lp - pd.Series(lp).shift(cfg.ret_window).to_numpy(),
                "spy_rvol": rvol * math.sqrt(cfg.annualization),
                "baa_chg": baa - pd.Series(baa).shift(cfg.credit_chg_window).to_numpy(),
                "vix_ratio": vix / vix3m,
            },
            index=pd.Index(days, name="date"),
        )
    out = out[out.index >= start]
    m._cache[key] = out
    return out


def weekly_obs(m: MarketState, mcfg: MarketCfg, cfg: HmmCfg) -> pd.DataFrame:
    """One row per week whose last session's decision time (``sample_et``) is <= T."""
    key = ("hmm_weekly_obs",)
    if key in m._cache:
        return m._cache[key]
    obs = session_obs(m, mcfg, cfg)
    ends = week_end_sessions(m.sessions)
    ends = ends[(ends >= np.datetime64(cfg.first_obs, "D")) & (ends <= m.today)]
    if len(ends) and ends[-1] == m.today:
        if m.asof < _at_et(pd.Timestamp(ends[-1]).date(), cfg.sample_et):
            ends = ends[:-1]
    out = obs.reindex(pd.Index(ends, name="date"))
    m._cache[key] = out
    return out


# ---------------------------------------------------------------- fit
@dataclass(frozen=True)
class HmmFit:
    fit_date: date
    available_ns: int  # UTC epoch ns from which the fit may be used
    train_start: date
    train_end: date
    n_weeks: int
    seed: int
    log_likelihood: float
    mu: np.ndarray
    sd: np.ndarray
    startprob: np.ndarray  # all state arrays already ordered calm, normal, stress
    transmat: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    state_order: tuple[int, ...]  # fitted index of calm, normal, stress
    regime_version: int = 1
    _cache: dict = field(default_factory=dict, compare=False, repr=False)


def fit_dates(start: date, end: date, weekday: int) -> list[date]:
    """The first ``weekday`` (0 = Monday) of every month with ``start <= d <= end``."""
    out = []
    y, mo = start.year, start.month
    while date(y, mo, 1) <= end:
        d = date(y, mo, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        if start <= d <= end:
            out.append(d)
        y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
    return out


def available_at(fit_date: date, cfg: HmmCfg) -> pd.Timestamp:
    return _at_et(fit_date, cfg.refit_et).tz_convert("UTC")


def training_set(weekly: pd.DataFrame) -> pd.DataFrame:
    """Rows with all four inputs present (leading weeks before an input exists drop out)."""
    return weekly.dropna(how="any")


def fit(x: pd.DataFrame, cfg: HmmCfg, fit_date: date, *, regime_version: int = 1) -> HmmFit:
    """Fit on the complete weekly rows ``x``; raises ValueError if there are too few."""
    from hmmlearn.hmm import GaussianHMM

    if len(x) < cfg.min_weeks:
        raise ValueError(f"{len(x)} weekly samples < min_weeks {cfg.min_weeks}")
    xa = x[list(OBS_COLS)].to_numpy(dtype=float)
    mu, sd = xa.mean(axis=0), xa.std(axis=0, ddof=1)
    if not (sd > 0).all():
        raise ValueError("an HMM input has zero variance in the training window")
    xs = (xa - mu) / sd
    best: tuple[float, int, object] | None = None
    hmm_log = logging.getLogger("hmmlearn")
    level = hmm_log.level
    hmm_log.setLevel(logging.ERROR)  # non-convergence warnings for single restarts are expected
    try:
        for seed in range(cfg.n_seeds):
            model = GaussianHMM(n_components=cfg.states, covariance_type=cfg.covariance,
                                n_iter=cfg.n_iter, tol=cfg.tol, min_covar=cfg.min_covar,
                                random_state=seed)  # fmt: skip
            try:
                model.fit(xs)
                ll = float(model.score(xs))
            except (ValueError, np.linalg.LinAlgError) as e:
                log.debug("hmm seed %d failed: %s", seed, e)
                continue
            if math.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, seed, model)
    finally:
        hmm_log.setLevel(level)
    if best is None:
        raise ValueError("every HMM restart failed")
    ll, seed, model = best
    order = np.argsort(model.means_[:, VOL_COL], kind="stable")
    return HmmFit(
        fit_date=fit_date,
        available_ns=int(available_at(fit_date, cfg).as_unit("ns").value),
        train_start=pd.Timestamp(x.index[0]).date(),
        train_end=pd.Timestamp(x.index[-1]).date(),
        n_weeks=len(x),
        seed=seed,
        log_likelihood=ll,
        mu=mu,
        sd=sd,
        startprob=model.startprob_[order].copy(),
        transmat=model.transmat_[np.ix_(order, order)].copy(),
        means=model.means_[order].copy(),
        covars=np.asarray(model.covars_)[order].copy(),
        state_order=tuple(int(i) for i in order),
        regime_version=regime_version,
    )


# ---------------------------------------------------------------- filter
def filtered(f: HmmFit, x: pd.DataFrame) -> np.ndarray:
    """Forward-filtered state probabilities p(state_t | obs_1..t), one row per row of ``x``.

    A row with a missing input is a missing observation: only the transition step applies.
    """
    xa = (x[list(OBS_COLS)].to_numpy(dtype=float) - f.mu) / f.sd
    n, k = len(xa), len(f.startprob)
    logb = np.zeros((n, k))
    ok = np.isfinite(xa).all(axis=1)
    if ok.any():
        for j in range(k):
            logb[ok, j] = multivariate_normal.logpdf(xa[ok], mean=f.means[j], cov=f.covars[j],
                                                     allow_singular=True)  # fmt: skip
    with np.errstate(divide="ignore"):
        loga = np.log(f.transmat)
        alpha = np.log(f.startprob)
    out = np.empty((n, k))
    for t in range(n):
        if t:
            alpha = logsumexp(alpha[:, None] + loga, axis=0)
        alpha = alpha + logb[t]
        alpha = alpha - logsumexp(alpha)
        out[t] = np.exp(alpha)
    return out


def relabel_change(prev: HmmFit, new: HmmFit, x: pd.DataFrame, weeks: int) -> float:
    """Share of the last ``weeks`` complete rows whose argmax state differs between two fits.

    Both fits filter the same history (from its first complete row, as :func:`probabilities`).
    """
    x = from_first_complete(x)
    if x.empty:
        return math.nan
    ok = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    a = filtered(prev, x)[ok][-weeks:].argmax(axis=1)
    b = filtered(new, x)[ok][-weeks:].argmax(axis=1)
    return float((a != b).mean())


def from_first_complete(x: pd.DataFrame) -> pd.DataFrame:
    """``x`` from its first row with all inputs present (earlier rows carry no information)."""
    ok = np.flatnonzero(np.isfinite(x[list(OBS_COLS)].to_numpy(dtype=float)).all(axis=1))
    return x.iloc[ok[0] :] if len(ok) else x.iloc[:0]


# ---------------------------------------------------------------- inputs for features
@dataclass
class HmmInputs:
    fits: list[HmmFit]  # any order
    mcfg: MarketCfg
    cfg: HmmCfg

    def __post_init__(self) -> None:
        self.fits = sorted(self.fits, key=lambda f: f.available_ns)
        self._avail = np.array([f.available_ns for f in self.fits], dtype="int64")

    def fit_at(self, asof_ns: int) -> HmmFit | None:
        i = int(np.searchsorted(self._avail, asof_ns, side="right")) - 1
        return self.fits[i] if i >= 0 else None


def probabilities(m: MarketState) -> np.ndarray:
    """calm / normal / stress at T (NaN without a fit, or if the newest weekly row is partial)."""
    key = ("hmm_probs",)
    if key in m._cache:
        return m._cache[key]
    nan = np.full(3, np.nan)
    h: HmmInputs | None = m.regime
    f = h.fit_at(m.asof_ns) if h is not None else None
    if f is None:
        m._cache[key] = nan
        return nan
    x = weekly_obs(m, h.mcfg, h.cfg)
    if x.empty or not np.isfinite(x.iloc[-1].to_numpy(dtype=float)).all():
        m._cache[key] = nan
        return nan
    # Start the filter at the first complete row, so the result doesn't depend on how far back
    # first_obs reaches.
    p = filtered(f, from_first_complete(x))[-1]
    m._cache[key] = p
    return p


# ---------------------------------------------------------------- persistence
def model_path(model_dir: Path, f: HmmFit) -> Path:
    return model_dir / f"v{f.regime_version}" / f"{f.fit_date.isoformat()}.joblib"


_ARRAYS = ("mu", "sd", "startprob", "transmat", "means", "covars")


def save(f: HmmFit, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {k: getattr(f, k) for k in HmmFit.__dataclass_fields__ if not k.startswith("_")}
    joblib.dump(d, path)


def load(path: Path) -> HmmFit:
    d = joblib.load(path)
    for k in _ARRAYS:
        d[k] = np.asarray(d[k], dtype=float)
    d["state_order"] = tuple(d["state_order"])
    return HmmFit(**d)
