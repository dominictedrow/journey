"""Per-asset price features (component doc 03 section 5.1).

Every feature is a pure function of one :class:`PriceFrame` (one asset's bars as known at a
decision time, adjusted in memory) and returns one value per bar date. The value at date d uses
only bars and context rows dated on or before d, so a whole history is computed in one pass
(backfill) and the last row is the daily value. :func:`prepare` does the point-in-time cut: it
keeps only rows with ``released_at <= asof`` (bars, corporate actions, context series, earnings).

Scale: models never see price levels. Every feature here is a return, a ratio or a statistic of
returns, so a later corporate action (which rescales all earlier bars by one constant) leaves the
values unchanged; that is what lets one backfill pass stand in for each decision time.

Windows come from ``features.price`` and appear in the feature names (``ret_21d``). The formula
text of each feature includes its numbers, so a config change of an existing name fails the
registry sync until that feature's version is bumped.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from qt.common.config import AdjustCfg, PriceCfg
from qt.features import registry
from qt.features.adjust import adjust

GROUP_RETURNS = "price_returns"
GROUP_VOL = "price_vol"
GROUP_TREND = "price_trend"
GROUP_LIQ = "price_liquidity"
GROUP_LINK = "price_market_link"
GROUP_MR = "price_mean_reversion"
GROUP_EARN = "earnings"

BARS = "bars"  # registry source labels (not table names: inputs.py owns those)
EARNINGS = "earnings_calendar"
NS_INF = np.iinfo(np.int64).max


@dataclass(frozen=True)
class CtxSeries:
    """A context series (benchmark, DXY, 10y yield): ``date, value, released_at`` rows.

    ``kind`` says how a daily change is taken: ``log`` (prices) or ``diff`` (yields).
    """

    frame: pd.DataFrame
    kind: Literal["log", "diff"] = "log"


@dataclass
class PriceFrame:
    """One asset's adjusted bars at a decision time, with aligned context changes."""

    asset_class: str
    dates: np.ndarray  # datetime64[D], ascending
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray  # adjusted
    volume: np.ndarray  # split-adjusted
    close_raw: np.ndarray
    volume_raw: np.ndarray
    ready_ns: np.ndarray  # UTC ns: when this date's bar was released
    ctx: dict[str, np.ndarray]  # daily change per context key, NaN where the date is missing
    # UTC ns per date and context key: when that series had caught up to the date (its first row
    # dated on/after it was released). Until then a row of a feature using the key is not ready:
    # "no row yet" and "no row ever" (a holiday, a weekend for SPY) can't be told apart earlier.
    ctx_ready_ns: dict[str, np.ndarray]
    asof_ns: int
    earnings: pd.DataFrame
    sessions: np.ndarray  # XNYS sessions, datetime64[D]
    cfg: PriceCfg
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.dates)

    @property
    def ann(self) -> float:
        return float(self.cfg.ann(self.asset_class))

    @property
    def logp(self) -> np.ndarray:
        if "logp" not in self._cache:
            with np.errstate(divide="ignore", invalid="ignore"):
                self._cache["logp"] = np.where(self.close > 0, np.log(self.close), np.nan)
        return self._cache["logp"]

    @property
    def r(self) -> np.ndarray:
        """Daily log return (first bar NaN)."""
        if "r" not in self._cache:
            self._cache["r"] = _lag_diff(self.logp, 1)
        return self._cache["r"]


# ---------------------------------------------------------------- preparation (the PIT cut)
def _known(df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    return df[df["released_at"] <= asof] if not df.empty else df


def _ctx_changes(
    series: CtxSeries, asof: pd.Timestamp, dates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Daily change aligned to ``dates`` and the caught-up time per date (rows known at asof)."""
    f = _known(series.frame, asof)
    if f.empty:
        return np.full(len(dates), np.nan), np.full(len(dates), NS_INF, dtype="int64")
    f = f.sort_values(["date", "released_at"]).drop_duplicates("date", keep="last")
    v = f["value"].astype(float).to_numpy()
    if series.kind == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            lv = np.where(v > 0, np.log(v), np.nan)
        chg = _lag_diff(lv, 1)
    else:
        chg = _lag_diff(v, 1)
    rel = pd.to_datetime(f["released_at"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    # a change needs the previous row too: it is ready when the later of the two is released
    rel = np.maximum(rel, np.concatenate([[rel[0]], rel[:-1]]))
    cd = np.array(f["date"].tolist(), dtype="datetime64[D]")
    aligned = pd.Series(chg, index=pd.Index(cd)).reindex(pd.Index(dates)).to_numpy(dtype=float)
    first_after = np.minimum.accumulate(rel[::-1])[::-1]  # earliest release among rows >= date
    j = np.searchsorted(cd, dates, side="left")
    caught = np.where(j < len(cd), first_after[np.minimum(j, len(cd) - 1)], NS_INF)
    return aligned, caught.astype("int64")


def prepare(
    bars: pd.DataFrame,
    asof: datetime,
    cfg: PriceCfg,
    adjust_cfg: AdjustCfg,
    *,
    asset_class: str,
    actions: pd.DataFrame | None = None,
    ctx: Mapping[str, CtxSeries] | None = None,
    earnings: pd.DataFrame | None = None,
    sessions: np.ndarray | None = None,
) -> PriceFrame | None:
    """Cut one asset's inputs at ``asof`` (``released_at <= asof``) and build its frame.

    ``bars`` are that asset's rows from :func:`qt.features.inputs.load_bars`; ``actions`` its
    mapped corporate actions; ``ctx`` the context series by key (``bench`` plus the keys of
    ``cfg.corr.series``); ``earnings`` its rows from :func:`qt.features.inputs.load_earnings`.
    """
    t = pd.Timestamp(asof)
    if t.tzinfo is None:
        raise ValueError("asof must be timezone-aware")
    b = _known(bars, t)
    if b.empty:
        return None
    acts = actions if actions is not None else pd.DataFrame()
    acts = _known(acts, t) if not acts.empty else acts
    a = adjust(b, acts, t.to_pydatetime(), adjust_cfg) if not acts.empty else _unadjusted(b)
    dates = np.array(a["date"].tolist(), dtype="datetime64[D]")
    ready = pd.to_datetime(a["released_at"], utc=True).dt.as_unit("ns").astype("int64").to_numpy()
    aligned: dict[str, np.ndarray] = {}
    ctx_ready: dict[str, np.ndarray] = {}
    for key, series in (ctx or {}).items():
        aligned[key], ctx_ready[key] = _ctx_changes(series, t, dates)
    e = earnings if earnings is not None else pd.DataFrame()
    return PriceFrame(
        asset_class=asset_class,
        dates=dates,
        open=a["adj_open"].to_numpy(dtype=float),
        high=a["adj_high"].to_numpy(dtype=float),
        low=a["adj_low"].to_numpy(dtype=float),
        close=a["adj_close"].to_numpy(dtype=float),
        volume=a["adj_volume"].to_numpy(dtype=float),
        close_raw=a["close"].to_numpy(dtype=float),
        volume_raw=a["volume"].to_numpy(dtype=float),
        ready_ns=ready,
        ctx=aligned,
        ctx_ready_ns=ctx_ready,
        asof_ns=int(t.tz_convert("UTC").as_unit("ns").value),
        earnings=_known(e, t) if not e.empty else e,
        sessions=sessions if sessions is not None else _xnys_sessions(),
        cfg=cfg,
    )


def _unadjusted(b: pd.DataFrame) -> pd.DataFrame:
    out = b.sort_values("date").reset_index(drop=True).copy()
    for c in ("open", "high", "low", "close"):
        out[f"adj_{c}"] = out[c]
    out["adj_volume"] = out["volume"]
    return out


_SESSIONS: np.ndarray | None = None


def _xnys_sessions() -> np.ndarray:
    global _SESSIONS
    if _SESSIONS is None:
        from qt.common.calendar import TradingCalendar

        cal = TradingCalendar().cal
        _SESSIONS = np.array(cal.sessions.date, dtype="datetime64[D]")
    return _SESSIONS


# ---------------------------------------------------------------- evaluation
def ctx_keys(spec: registry.FeatureSpec) -> list[str]:
    return [x[4:] for x in spec.sources if x.startswith("ctx:")]


def ready_ns(frame: PriceFrame, spec: registry.FeatureSpec) -> np.ndarray:
    """Per bar date, from when (UTC ns) the spec's value on that row may be used."""
    r = frame.ready_ns
    for key in ctx_keys(spec):
        r = np.maximum(r, frame.ctx_ready_ns.get(key, np.full(frame.n, NS_INF, dtype="int64")))
    return r


def compute_series(
    frame: PriceFrame, specs: list[registry.FeatureSpec]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every spec over the frame's whole history: ``(values, ready_at)``.

    Both are indexed by bar date with one column per spec key. ``ready_at`` (UTC epoch ns) says from which
    decision time each value may be used; at decision time T a model reads, per feature, the
    newest row whose ``ready_at <= T``.
    """
    idx = pd.Index(frame.dates, name="date")
    cols = pd.Index([s.key for s in specs], tupleize_cols=False)  # (name, version), no MultiIndex
    vals = np.column_stack([np.asarray(s.fn(frame), dtype=float) for s in specs])
    ready = np.column_stack([ready_ns(frame, s) for s in specs])
    values = pd.DataFrame(vals, index=idx, columns=cols)
    ready_at = pd.DataFrame(ready, index=idx, columns=cols)
    return values, ready_at


def compute_asof(frame: PriceFrame, specs: list[registry.FeatureSpec]) -> dict:
    """Each spec's value at the frame's decision time: its newest row ready by then."""
    values, ready = compute_series(frame, specs)
    out = {}
    for k in values.columns:
        ok = np.flatnonzero(ready[k].to_numpy() <= frame.asof_ns)
        out[k] = float(values[k].iloc[ok[-1]]) if len(ok) else math.nan
    return out


def price_specs() -> list[registry.FeatureSpec]:
    groups = (GROUP_RETURNS, GROUP_VOL, GROUP_TREND, GROUP_LIQ, GROUP_LINK, GROUP_MR, GROUP_EARN)
    return [s for s in registry.all_specs() if s.group in groups]


# ---------------------------------------------------------------- numeric helpers
def _lag_diff(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) > n:
        out[n:] = x[n:] - x[:-n]
    return out


def _lag(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if n == 0:
        return x.astype(float).copy()
    if len(x) > n:
        out[n:] = x[:-n]
    return out


def _roll(x: np.ndarray, n: int, how: str, ddof: int = 1) -> np.ndarray:
    """Rolling stat over n bars, NaN unless all n are finite."""
    s = pd.Series(x, dtype=float).rolling(n, min_periods=n)
    if how == "mean":
        return s.mean().to_numpy()
    if how == "std":
        return s.std(ddof=ddof).to_numpy()
    if how == "sum":
        return s.sum().to_numpy()
    raise ValueError(how)


def _pair_sums(y: np.ndarray, x: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """Rolling sums over the last n bars of the dates where both x and y are finite."""
    v = np.isfinite(x) & np.isfinite(y)
    xs = np.where(v, x, 0.0)
    ys = np.where(v, y, 0.0)

    def rs(a: np.ndarray) -> np.ndarray:
        return pd.Series(a).rolling(n, min_periods=n).sum().to_numpy()

    return {
        "k": rs(v.astype(float)),
        "x": rs(xs),
        "y": rs(ys),
        "xx": rs(xs * xs),
        "yy": rs(ys * ys),
        "xy": rs(xs * ys),
    }


def _ols(y: np.ndarray, x: np.ndarray, n: int, min_k: int) -> dict[str, np.ndarray]:
    """Rolling OLS y = a + b x on paired dates in the last n bars (needs >= min_k pairs)."""
    s = _pair_sums(y, x, n)
    k = s["k"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cxx = s["xx"] - s["x"] ** 2 / k
        cyy = s["yy"] - s["y"] ** 2 / k
        cxy = s["xy"] - s["x"] * s["y"] / k
        b = cxy / cxx
        ssr = np.maximum(cyy - b * cxy, 0.0)
        corr = cxy / np.sqrt(cxx * cyy)
    ok = (k >= min_k) & (cxx > 0)
    nan = np.full(len(y), np.nan)
    return {
        "b": np.where(ok, b, nan),
        "ssr": np.where(ok, ssr, nan),
        "k": k,
        "corr": np.where(ok & (cyy > 0), corr, nan),
    }


def _ema_trunc(x: np.ndarray, span: int, length: int) -> np.ndarray:
    """EMA with weights (1 - a)^j for j < length, renormalized; NaN unless all are finite."""
    a = 2.0 / (span + 1.0)
    w = (1.0 - a) ** np.arange(length)  # w[0] = newest
    out = np.full(len(x), np.nan)
    if len(x) < length:
        return out
    win = sliding_window_view(x, length)  # row i covers x[i .. i+length-1]
    out[length - 1 :] = win @ w[::-1] / w.sum()
    return out


def _windows(x: np.ndarray, n: int) -> np.ndarray | None:
    return sliding_window_view(x, n) if len(x) >= n else None


# ---------------------------------------------------------------- feature families
def register(cfg: PriceCfg) -> list[registry.FeatureSpec]:
    """Register every §5.1 feature for these windows (idempotent). Returns the specs."""
    keys: list[tuple[str, int]] = []

    def reg(name: str, group: str, formula: str, window, sources, fn: Callable) -> None:
        registry.feature(
            name, 1, kind="asset", group=group, formula=formula, window=window, sources=sources
        )(fn)
        keys.append((name, 1))

    bars = (BARS,)

    # -- returns
    for n in cfg.returns:
        reg(f"ret_{n}d", GROUP_RETURNS, f"ln(adj_close[t] / adj_close[t-{n}])", n, bars,
            lambda f, n=n: _lag_diff(f.logp, n))  # fmt: skip
    m = cfg.momentum
    reg("mom_12_1", GROUP_RETURNS, f"ln(adj_close[t-{m.skip}] / adj_close[t-{m.long}])", m.long,
        bars, lambda f: _lag(f.logp, m.skip) - _lag(f.logp, m.long))  # fmt: skip

    # -- volatility
    for n in cfg.rvol:
        reg(f"rvol_{n}d", GROUP_VOL,
            f"stdev(r[t-{n - 1}..t], ddof=1) * sqrt(ann); r = daily log return", n, bars,
            lambda f, n=n: _roll(f.r, n, "std") * math.sqrt(f.ann))  # fmt: skip
    for n in cfg.gk_vol:
        reg(f"gk_vol_{n}d", GROUP_VOL,
            f"sqrt(ann * mean over {n} bars of 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2)", n, bars,
            lambda f, n=n: _gk_vol(f, n))  # fmt: skip
    for n in cfg.downside_dev:
        reg(f"downside_dev_{n}d", GROUP_VOL,
            f"sqrt(mean(min(r, 0)^2) over {n} bars) * sqrt(ann)", n, bars,
            lambda f, n=n: _downside(f, n))  # fmt: skip
    for n in cfg.max_dd:
        reg(f"max_dd_{n}d", GROUP_VOL,
            f"min over the last {n} closes of close / running max - 1", n, bars,
            lambda f, n=n: _max_dd(f.close, n))  # fmt: skip

    # -- trend and technicals
    fa, sl = cfg.sma.fast, cfg.sma.slow
    reg(f"p_sma{fa}", GROUP_TREND, f"adj_close / SMA{fa}(adj_close) - 1", fa, bars,
        lambda f: f.close / _roll(f.close, fa, "mean") - 1.0)  # fmt: skip
    reg(f"p_sma{sl}", GROUP_TREND, f"adj_close / SMA{sl}(adj_close) - 1", sl, bars,
        lambda f: f.close / _roll(f.close, sl, "mean") - 1.0)  # fmt: skip
    reg(f"sma{fa}_sma{sl}", GROUP_TREND, f"SMA{fa}(adj_close) / SMA{sl}(adj_close) - 1", sl, bars,
        lambda f: _roll(f.close, fa, "mean") / _roll(f.close, sl, "mean") - 1.0)  # fmt: skip
    k = cfg.rsi
    reg(f"rsi_{k}", GROUP_TREND,
        f"100 * sum(gains) / (sum(gains) + sum(losses)) over the last {k} close changes "
        "(simple averages; 50 when flat)", k, bars, lambda f: _rsi(f.close, k))  # fmt: skip
    mc = cfg.macd
    reg("macd_hist_p", GROUP_TREND,
        f"(MACD - EMA{mc.signal}(MACD)) / adj_close; MACD = EMA{mc.fast} - EMA{mc.slow} of "
        f"adj_close; EMAs truncated to {mc.trunc_spans} x span bars", mc.slow, bars,
        lambda f: _macd_hist(f.close, mc.fast, mc.slow, mc.signal, mc.trunc_spans))  # fmt: skip
    bb = cfg.bollinger
    reg(f"bb_pctb_{bb.window}", GROUP_TREND,
        f"(adj_close - (SMA{bb.window} - {bb.k} sd)) / (2 x {bb.k} sd); sd ddof=0", bb.window,
        bars, lambda f: _pctb(f.close, bb.window, bb.k))  # fmt: skip
    a = cfg.atr
    reg(f"atr_{a}_pct", GROUP_TREND,
        f"mean(true range over {a} bars) / adj_close", a, bars,
        lambda f: _atr(f, a) / f.close)  # fmt: skip

    # -- liquidity
    vz = cfg.volume_z
    reg(f"vol_z_{vz}d", GROUP_LIQ,
        f"(ln V[t] - mean(ln V[t-{vz}..t-1])) / stdev(ln V[t-{vz}..t-1]); V split-adjusted",
        vz, bars, lambda f: _volume_z(f.volume, vz))  # fmt: skip
    am = cfg.amihud
    reg(f"amihud_{am.window}d", GROUP_LIQ,
        f"{am.scale:g} x mean(|r| / (raw close x raw volume)) over {am.window} bars",
        am.window, bars, lambda f: _amihud(f, am.window, am.scale))  # fmt: skip

    # -- market link
    frac = cfg.min_pair_frac
    for n in cfg.beta:
        src = (BARS, "ctx:bench")
        reg(f"beta_{n}d", GROUP_LINK,
            f"OLS slope of r on bench r over {n} bars (>= {frac:g} x {n} paired days); "
            "bench = SPY, coins BTC", n, src,
            lambda f, n=n: _ols(f.r, f.ctx["bench"], n, math.ceil(frac * n))["b"])  # fmt: skip
        reg(f"resid_vol_{n}d", GROUP_LINK,
            f"sqrt(SSR / (k - 2)) * sqrt(ann) of that regression over {n} bars", n, src,
            lambda f, n=n: _resid_vol(f, n, frac))  # fmt: skip
    cw = cfg.corr.window
    for key, target in cfg.corr.series.items():
        reg(f"corr_{key}_{cw}d", GROUP_LINK,
            f"Pearson corr of r with daily change of {target} over {cw} bars "
            f"(>= {frac:g} x {cw} paired days)", cw, (BARS, f"ctx:{key}"),
            lambda f, key=key: _ols(f.r, f.ctx[key], cw, math.ceil(frac * cw))["corr"])  # fmt: skip

    # -- mean reversion
    mz = cfg.mr_z
    reg(f"mr_z_{mz}d", GROUP_MR, f"(ln P - mean(ln P)) / stdev(ln P, ddof=1) over {mz} bars",
        mz, bars, lambda f: _mr_z(f.logp, mz))  # fmt: skip
    ou = cfg.ou
    reg(f"ou_halflife_{ou.window}d", GROUP_MR,
        f"-ln 2 / ln(1 + b), b = OLS slope of d ln P[t] on ln P[t-1] over {ou.window} bars; "
        f"b >= 0 or longer -> {ou.halflife_cap:g}", ou.window, bars,
        lambda f: _ou_halflife(f.logp, ou.window, ou.halflife_cap))  # fmt: skip
    h = cfg.hurst
    reg(f"hurst_vr_{h.window}d", GROUP_MR,
        f"ln(var(q-bar sums) / var(r)) / (2 ln q), q={h.q}, over the last {h.window} returns",
        h.window, bars, lambda f: _hurst_vr(f.r, h.window, h.q))  # fmt: skip

    # -- earnings proximity (stocks)
    e = cfg.earnings
    reg("earn_days_next", GROUP_EARN,
        "XNYS sessions from the bar date to the next report known when the bar is ready "
        f"(0 = after today's close; > {e.max_sessions_ahead} -> NaN)", None, (EARNINGS,),
        lambda f: _earnings(f)["days"])  # fmt: skip
    reg("earn_surprise_last", GROUP_EARN,
        "Nasdaq % EPS surprise of the latest report already out (earlier date, or today "
        "pre-market)", None, (EARNINGS,), lambda f: _earnings(f)["surprise"])  # fmt: skip
    reg("earn_date_approx", GROUP_EARN,
        "1 if the next report date comes from the backfill (date - 7 days, D-031), else 0",
        None, (EARNINGS,), lambda f: _earnings(f)["approx"])  # fmt: skip

    return [registry.get(*key) for key in keys]


# ---------------------------------------------------------------- feature math
def _gk_vol(f: PriceFrame, n: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(f.high / f.low)
        co = np.log(f.close / f.open)
    term = 0.5 * hl**2 - (2.0 * math.log(2.0) - 1.0) * co**2
    ok = (f.open > 0) & (f.high > 0) & (f.low > 0) & (f.close > 0)
    m = _roll(np.where(ok, term, np.nan), n, "mean")
    return np.sqrt(np.maximum(m, 0.0) * f.ann)


def _downside(f: PriceFrame, n: int) -> np.ndarray:
    return np.sqrt(_roll(np.minimum(f.r, 0.0) ** 2, n, "mean")) * math.sqrt(f.ann)


def _mr_z(logp: np.ndarray, n: int) -> np.ndarray:
    sd = _roll(logp, n, "std")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (logp - _roll(logp, n, "mean")) / sd, np.nan)


def _max_dd(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(c), np.nan)
    w = _windows(c, n)
    if w is None:
        return out
    peak = np.maximum.accumulate(w, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (w / peak - 1.0).min(axis=1)
    out[n - 1 :] = np.where(np.isfinite(w).all(axis=1), dd, np.nan)
    return out


def _rsi(c: np.ndarray, n: int) -> np.ndarray:
    d = _lag_diff(c, 1)
    g = _roll(np.where(np.isfinite(d), np.maximum(d, 0.0), np.nan), n, "sum")
    lo = _roll(np.where(np.isfinite(d), np.maximum(-d, 0.0), np.nan), n, "sum")
    tot = g + lo
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tot > 0, 100.0 * g / tot, np.where(np.isfinite(tot), 50.0, np.nan))


def _macd_hist(c: np.ndarray, fast: int, slow: int, signal: int, spans: int) -> np.ndarray:
    macd = _ema_trunc(c, fast, spans * fast) - _ema_trunc(c, slow, spans * slow)
    sig = _ema_trunc(macd, signal, spans * signal)
    return (macd - sig) / c


def _pctb(c: np.ndarray, n: int, k: float) -> np.ndarray:
    m = _roll(c, n, "mean")
    sd = _roll(c, n, "std", ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (c - (m - k * sd)) / (2.0 * k * sd), np.nan)


def _atr(f: PriceFrame, n: int) -> np.ndarray:
    pc = _lag(f.close, 1)
    tr = np.fmax(f.high - f.low, np.fmax(np.abs(f.high - pc), np.abs(f.low - pc)))
    tr = np.where(np.isfinite(pc), tr, np.nan)
    return _roll(tr, n, "mean")


def _volume_z(v: np.ndarray, n: int) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.where(v > 0, np.log(v), np.nan)
    prev = _lag(lv, 1)
    m = _roll(prev, n, "mean")
    sd = _roll(prev, n, "std")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, (lv - m) / sd, np.nan)


def _amihud(f: PriceFrame, n: int, scale: float) -> np.ndarray:
    dv = f.close_raw * f.volume_raw
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(dv > 0, np.abs(f.r) / dv, np.nan)
    return _roll(x, n, "mean") * scale


def _resid_vol(f: PriceFrame, n: int, frac: float) -> np.ndarray:
    o = _ols(f.r, f.ctx["bench"], n, math.ceil(frac * n))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(o["ssr"] / (o["k"] - 2.0)) * math.sqrt(f.ann)


def _ou_halflife(logp: np.ndarray, n: int, cap: float) -> np.ndarray:
    # centring on the first known value keeps the sums small; the slope ignores the shift
    fin = logp[np.isfinite(logp)]
    x = logp - fin[0] if len(fin) else logp
    o = _ols(_lag_diff(x, 1), _lag(x, 1), n, n)
    b = o["b"]
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = -math.log(2.0) / np.log1p(b)
    hl = np.where((b < 0) & (b > -1), np.minimum(hl, cap), np.where(b >= 0, cap, np.nan))
    return np.where(np.isfinite(b), hl, np.nan)


def _hurst_vr(r: np.ndarray, n: int, q: int) -> np.ndarray:
    v1 = _roll(r, n, "std") ** 2
    sq = _roll(r, q, "sum")  # sum of r[t-q+1..t]
    vq = _roll(sq, n - q + 1, "std") ** 2  # the q-sums lying inside the last n returns
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((v1 > 0) & (vq > 0), np.log(vq / v1) / (2.0 * math.log(q)), np.nan)


def _earnings(f: PriceFrame) -> dict[str, np.ndarray]:
    if "earn_days" in f._cache:
        return {k: f._cache[f"earn_{k}"] for k in ("days", "surprise", "approx")}
    n = f.n
    days = np.full(n, np.nan)
    approx = np.full(n, np.nan)
    surprise = np.full(n, np.nan)
    e = f.earnings
    if not e.empty:
        e = e.sort_values(["quarter_key", "released_at"]).reset_index(drop=True)
        rel = pd.to_datetime(e["released_at"], utc=True).dt.as_unit("ns").astype("int64")
        nxt = rel.groupby(e["quarter_key"]).shift(-1).fillna(NS_INF).astype("int64").to_numpy()
        rel = rel.to_numpy()
        rdates = np.array(e["report_date"].tolist(), dtype="datetime64[D]")
        pre = e["pre_market"].to_numpy(dtype=bool)
        sur = e["surprise"].to_numpy(dtype=float)
        apx = e["approx"].to_numpy(dtype=bool)
        sidx_d = np.searchsorted(f.sessions, f.dates, side="left")
        best_r = np.full(n, np.datetime64("NaT", "D"))
        cap = f.cfg.earnings.max_sessions_ahead
        for i in range(len(e)):
            live = (f.ready_ns >= rel[i]) & (f.ready_ns < nxt[i])
            if not live.any():
                continue
            r = rdates[i]
            ahead = live & ((r > f.dates) | ((r == f.dates) & ~pre[i]))
            dd = float(np.searchsorted(f.sessions, r, side="left")) - sidx_d
            better = ahead & (dd <= cap) & ~(dd >= days)  # NaN days -> take it
            days = np.where(better, dd, days)
            approx = np.where(better, float(apx[i]), approx)
            done = live & ~ahead & np.isfinite(sur[i])
            later = done & ((best_r < r) | np.isnat(best_r))
            best_r = np.where(later, r, best_r)
            surprise = np.where(later, sur[i], surprise)
    f._cache.update({"earn_days": days, "earn_surprise": surprise, "earn_approx": approx})
    return {"days": days, "surprise": surprise, "approx": approx}
