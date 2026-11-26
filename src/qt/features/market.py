"""Market vector (component doc 03 section 5.3): macro, cross-asset, GLI, breadth.

Each feature is a function of a :class:`MarketState` at one decision time T and returns one float.
The state holds only what is known at T:

- macro: a :class:`qt.features.inputs.MacroState` advanced to T (the ALFRED vintages released by
  T; later revisions stay invisible).
- prices: adjusted closes per ticker with ``released_at``, cut at T. Adjusted as of the run's
  last T, which only rescales history by a constant, so returns, ratios and z-scores of log levels
  are unaffected (as for the context series in ``price.py``, D-036 item 5).
- breadth: a per-date table built from the index members' bars (:func:`breadth_table`), with the
  time each row was ready.

"n-day" windows count XNYS sessions: a series is sampled at each session date on or before T as
its newest observation dated on or before that session, so weekly and monthly series forward-fill
and never back-fill. Crypto ratios use the coins' own daily observations instead (they trade on
weekends).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from qt.common.config import MarketCfg
from qt.features import registry
from qt.features.inputs import ET, MacroState

G_RATES = "macro_rates"
G_GROWTH = "macro_growth_inflation"
G_CREDIT_VOL = "credit_vol"
G_CROSS = "cross_asset"
G_FRAMEWORK = "framework"
G_RISK = "risk_appetite"
G_BREADTH = "breadth"
G_GLI = "gli"
G_CRYPTO = "crypto_ratio"
GROUPS = (G_RATES, G_GROWTH, G_CREDIT_VOL, G_CROSS, G_FRAMEWORK, G_RISK, G_BREADTH, G_GLI,
          G_CRYPTO)  # fmt: skip


@dataclass
class MarketState:
    asof: pd.Timestamp
    macro: MacroState  # already advanced to asof
    prices: Mapping[str, pd.DataFrame]  # ticker -> date, value, released_at (all rows)
    sessions: np.ndarray  # XNYS sessions, datetime64[D]
    cfg: MarketCfg
    breadth: pd.DataFrame | None = None  # index date; columns + ready_ns
    events: Any = None  # qt.features.events.EventInputs (event clock, step 4b)
    crypto: Any = None  # qt.features.crypto.CryptoInputs (crypto structure, step 4b)
    regime: Any = None  # qt.features.regime.hmm.HmmInputs (fits; step 5)
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        t = pd.Timestamp(self.asof)
        if t.tzinfo is None:
            raise ValueError("asof must be timezone-aware")
        self.asof = t
        self.today = np.datetime64(t.tz_convert(ET).date(), "D")
        self.asof_ns = int(t.tz_convert("UTC").as_unit("ns").value)
        self.sess = self.sessions[self.sessions <= self.today]

    # ------------------------------------------------------------ raw access (known at T)
    def mac(self, sid: str) -> pd.Series:
        """Macro series as known at T, NaN observations dropped; index datetime64[D]."""
        key = ("mac", sid)
        if key not in self._cache:
            s = self.macro.series(sid).dropna()
            s.index = pd.Index(np.array(list(s.index), dtype="datetime64[D]"))
            self._cache[key] = s
        return self._cache[key]

    def px(self, ticker: str) -> pd.Series:
        """Adjusted closes released by T; index datetime64[D]."""
        key = ("px", ticker)
        if key not in self._cache:
            f = self.prices.get(ticker)
            if f is None or f.empty:
                s = pd.Series(dtype=float)
            else:
                f = f[f["released_at"] <= self.asof]
                f = f.sort_values("date").drop_duplicates("date", keep="last")
                s = pd.Series(f["value"].to_numpy(dtype=float),
                              index=pd.Index(np.array(f["date"].tolist(), dtype="datetime64[D]")))
            self._cache[key] = s[s > 0]
        return self._cache[key]

    def series(self, src: str) -> pd.Series:
        """``px:<ticker>`` or a FRED series id."""
        return self.px(src[3:]) if src.startswith("px:") else self.mac(src)

    # ------------------------------------------------------------ sampling
    def sampled(self, src: str, n: int) -> np.ndarray:
        """The series at the last ``n`` sessions <= T (newest observation on/before each)."""
        key = ("smp", src, n)
        if key not in self._cache:
            s = self.series(src)
            out = np.full(n, np.nan)
            if len(s) and len(self.sess) >= n:
                d = self.sess[-n:]
                pos = np.searchsorted(s.index.to_numpy(), d, side="right") - 1
                vals = s.to_numpy()
                out = np.where(pos >= 0, vals[np.maximum(pos, 0)], np.nan)
            self._cache[key] = out
        return self._cache[key]

    def obs(self, src: str, n: int) -> np.ndarray:
        """The last ``n`` observations of the series (its own dates)."""
        s = self.series(src)
        return s.to_numpy()[-n:] if len(s) >= n else np.full(n, np.nan)

    def latest(self, src: str) -> float:
        s = self.series(src)
        if not len(s) or not len(self.sess):
            return math.nan
        return float(s.iloc[-1])

    def value_on(self, src: str, day: np.datetime64) -> float:
        s = self.series(src)
        pos = int(np.searchsorted(s.index.to_numpy(), day, side="right")) - 1
        return float(s.iloc[pos]) if pos >= 0 else math.nan


# ---------------------------------------------------------------- numeric helpers
def zlast(x: np.ndarray) -> float:
    if not np.isfinite(x).all() or len(x) < 3:
        return math.nan
    sd = float(np.std(x, ddof=1))
    return (float(x[-1]) - float(np.mean(x))) / sd if sd > 0 else math.nan


def chg(x: np.ndarray) -> float:
    return float(x[-1] - x[0]) if np.isfinite(x[[0, -1]]).all() else math.nan


def logret(x: np.ndarray) -> float:
    return math.log(x[-1] / x[0]) if np.isfinite(x[[0, -1]]).all() and (x[[0, -1]] > 0).all() \
        else math.nan  # fmt: skip


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return math.nan
    a, b = a[ok], b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _monthly_at(s: pd.Series, months_back: int) -> float:
    """Value at the period ``months_back`` months before the newest one (exact period match)."""
    if not len(s):
        return math.nan
    last = pd.Timestamp(s.index[-1])
    want = np.datetime64((last - pd.DateOffset(months=months_back)).date(), "D")
    hit = s.index == want
    return float(s[hit].iloc[0]) if hit.any() else math.nan


# ---------------------------------------------------------------- feature math
def curve(m: MarketState, long: str, short: str, n: int | None = None) -> float:
    if n is None:
        return m.latest(long) - m.latest(short)
    a, b = m.sampled(long, n + 1), m.sampled(short, n + 1)
    return chg(a - b)


def infl_momentum(m: MarketState, sid: str) -> float:
    s = m.mac(sid)
    c = m.cfg.infl_months
    if not len(s):
        return math.nan
    now = float(s.iloc[-1])
    short, long = _monthly_at(s, c.short), _monthly_at(s, c.long)
    if not (short > 0 and long > 0):
        return math.nan
    return (now / short) ** (12 / c.short) - 1 - (now / long - 1)


def sahm_values(m: MarketState) -> pd.Series:
    """The Sahm indicator for every calendar month of UNRATE as known at T (index: Period[M]).

    Month i: 3-month average of UNRATE minus its minimum over the prior 12 months. A missing month
    (e.g. October 2025, not published in the shutdown) is skipped: each average needs all but one
    of its months, and the prior window all but one of its averages.
    """
    key = ("sahm_values",)
    if key in m._cache:
        return m._cache[key]
    s = m.mac("UNRATE")
    c = m.cfg.sahm
    if not len(s):
        out = pd.Series(dtype=float)
    else:
        idx = pd.period_range(pd.Timestamp(s.index[0]), pd.Timestamp(s.index[-1]), freq="M")
        monthly = pd.Series(s.to_numpy(), index=pd.PeriodIndex(pd.to_datetime(s.index), freq="M"))
        monthly = monthly[~monthly.index.duplicated(keep="last")].reindex(idx)
        avg = monthly.rolling(c.avg_months, min_periods=max(c.avg_months - 1, 1)).mean().to_numpy()
        vals = np.full(len(avg), np.nan)
        lb = c.lookback_months
        for i in range(lb, len(avg)):
            prior = avg[i - lb : i]
            if np.isfinite(avg[i]) and np.isfinite(prior).sum() >= lb - 1:
                vals[i] = avg[i] - np.nanmin(prior)
        out = pd.Series(vals, index=idx)
    m._cache[key] = out
    return out


def sahm(m: MarketState) -> float:
    """The newest month of :func:`sahm_values`."""
    v = sahm_values(m)
    return float(v.iloc[-1]) if len(v) else math.nan


def gdp_qoq_values(m: MarketState) -> pd.Series:
    """(GDPC1 / previous quarter)^4 - 1 for every quarter known at T; NaN where the previous
    quarter is missing."""
    s = m.mac("GDPC1")
    if len(s) < 2:
        return pd.Series(dtype=float)
    per = pd.PeriodIndex(pd.to_datetime(s.index), freq="Q")
    v = s.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    consecutive = (per[1:] - per[:-1]).map(lambda d: d.n) == 1
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.where(consecutive, (v[1:] / v[:-1]) ** 4 - 1, np.nan)
    return pd.Series(out, index=per)


def gdp_qoq(m: MarketState) -> float:
    s = m.mac("GDPC1")
    if len(s) < 2 or _monthly_at(s, 3) != float(s.iloc[-2]):
        return math.nan
    return (float(s.iloc[-1]) / float(s.iloc[-2])) ** 4 - 1


def claims_yoy(m: MarketState) -> float:
    s = m.mac("ICSA")
    c = m.cfg.claims
    if len(s) < c.avg_weeks:
        return math.nan
    now = float(s.iloc[-c.avg_weeks :].mean())
    cut = s.index[-1] - np.timedelta64(c.yoy_days, "D")
    past = s[s.index <= cut]
    if len(past) < c.avg_weeks or (cut - past.index[-1]) > np.timedelta64(6, "D"):
        return math.nan
    return now / float(past.iloc[-c.avg_weeks :].mean()) - 1


def days_change(m: MarketState, sid: str, days: int) -> float:
    s = m.mac(sid)
    if not len(s):
        return math.nan
    return float(s.iloc[-1]) - m.value_on(sid, s.index[-1] - np.timedelta64(days, "D"))


def rets(m: MarketState, src: str, n: int) -> np.ndarray:
    x = m.sampled(src, n + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(x))


def gold_resid_z(m: MarketState) -> float:
    c = m.cfg.gold_resid
    g = rets(m, f"px:{m.cfg.tickers['gold']}", c.fit_window)
    dr = np.diff(m.sampled("DFII10", c.fit_window + 1))
    if not (np.isfinite(g).all() and np.isfinite(dr).all()) or np.std(dr) == 0:
        return math.nan
    b, a = np.polyfit(dr, g, 1)
    e = g - (a + b * dr)
    sd = float(np.std(e, ddof=2))
    return float(e[-c.horizon :].sum()) / (sd * math.sqrt(c.horizon)) if sd > 0 else math.nan


def gli_value(m: MarketState, day: np.datetime64) -> float:
    total = 0.0
    for t in m.cfg.gli.terms:
        v = m.value_on(t.series, day)
        if t.fx:
            fx = m.value_on(t.fx, day)
            v = v * fx if t.fx_op == "mul" else v / fx
        if not math.isfinite(v):
            return math.nan
        total += t.factor * v
    return total


def gli_sampled(m: MarketState, n: int) -> np.ndarray:
    key = ("gli", n)
    if key not in m._cache:
        d = m.sess[-n:] if len(m.sess) >= n else np.array([], dtype="datetime64[D]")
        m._cache[key] = (np.array([gli_value(m, x) for x in d]) if len(d) == n
                         else np.full(n, np.nan))  # fmt: skip
    return m._cache[key]


def gli_change(m: MarketState) -> float:
    if not len(m.sess):
        return math.nan
    now = gli_value(m, m.sess[-1])
    past = gli_value(m, m.sess[-1] - np.timedelta64(m.cfg.gli.window_days, "D"))
    return now / past - 1 if past > 0 else math.nan


def breadth_rows(m: MarketState, n: int) -> pd.DataFrame | None:
    b = m.breadth
    if b is None or b.empty:
        return None
    b = b[b["ready_ns"] <= m.asof_ns]
    return b.iloc[-n:] if len(b) >= n else None


def breadth_last(m: MarketState, col: str) -> float:
    b = breadth_rows(m, 1)
    return math.nan if b is None else float(b[col].iloc[-1])


def breadth_adl(m: MarketState, n: int) -> float:
    b = breadth_rows(m, n)
    return math.nan if b is None else float(b["adv_dec"].sum())


def crypto_ratio(m: MarketState, n: int, z: bool) -> float:
    tk = m.cfg.tickers
    eth, btc = m.px(tk["eth"]), m.px(tk["btc"])
    r = (eth / btc).dropna()
    if len(r) < n + 1:
        return math.nan
    x = np.log(r.to_numpy()[-(n + 1) :] if not z else r.to_numpy()[-n:])
    return zlast(x) if z else float(x[-1] - x[0])


# ---------------------------------------------------------------- breadth table
def breadth_table(
    closes: pd.DataFrame, ready_ns: pd.DataFrame, members: pd.DataFrame, sma: int, hl: int
) -> pd.DataFrame:
    """Per date, over that date's index members with data.

    ``closes`` / ``ready_ns``: dates x assets (adjusted closes, and when each bar was released);
    ``members``: dates x assets booleans (membership as of that date's decision time).
    Columns: ``above_sma`` (share above their ``sma``-day mean), ``nh_nl`` (share at a ``hl``-day
    high minus share at a ``hl``-day low), ``adv_dec`` ((advancers - decliners) / N), ``ready_ns``.
    """
    c = closes.sort_index()
    m = members.reindex_like(c).fillna(False).astype(bool)
    mean = c.rolling(sma, min_periods=sma).mean()
    hi = c.rolling(hl, min_periods=hl).max()
    lo = c.rolling(hl, min_periods=hl).min()
    chg_ = c.diff()
    ok = m & c.notna()
    n_sma = (ok & mean.notna()).sum(axis=1)
    n_hl = (ok & hi.notna()).sum(axis=1)
    n_ad = (ok & chg_.notna()).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        above = ((c > mean) & ok).sum(axis=1) / n_sma
        nh_nl = (((c >= hi) & ok & hi.notna()).sum(axis=1)
                 - ((c <= lo) & ok & lo.notna()).sum(axis=1)) / n_hl  # fmt: skip
        ad = (((chg_ > 0) & ok).sum(axis=1) - ((chg_ < 0) & ok).sum(axis=1)) / n_ad
    rd = ready_ns.reindex_like(c).where(ok).max(axis=1)
    out = pd.DataFrame({"above_sma": above.where(n_sma > 0), "nh_nl": nh_nl.where(n_hl > 0),
                        "adv_dec": ad.where(n_ad > 0), "ready_ns": rd})  # fmt: skip
    return out.dropna(subset=["ready_ns"]).astype({"ready_ns": "int64"})


# ---------------------------------------------------------------- registration
def register(cfg: MarketCfg) -> list[registry.FeatureSpec]:
    """Register every §5.3 feature built here (idempotent). Returns the specs."""
    keys: list[tuple[str, int]] = []
    n_c, n_r, n_z, n_k, n_q = (cfg.chg_window, cfg.ret_window, cfg.z_window, cfg.corr_window,
                               cfg.ratio_window)  # fmt: skip
    tk = cfg.tickers

    def reg(name, group, formula, sources, fn: Callable[[MarketState], float], window=None):
        registry.feature(name, 1, kind="market", group=group, formula=formula, window=window,
                         sources=tuple(sources))(fn)  # fmt: skip
        keys.append((name, 1))

    def px(k):
        return f"px:{tk[k]}"

    # -- rates and policy
    reg("curve_10y2y", G_RATES, "DGS10 - DGS2", ["DGS10", "DGS2"],
        lambda m: curve(m, "DGS10", "DGS2"))  # fmt: skip
    reg(f"curve_10y2y_chg{n_c}", G_RATES, f"{n_c}-session change of DGS10 - DGS2",
        ["DGS10", "DGS2"], lambda m: curve(m, "DGS10", "DGS2", n_c), n_c)  # fmt: skip
    reg("curve_10y3m", G_RATES, "DGS10 - DGS3MO", ["DGS10", "DGS3MO"],
        lambda m: curve(m, "DGS10", "DGS3MO"))  # fmt: skip
    reg(f"curve_10y3m_chg{n_c}", G_RATES, f"{n_c}-session change of DGS10 - DGS3MO",
        ["DGS10", "DGS3MO"], lambda m: curve(m, "DGS10", "DGS3MO", n_c), n_c)  # fmt: skip
    reg("real_yield_10y", G_RATES, "DFII10", ["DFII10"], lambda m: m.latest("DFII10"))
    reg(f"real_yield_10y_chg{n_c}", G_RATES, f"{n_c}-session change of DFII10", ["DFII10"],
        lambda m: chg(m.sampled("DFII10", n_c + 1)), n_c)  # fmt: skip
    reg("policy_path", G_RATES, "DGS2 - DFF", ["DGS2", "DFF"], lambda m: curve(m, "DGS2", "DFF"))
    reg("breakeven_10y", G_RATES, "T10YIE", ["T10YIE"], lambda m: m.latest("T10YIE"))
    reg(f"breakeven_10y_chg{n_c}", G_RATES, f"{n_c}-session change of T10YIE", ["T10YIE"],
        lambda m: chg(m.sampled("T10YIE", n_c + 1)), n_c)  # fmt: skip
    reg("fwd_5y5y", G_RATES, "T5YIFR", ["T5YIFR"], lambda m: m.latest("T5YIFR"))
    reg("term_premium_10y", G_RATES, "THREEFYTP10 (ACM)", ["THREEFYTP10"],
        lambda m: m.latest("THREEFYTP10"))  # fmt: skip
    reg(f"term_premium_10y_chg{n_c}", G_RATES, f"{n_c}-session change of THREEFYTP10",
        ["THREEFYTP10"], lambda m: chg(m.sampled("THREEFYTP10", n_c + 1)), n_c)  # fmt: skip

    # -- growth and inflation
    im = cfg.infl_months
    for sid, nm in (("CPILFESL", "core_cpi"), ("PCEPILFE", "core_pce")):
        reg(f"infl_mom_{nm}", G_GROWTH,
            f"{sid}: {im.short}-month annualized - {im.long}-month change", [sid],
            lambda m, sid=sid: infl_momentum(m, sid))  # fmt: skip
    sc = cfg.sahm
    reg("sahm", G_GROWTH, f"UNRATE {sc.avg_months}-month avg - min of it over the prior "
        f"{sc.lookback_months} months", ["UNRATE"], sahm)  # fmt: skip
    reg("gdp_qoq_ann", G_GROWTH, "(GDPC1 / previous quarter)^4 - 1", ["GDPC1"], gdp_qoq)
    reg("nfci", G_GROWTH, "NFCI", ["NFCI"], lambda m: m.latest("NFCI"))
    reg("nfci_chg4w", G_GROWTH, f"NFCI - NFCI {cfg.nfci_chg_days} days before its latest week",
        ["NFCI"], lambda m: days_change(m, "NFCI", cfg.nfci_chg_days))  # fmt: skip
    cc = cfg.claims
    reg("claims_4w_yoy", G_GROWTH, f"mean of last {cc.avg_weeks} ICSA / same {cc.yoy_days} "
        "days earlier - 1", ["ICSA"], claims_yoy)  # fmt: skip

    # -- credit and volatility
    reg("credit_baa10y", G_CREDIT_VOL, "BAA10Y", ["BAA10Y"], lambda m: m.latest("BAA10Y"))
    reg(f"credit_baa10y_z{n_z}", G_CREDIT_VOL, f"z of BAA10Y over {n_z} sessions", ["BAA10Y"],
        lambda m: zlast(m.sampled("BAA10Y", n_z)), n_z)  # fmt: skip
    reg(f"credit_baa10y_chg{n_c}", G_CREDIT_VOL, f"{n_c}-session change of BAA10Y", ["BAA10Y"],
        lambda m: chg(m.sampled("BAA10Y", n_c + 1)), n_c)  # fmt: skip
    reg("vix_vix3m", G_CREDIT_VOL, "VIX / VIX3M (closes)", [px("vix"), px("vix3m")],
        lambda m: m.latest(px("vix")) / m.latest(px("vix3m")))  # fmt: skip
    reg(f"vix_z{n_z}", G_CREDIT_VOL, f"z of VIX over {n_z} sessions", [px("vix")],
        lambda m: zlast(m.sampled(px("vix"), n_z)), n_z)  # fmt: skip
    reg("move", G_CREDIT_VOL, "MOVE index", [px("move")], lambda m: m.latest(px("move")))
    reg(f"move_z{n_z}", G_CREDIT_VOL, f"z of MOVE over {n_z} sessions", [px("move")],
        lambda m: zlast(m.sampled(px("move"), n_z)), n_z)  # fmt: skip
    reg("move_vix", G_CREDIT_VOL, "MOVE / VIX", [px("move"), px("vix")],
        lambda m: m.latest(px("move")) / m.latest(px("vix")))  # fmt: skip
    reg(f"skew_z{n_z}", G_CREDIT_VOL, f"z of SKEW over {n_z} sessions", [px("skew")],
        lambda m: zlast(m.sampled(px("skew"), n_z)), n_z)  # fmt: skip
    reg(f"vvix_z{n_z}", G_CREDIT_VOL, f"z of VVIX over {n_z} sessions", [px("vvix")],
        lambda m: zlast(m.sampled(px("vvix"), n_z)), n_z)  # fmt: skip
    reg("vvix_vix", G_CREDIT_VOL, "VVIX / VIX", [px("vvix"), px("vix")],
        lambda m: m.latest(px("vvix")) / m.latest(px("vix")))  # fmt: skip

    # -- cross-asset
    reg(f"dxy_ret{n_r}", G_CROSS, f"{n_r}-session log return of DXY", [px("dxy")],
        lambda m: logret(m.sampled(px("dxy"), n_r + 1)), n_r)  # fmt: skip
    reg(f"dxy_z{n_z}", G_CROSS, f"z of ln DXY over {n_z} sessions", [px("dxy")],
        lambda m: zlast(np.log(m.sampled(px("dxy"), n_z))), n_z)  # fmt: skip
    reg(f"spy_tlt_corr{n_k}", G_CROSS, f"corr of SPY and TLT daily log returns, {n_k} sessions",
        [px("spy"), px("tlt")],
        lambda m: corr(rets(m, px("spy"), n_k), rets(m, px("tlt"), n_k)), n_k)  # fmt: skip
    gr = cfg.gold_resid
    reg("gold_resid_z", G_CROSS,
        f"{gr.horizon}-session sum of gold return residuals vs OLS on dDFII10 over "
        f"{gr.fit_window} sessions / (sd x sqrt {gr.horizon})", [px("gold"), "DFII10"],
        gold_resid_z, gr.fit_window)  # fmt: skip
    reg(f"vnq_spy_ret{n_q}", G_CROSS, f"{n_q}-session log return of VNQ - SPY",
        [px("vnq"), px("spy")],
        lambda m: logret(m.sampled(px("vnq"), n_q + 1)) - logret(m.sampled(px("spy"), n_q + 1)),
        n_q)  # fmt: skip
    reg(f"wti_ret{n_r}", G_CROSS, f"{n_r}-session log return of WTI", [px("wti")],
        lambda m: logret(m.sampled(px("wti"), n_r + 1)), n_r)  # fmt: skip
    reg(f"wti_z{n_z}", G_CROSS, f"z of ln WTI over {n_z} sessions", [px("wti")],
        lambda m: zlast(np.log(m.sampled(px("wti"), n_z))), n_z)  # fmt: skip

    # -- framework four
    def cg(m, n):
        a, b = m.sampled(px("copper"), n), m.sampled(px("gold"), n)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(a / b)

    reg(f"copper_gold_chg{n_c}", G_FRAMEWORK, f"{n_c}-session change of ln(copper / gold)",
        [px("copper"), px("gold")], lambda m: chg(cg(m, n_c + 1)), n_c)  # fmt: skip
    reg(f"copper_gold_z{n_z}", G_FRAMEWORK, f"z of ln(copper / gold) over {n_z} sessions",
        [px("copper"), px("gold")], lambda m: zlast(cg(m, n_z)), n_z)  # fmt: skip
    reg(f"aud_ret{n_r}", G_FRAMEWORK, f"{n_r}-session log return of DEXUSAL (USD per AUD)",
        ["DEXUSAL"], lambda m: logret(m.sampled("DEXUSAL", n_r + 1)), n_r)  # fmt: skip
    reg("usdjpy_carry", G_FRAMEWORK, "DGS3MO - IR3TIB01JPM156N (Japan 3-month)",
        ["DGS3MO", "IR3TIB01JPM156N"], lambda m: curve(m, "DGS3MO", "IR3TIB01JPM156N"))  # fmt: skip
    reg(f"usdjpy_ret{n_r}", G_FRAMEWORK, f"{n_r}-session log return of DEXJPUS (JPY per USD)",
        ["DEXJPUS"], lambda m: logret(m.sampled("DEXJPUS", n_r + 1)), n_r)  # fmt: skip

    # -- risk appetite
    for a, b in cfg.ratios:
        reg(f"ratio_{a.lower()}_{b.lower()}_{n_q}d", G_RISK,
            f"{n_q}-session log return of {a} / {b}", [f"px:{a}", f"px:{b}"],
            lambda m, a=a, b=b: logret(m.sampled(f"px:{a}", n_q + 1))
            - logret(m.sampled(f"px:{b}", n_q + 1)), n_q)  # fmt: skip

    # -- breadth (own universe)
    reg("breadth_above_sma200", G_BREADTH, "share of index members above their 200-day mean",
        ["breadth"], lambda m: breadth_last(m, "above_sma"))  # fmt: skip
    reg("breadth_nh_nl", G_BREADTH, "share of members at a 252-day high - share at a low",
        ["breadth"], lambda m: breadth_last(m, "nh_nl"))  # fmt: skip
    reg(f"breadth_adl_chg{n_c}", G_BREADTH,
        f"advance/decline line change over {n_c} sessions: sum of (adv - dec) / N", ["breadth"],
        lambda m: breadth_adl(m, n_c), n_c)  # fmt: skip

    # -- GLI
    terms = " + ".join(f"{t.factor:g} x {t.series}" + (f" {'x' if t.fx_op == 'mul' else '/'} "
                                                      f"{t.fx}" if t.fx else "")
                       for t in cfg.gli.terms)  # fmt: skip
    gsrc = [t.series for t in cfg.gli.terms] + [t.fx for t in cfg.gli.terms if t.fx]
    reg("gli", G_GLI, f"USD bn = {terms}", gsrc,
        lambda m: gli_value(m, m.sess[-1]) if len(m.sess) else math.nan)  # fmt: skip
    reg("gli_chg13w", G_GLI, f"GLI / GLI {cfg.gli.window_days} days earlier - 1", gsrc,
        gli_change)  # fmt: skip
    reg(f"gli_z{n_z}", G_GLI, f"z of GLI over {n_z} sessions", gsrc,
        lambda m: zlast(gli_sampled(m, n_z)), n_z)  # fmt: skip

    # -- crypto ratio (bars we already have)
    reg(f"eth_btc_chg{n_c}", G_CRYPTO, f"change of ln(ETH / BTC) over {n_c} daily bars",
        [px("eth"), px("btc")], lambda m: crypto_ratio(m, n_c, z=False), n_c)  # fmt: skip
    reg(f"eth_btc_z{n_z}", G_CRYPTO, f"z of ln(ETH / BTC) over {n_z} daily bars",
        [px("eth"), px("btc")], lambda m: crypto_ratio(m, n_z, z=True), n_z)  # fmt: skip

    return [registry.get(*k) for k in keys]


def compute(state: MarketState, specs: list[registry.FeatureSpec]) -> dict:
    out = {}
    for s in specs:
        with np.errstate(all="ignore"):
            v = s.fn(state)
        out[s.key] = float(v) if v is not None and math.isfinite(v) else math.nan
    return out
