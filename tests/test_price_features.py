"""Phase 3 step 2: per-asset price features (component doc 03 sections 5.1 and 11).

- ``test_feature_hand_calcs``: every registered price feature against a naive, loop-based
  re-implementation on a small fixture, at early dates (window not full -> NaN) and late ones.
  A registered feature without a hand case fails.
- ``test_no_lookahead_features``: mutating every input row released after T (bars, corporate
  actions, context series, earnings) leaves each value at T unchanged.
- ``test_backfill_pass_matches_daily``: one pass over the history equals the daily computation
  at each decision time (what lets the backfill stand in for daily runs).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from qt.common import config
from qt.common.calendar import TradingCalendar
from qt.features import price, registry

CFG = config.load().config.features
PC = CFG.price
SPECS = price.register(PC)
SESSIONS = np.array(TradingCalendar().cal.sessions.date, dtype="datetime64[D]")
N_BARS = 330


def _released(d) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, 21, 15, tzinfo=UTC))


def _fixture(seed: int = 7):
    rng = np.random.default_rng(seed)
    days = [pd.Timestamp(x).date() for x in SESSIONS[SESSIONS >= np.datetime64("2021-01-04")]]
    days = days[:N_BARS]
    n = len(days)
    m = rng.normal(0.0003, 0.011, n)  # bench returns
    r = 0.0002 + 1.2 * m + rng.normal(0, 0.012, n)
    close = 50.0 * np.exp(np.cumsum(r))
    # a 2-for-1 split on day 200: raw prices halve from there (and volume doubles)
    split_at = 200
    raw_close = close.copy()
    raw_close[split_at:] /= 2.0
    open_ = raw_close * np.exp(rng.normal(0, 0.004, n))
    high = np.maximum(open_, raw_close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    low = np.minimum(open_, raw_close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    vol = rng.lognormal(13, 0.4, n)
    vol[split_at:] *= 2.0
    bars = pd.DataFrame(
        {
            "asset_id": 1,
            "date": days,
            "open": open_,
            "high": high,
            "low": low,
            "close": raw_close,
            "volume": vol,
            "source": "alpaca",
            "released_at": [_released(d) for d in days],
            "asset_class": "equity",
        }
    )
    split = {
        "external_id": "s1",
        "kind": "ca_forward_splits",
        "symbol": "XYZ",
        "ex_date": days[split_at],
        "released_at": _released(days[split_at - 15]),
        "old_rate": 1.0,
        "new_rate": 2.0,
        "rate": np.nan,
        "special": False,
        "asset_id": 1,
        "manual": False,
    }
    div = {
        **split,
        "external_id": "d1",
        "kind": "ca_cash_dividends",
        "ex_date": days[120],
        "released_at": _released(days[100]),
        "old_rate": np.nan,
        "new_rate": np.nan,
        "rate": 0.4,
    }
    actions = pd.DataFrame([split, div])

    def ctx_frame(values, keep, lag_min=0):
        rel = [_released(d) + pd.Timedelta(minutes=lag_min) for d in days]
        f = pd.DataFrame({"date": days, "value": values, "released_at": rel})
        return f[keep].reset_index(drop=True)

    keep_b = rng.random(n) > 0.05
    keep_d = rng.random(n) > 0.08
    bench = ctx_frame(400 * np.exp(np.cumsum(m)), keep_b)
    dxy = ctx_frame(100 * np.exp(np.cumsum(rng.normal(0, 0.004, n) - 0.2 * m)), keep_d)
    y10 = ctx_frame(4.0 + np.cumsum(rng.normal(0, 0.03, n)), np.ones(n, bool), lag_min=45)
    ctx = {
        "bench": price.CtxSeries(bench),
        "spy": price.CtxSeries(bench),
        "dxy": price.CtxSeries(dxy),
        "y10": price.CtxSeries(y10, kind="diff"),
    }

    def erow(q, rep_i, rel_i, pre, surprise, approx):
        return {
            "asset_id": 1,
            "quarter_key": q,
            "report_date": days[rep_i] if rep_i < n else days[-1] + pd.Timedelta(days=40),
            "pre_market": pre,
            "surprise": surprise,
            "approx": approx,
            "released_at": _released(days[rel_i]) - pd.Timedelta(hours=2),
        }

    earnings = pd.DataFrame(
        [
            erow("XYZ:Q1", 60, 55, False, 4.2, True),
            erow("XYZ:Q2", 125, 120, True, -3.1, True),
            # Q3 first announced for day 190, then moved to day 186 (seen on day 180)
            erow("XYZ:Q3", 190, 150, False, np.nan, False),
            erow("XYZ:Q3", 186, 180, False, 7.5, False),
            erow("XYZ:Q4", 250, 245, False, 1.0, True),
            erow("XYZ:Q5", 315, 290, True, np.nan, False),
            erow("XYZ:Q6", n + 5, 320, False, np.nan, False),
        ]
    )
    return bars, actions, ctx, earnings


BARS, ACTIONS, CTX, EARN = _fixture()
ASOF_END = BARS["released_at"].max() + pd.Timedelta(hours=3)


def _frame(asof, bars=BARS, actions=ACTIONS, ctx=CTX, earnings=EARN) -> price.PriceFrame:
    f = price.prepare(
        bars, asof, PC, CFG.adjust, asset_class="equity", actions=actions, ctx=ctx,
        earnings=earnings, sessions=SESSIONS,
    )  # fmt: skip
    assert f is not None
    return f


FULL = _frame(ASOF_END)


# ---------------------------------------------------------------- naive references
def _r(c, i):
    return math.log(c[i] / c[i - 1]) if i >= 1 else math.nan


def _rets(f, t, n):
    """The n daily log returns ending at t, or None if they don't all exist."""
    if t - n < 0:
        return None
    return np.array([_r(f.close, i) for i in range(t - n + 1, t + 1)])


def _sma(x, t, n):
    return float(np.mean(x[t - n + 1 : t + 1])) if t - n + 1 >= 0 else math.nan


def _ema(x, t, span, length):
    if t - length + 1 < 0:
        return math.nan
    a = 2 / (span + 1)
    num = den = 0.0
    for j in range(length):
        w = (1 - a) ** j
        num += w * x[t - j]
        den += w
    return num / den


def _pairs(f, t, n, key):
    xs, ys = [], []
    for i in range(max(t - n + 1, 0), t + 1):
        a, b = _r(f.close, i), f.ctx[key][i]
        if np.isfinite(a) and np.isfinite(b):
            ys.append(a)
            xs.append(b)
    return np.array(xs), np.array(ys)


def _need_pairs(n):
    return math.ceil(PC.min_pair_frac * n)


def h_ret(n):
    return lambda f, t: math.log(f.close[t] / f.close[t - n]) if t >= n else math.nan


def h_mom(f, t):
    m = PC.momentum
    return math.log(f.close[t - m.skip] / f.close[t - m.long]) if t >= m.long else math.nan


def h_rvol(n):
    def g(f, t):
        r = _rets(f, t, n)
        return math.nan if r is None else float(np.std(r, ddof=1)) * math.sqrt(252)

    return g


def h_gk(n):
    def g(f, t):
        if t - n + 1 < 0:
            return math.nan
        s = 0.0
        for i in range(t - n + 1, t + 1):
            s += (
                0.5 * math.log(f.high[i] / f.low[i]) ** 2
                - (2 * math.log(2) - 1) * math.log(f.close[i] / f.open[i]) ** 2
            )
        return math.sqrt(max(s / n, 0) * 252)

    return g


def h_down(n):
    def g(f, t):
        r = _rets(f, t, n)
        if r is None:
            return math.nan
        return math.sqrt(sum(min(x, 0) ** 2 for x in r) / n) * math.sqrt(252)

    return g


def h_dd(n):
    def g(f, t):
        if t - n + 1 < 0:
            return math.nan
        peak, worst = -math.inf, 0.0
        for i in range(t - n + 1, t + 1):
            peak = max(peak, f.close[i])
            worst = min(worst, f.close[i] / peak - 1)
        return worst

    return g


def h_p_sma(n):
    return lambda f, t: f.close[t] / _sma(f.close, t, n) - 1


def h_sma_ratio(f, t):
    return _sma(f.close, t, PC.sma.fast) / _sma(f.close, t, PC.sma.slow) - 1


def h_rsi(f, t):
    n = PC.rsi
    if t - n < 0:
        return math.nan
    g = lo = 0.0
    for i in range(t - n + 1, t + 1):
        d = f.close[i] - f.close[i - 1]
        g += max(d, 0)
        lo += max(-d, 0)
    return 50.0 if g + lo == 0 else 100 * g / (g + lo)


def h_macd(f, t):
    mc = PC.macd
    s = mc.trunc_spans

    def macd(i):
        return _ema(f.close, i, mc.fast, s * mc.fast) - _ema(f.close, i, mc.slow, s * mc.slow)

    length = s * mc.signal
    if t - length + 1 < 0:
        return math.nan
    a = 2 / (mc.signal + 1)
    num = den = 0.0
    for j in range(length):
        w = (1 - a) ** j
        num += w * macd(t - j)
        den += w
    return (macd(t) - num / den) / f.close[t]


def h_bb(f, t):
    n, k = PC.bollinger.window, PC.bollinger.k
    if t - n + 1 < 0:
        return math.nan
    w = f.close[t - n + 1 : t + 1]
    m, sd = np.mean(w), np.std(w, ddof=0)
    return (f.close[t] - (m - k * sd)) / (2 * k * sd)


def h_atr(f, t):
    n = PC.atr
    if t - n < 0:
        return math.nan
    trs = []
    for i in range(t - n + 1, t + 1):
        pc = f.close[i - 1]
        trs.append(max(f.high[i] - f.low[i], abs(f.high[i] - pc), abs(f.low[i] - pc)))
    return float(np.mean(trs)) / f.close[t]


def h_volz(f, t):
    n = PC.volume_z
    if t - n < 0:
        return math.nan
    prev = np.log(f.volume[t - n : t])
    return (math.log(f.volume[t]) - prev.mean()) / prev.std(ddof=1)


def h_amihud(f, t):
    n = PC.amihud.window
    r = _rets(f, t, n)
    if r is None:
        return math.nan
    dv = f.close_raw[t - n + 1 : t + 1] * f.volume_raw[t - n + 1 : t + 1]
    return float(np.mean(np.abs(r) / dv)) * PC.amihud.scale


def h_beta(n):
    def g(f, t):
        x, y = _pairs(f, t, n, "bench")
        if t - n + 1 < 0 or len(x) < _need_pairs(n):
            return math.nan
        return float(np.polyfit(x, y, 1)[0])

    return g


def h_resid(n):
    def g(f, t):
        x, y = _pairs(f, t, n, "bench")
        if t - n + 1 < 0 or len(x) < _need_pairs(n):
            return math.nan
        b, a = np.polyfit(x, y, 1)
        res = y - (a + b * x)
        return math.sqrt(float(np.sum(res**2)) / (len(x) - 2)) * math.sqrt(252)

    return g


def h_corr(key):
    n = PC.corr.window

    def g(f, t):
        x, y = _pairs(f, t, n, key)
        if t - n + 1 < 0 or len(x) < _need_pairs(n):
            return math.nan
        return float(np.corrcoef(x, y)[0, 1])

    return g


def h_mrz(f, t):
    n = PC.mr_z
    if t - n + 1 < 0:
        return math.nan
    lp = np.log(f.close[t - n + 1 : t + 1])
    return (lp[-1] - lp.mean()) / lp.std(ddof=1)


def h_ou(f, t):
    n, cap = PC.ou.window, PC.ou.halflife_cap
    if t - n < 0:
        return math.nan
    lp = np.log(f.close)
    x = np.array([lp[i - 1] for i in range(t - n + 1, t + 1)])
    y = np.array([lp[i] - lp[i - 1] for i in range(t - n + 1, t + 1)])
    b = float(np.polyfit(x, y, 1)[0])
    if b >= 0:
        return cap
    if b <= -1:
        return math.nan
    return min(-math.log(2) / math.log(1 + b), cap)


def h_hurst(f, t):
    n, q = PC.hurst.window, PC.hurst.q
    r = _rets(f, t, n)
    if r is None:
        return math.nan
    sums = [r[j : j + q].sum() for j in range(n - q + 1)]
    return math.log(np.var(sums, ddof=1) / np.var(r, ddof=1)) / (2 * math.log(q))


def _earn_naive(f, t):
    """(days, surprise, approx) at bar t from first principles."""
    ready = pd.Timestamp(int(f.ready_ns[t]), tz=UTC)
    d = pd.Timestamp(f.dates[t]).date()
    e = f.earnings[f.earnings["released_at"] <= ready]
    e = e.sort_values("released_at").groupby("quarter_key").tail(1)  # newest per quarter
    sess = [pd.Timestamp(x).date() for x in f.sessions]
    best = None
    last = None
    for row in e.itertuples(index=False):
        r = row.report_date
        upcoming = r > d or (r == d and not row.pre_market)
        if upcoming:
            days = sum(1 for s in sess if d <= s < r)
            if days <= PC.earnings.max_sessions_ahead and (best is None or days < best[0]):
                best = (days, float(row.approx))
        elif np.isfinite(row.surprise) and (last is None or r > last[0]):
            last = (r, row.surprise)
    return (
        math.nan if best is None else best[0],
        math.nan if last is None else last[1],
        math.nan if best is None else best[1],
    )


HAND = {
    **{f"ret_{n}d": h_ret(n) for n in PC.returns},
    "mom_12_1": h_mom,
    **{f"rvol_{n}d": h_rvol(n) for n in PC.rvol},
    **{f"gk_vol_{n}d": h_gk(n) for n in PC.gk_vol},
    **{f"downside_dev_{n}d": h_down(n) for n in PC.downside_dev},
    **{f"max_dd_{n}d": h_dd(n) for n in PC.max_dd},
    f"p_sma{PC.sma.fast}": h_p_sma(PC.sma.fast),
    f"p_sma{PC.sma.slow}": h_p_sma(PC.sma.slow),
    f"sma{PC.sma.fast}_sma{PC.sma.slow}": h_sma_ratio,
    f"rsi_{PC.rsi}": h_rsi,
    "macd_hist_p": h_macd,
    f"bb_pctb_{PC.bollinger.window}": h_bb,
    f"atr_{PC.atr}_pct": h_atr,
    f"vol_z_{PC.volume_z}d": h_volz,
    f"amihud_{PC.amihud.window}d": h_amihud,
    **{f"beta_{n}d": h_beta(n) for n in PC.beta},
    **{f"resid_vol_{n}d": h_resid(n) for n in PC.beta},
    **{f"corr_{k}_{PC.corr.window}d": h_corr(k) for k in PC.corr.series},
    f"mr_z_{PC.mr_z}d": h_mrz,
    f"ou_halflife_{PC.ou.window}d": h_ou,
    f"hurst_vr_{PC.hurst.window}d": h_hurst,
    "earn_days_next": lambda f, t: _earn_naive(f, t)[0],
    "earn_surprise_last": lambda f, t: _earn_naive(f, t)[1],
    "earn_date_approx": lambda f, t: _earn_naive(f, t)[2],
}

CHECK_AT = [5, 13, 20, 62, 150, 185, 187, 199, 200, 230, 260, 300, N_BARS - 1]
SPEC_IDS = [s.name for s in SPECS]


def _same(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_feature_hand_calcs(spec):
    assert spec.name in HAND, f"{spec.name} has no hand calculation"
    got = np.asarray(spec.fn(FULL), dtype=float)
    assert got.shape == (FULL.n,)
    for t in CHECK_AT:
        want = HAND[spec.name](FULL, t)
        assert _same(float(got[t]), float(want)), (spec.name, t, got[t], want)
    # the window fills at some point: not all NaN at the end
    assert np.isfinite(got[-60:]).any(), spec.name


def test_every_price_feature_registered_has_a_hand_case():
    names = {s.name for s in price.price_specs()}
    assert names == {s.name for s in SPECS}
    assert names <= set(HAND)


def test_hand_fixture_exercises_the_edges():
    """Guards that the fixture hits the cases the hand references branch on."""
    surprise = np.asarray(registry.get("earn_surprise_last", 1).fn(FULL))
    days = np.asarray(registry.get("earn_days_next", 1).fn(FULL))
    assert np.isfinite(surprise).any() and np.isnan(surprise[:55]).all()
    assert (days == 0).any() and np.isnan(days).any()
    # the moved Q3 date: on day 181 the next report is day 186, not 190
    assert days[181] == 5
    # the split (day 200) is adjusted away: no -69% return
    assert abs(np.asarray(registry.get("ret_1d", 1).fn(FULL))[200]) < 0.1
    # some bench dates are missing, so pairing matters
    assert np.isnan(FULL.ctx["bench"]).sum() > 5


# ---------------------------------------------------------------- point in time
def _mutate_after(t: pd.Timestamp, seed: int = 99):
    rng = np.random.default_rng(seed)
    b = BARS.copy()
    late = b["released_at"] > t
    for c in ("open", "high", "low", "close"):
        b.loc[late, c] = b.loc[late, c] * rng.uniform(0.5, 1.5, late.sum())
    b.loc[late, "volume"] = b.loc[late, "volume"] * 3
    a = pd.concat(
        [
            ACTIONS,
            pd.DataFrame(
                [
                    {
                        **ACTIONS.iloc[0].to_dict(),
                        "external_id": "late_split",
                        "ex_date": BARS["date"].iloc[100],
                        "released_at": t + pd.Timedelta(minutes=1),
                        "old_rate": 1.0,
                        "new_rate": 3.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ctx = {}
    for k, s in CTX.items():
        f = s.frame.copy()
        lt = f["released_at"] > t
        f.loc[lt, "value"] = f.loc[lt, "value"] * 1.7
        # a revision of an old date, released after t
        rev = f.iloc[[10]].assign(
            value=f["value"].iloc[10] * 9, released_at=t + pd.Timedelta(1, "h")
        )
        ctx[k] = price.CtxSeries(pd.concat([f, rev], ignore_index=True), s.kind)
    e = EARN.copy()
    le = e["released_at"] > t
    e.loc[le, "surprise"] = 99.0
    e.loc[le, "report_date"] = BARS["date"].iloc[-1]
    extra = EARN.iloc[[0]].assign(
        quarter_key="XYZ:Q2", report_date=BARS["date"].iloc[5], released_at=t + pd.Timedelta(1, "s")
    )
    e = pd.concat([e, extra], ignore_index=True)
    return b, a, ctx, e


T_CHECKS = [BARS["released_at"].iloc[i] + pd.Timedelta(minutes=30) for i in (183, 199, 280)]


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
@pytest.mark.parametrize("ti", range(len(T_CHECKS)))
def test_no_lookahead_features(spec, ti):
    t = T_CHECKS[ti]
    clean = price.compute_asof(_frame(t), [spec])[spec.key]
    b, a, ctx, e = _mutate_after(t)
    dirty = price.compute_asof(_frame(t, b, a, ctx, e), [spec])[spec.key]
    assert _same(clean, dirty), (spec.name, clean, dirty)


def _ns(t: pd.Timestamp) -> int:
    return int(t.as_unit("ns").value)


def test_backfill_pass_matches_daily():
    vals, ready = price.compute_series(FULL, SPECS)
    for i in (150, 183, 199, 200, 201, 280, N_BARS - 2):
        # +30 min: the bar is out, the y10 row (+45 min) is not yet
        for lag in (30, 60):
            t = BARS["released_at"].iloc[i] + pd.Timedelta(minutes=lag)
            daily = price.compute_asof(_frame(t), SPECS)
            for s in SPECS:
                ok = np.flatnonzero(ready[s.key].to_numpy() <= _ns(t))
                want = float(vals[s.key].iloc[ok[-1]])
                assert _same(want, daily[s.key]), (s.name, i, lag, want, daily[s.key])


def test_ready_at_waits_for_context():
    """y10 posts 45 minutes after the bar: only features using it wait for it."""
    _, ready = price.compute_series(FULL, SPECS)
    bar_rel = _ns(BARS["released_at"].iloc[100])
    assert ready[("ret_1d", 1)].iloc[100] == bar_rel
    assert ready[(f"corr_y10_{PC.corr.window}d", 1)].iloc[100] == bar_rel + 45 * 60 * 10**9
    # a date the bench lacks is ready once the bench's next row is out
    miss = np.flatnonzero(np.isnan(FULL.ctx["bench"][1:-1])) + 1
    i = int(miss[0])
    nxt = BARS["released_at"].iloc[i + 1 :]
    later = [d for d in CTX["bench"].frame["date"] if d > BARS["date"].iloc[i]][0]
    want = _ns(nxt[BARS["date"].iloc[i + 1 :] == later].iloc[0])
    assert ready[("beta_63d", 1)].iloc[i] == want


def test_scale_free_under_a_later_split():
    """Whether the split is known by T or not, values at earlier rows are the same."""
    late = ACTIONS.copy()
    late.loc[0, "released_at"] = BARS["released_at"].iloc[160]  # still before its ex date
    t2 = BARS["released_at"].iloc[210] + pd.Timedelta(minutes=60)
    a, _ = price.compute_series(_frame(t2, actions=ACTIONS), SPECS)
    no_split, _ = price.compute_series(_frame(t2, actions=ACTIONS.iloc[[1]]), SPECS)
    b, _ = price.compute_series(_frame(t2, actions=late), SPECS)
    for s in SPECS:
        for i in (150, 190):
            assert _same(float(a[s.key].iloc[i]), float(b[s.key].iloc[i])), s.name
            assert _same(float(a[s.key].iloc[i]), float(no_split[s.key].iloc[i])), s.name


def test_non_equity_without_earnings_is_nan():
    f = price.prepare(
        BARS.assign(asset_class="crypto"), ASOF_END, PC, CFG.adjust, asset_class="crypto",
        ctx=CTX, sessions=SESSIONS,
    )  # fmt: skip
    vals = price.compute_asof(f, SPECS)
    assert math.isnan(vals[("earn_days_next", 1)])
    # crypto annualizes with 365
    r = np.array([_r(f.close, i) for i in range(f.n - 21, f.n)])
    assert _same(vals[("rvol_21d", 1)], float(np.std(r, ddof=1)) * math.sqrt(365))


def test_mutation_is_visible_without_the_cut():
    """The no-lookahead mutation is strong enough to change values it is allowed to see."""
    t = T_CHECKS[0]
    b, a, ctx, e = _mutate_after(t)
    clean = price.compute_asof(_frame(ASOF_END), SPECS)
    dirty = price.compute_asof(_frame(ASOF_END, b, a, ctx, e), SPECS)
    changed = {k[0] for k in clean if not _same(clean[k], dirty[k])}
    # every family except the earnings flag of a quarter the mutation leaves alone
    assert len(changed) >= len(SPECS) - 3, sorted({s.name for s in SPECS} - changed)
