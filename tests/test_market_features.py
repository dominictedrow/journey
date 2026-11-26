"""Phase 3 step 4a: market vector (component doc 03 sections 5.3 and 11).

Synthetic macro vintages (daily, weekly, monthly, quarterly series; revisions released later; a
missing month), session prices, weekend-trading coins and a breadth table. Every registered market
feature is checked against a naive re-implementation that rebuilds "what was known at T" by
looping over the raw rows, and against a mutation of every row released after T.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qt.common import config
from qt.common.calendar import TradingCalendar
from qt.features import market
from qt.features.inputs import MacroState

CFG = config.load().config.features.market
SPECS = market.register(CFG)
SESSIONS = np.array(TradingCalendar().cal.sessions.date, dtype="datetime64[D]")
DAYS = [pd.Timestamp(x).date() for x in SESSIONS
        if np.datetime64("2021-01-04") <= x <= np.datetime64("2023-06-30")]  # fmt: skip
RNG = np.random.default_rng(11)


def rel(d: date, hh: int = 22) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, hh, tzinfo=UTC))


def _macro():
    rows = []

    def add(sid, period, value, vintage, released):
        rows.append({"series_id": sid, "period": period, "value": value, "vintage": vintage,
                     "released_at": released})  # fmt: skip

    daily = {"DGS10": 1.5, "DGS2": 0.5, "DGS3MO": 0.1, "DFII10": -0.5, "DFF": 0.1,
             "T10YIE": 2.2, "T5YIFR": 2.1, "THREEFYTP10": 0.3, "BAA10Y": 2.0, "DEXUSAL": 0.75,
             "DEXJPUS": 110.0, "DEXUSEU": 1.15, "RRPONTSYD": 100.0}  # fmt: skip
    for sid, lvl in daily.items():
        x = lvl
        for i, d in enumerate(DAYS):
            x = x * math.exp(RNG.normal(0, 0.004)) if lvl > 5 else x + RNG.normal(0, 0.03)
            if sid == "DGS10" and i % 37 == 5:
                continue  # holidays / missing values
            add(sid, d, x, d, rel(d))
            if sid in ("DGS10", "BAA10Y") and i % 50 == 7 and i + 30 < len(DAYS):
                # a revision of this period, published 30 sessions later
                add(sid, d, x + 0.25, DAYS[i + 30], rel(DAYS[i + 30]))
    weekly = {"NFCI": -0.4, "ICSA": 220000.0, "WALCL": 7.5e6, "WTREGEN": 6e5, "ECBASSETSW": 7e6}
    for sid, lvl in weekly.items():
        x = lvl
        d = date(2020, 1, 3)
        while d <= DAYS[-1]:
            x = x * math.exp(RNG.normal(0, 0.01)) if abs(lvl) > 5 else x + RNG.normal(0, 0.02)
            add(sid, d, x, d + timedelta(days=5), rel(d + timedelta(days=5), 14))
            d += timedelta(days=7)
    monthly = {"CPILFESL": 270.0, "PCEPILFE": 115.0, "UNRATE": 4.0, "JPNASSETS": 7e6,
               "IR3TIB01JPM156N": -0.05}  # fmt: skip
    for sid, lvl in monthly.items():
        x = lvl
        for k in range(48):
            p = date(2019 + k // 12, k % 12 + 1, 1)
            x = x * (1 + RNG.normal(0.002, 0.002)) if lvl > 50 else x + RNG.normal(0, 0.1)
            if sid == "UNRATE" and p == date(2022, 10, 1):
                continue  # a month never published
            r = (pd.Timestamp(p) + pd.DateOffset(months=1, days=12)).date()
            add(sid, p, x, r, rel(r, 13))
            if sid == "CPILFESL" and k % 9 == 0:  # a later revision
                r2 = r + timedelta(days=95)
                add(sid, p, x * 1.003, r2, rel(r2, 13))
    x = 19000.0
    for k in range(16):
        p = date(2019 + k // 4, 3 * (k % 4) + 1, 1)
        x *= 1 + RNG.normal(0.005, 0.005)
        r = (pd.Timestamp(p) + pd.DateOffset(months=4)).date()
        add("GDPC1", p, x, r, rel(r, 13))
    df = pd.DataFrame(rows)
    df["released_at"] = pd.to_datetime(df["released_at"], utc=True)
    return df


MACRO = _macro()


def _prices():
    out = {}
    tks = set(CFG.tickers.values()) | {t for pair in CFG.ratios for t in pair}
    for tk in sorted(tks):
        crypto = "/" in tk
        days = ([DAYS[0] + timedelta(days=i) for i in range((DAYS[-1] - DAYS[0]).days + 1)]
                if crypto else DAYS)  # fmt: skip
        r = RNG.normal(0, 0.03 if crypto else 0.01, len(days))
        lvl = 20 * np.exp(np.cumsum(r))
        keep = RNG.random(len(days)) > 0.03
        hh = 1 if crypto else 21
        out[tk] = pd.DataFrame({
            "date": [d for d, k in zip(days, keep, strict=True) if k],
            "value": lvl[keep],
            "released_at": [rel(d, hh) + (timedelta(days=1) if crypto else timedelta(0))
                            for d, k in zip(days, keep, strict=True) if k],
        })  # fmt: skip
    return out


PRICES = _prices()


def _breadth():
    ids = list(range(30))
    c = pd.DataFrame(20 * np.exp(np.cumsum(RNG.normal(0, 0.015, (len(DAYS), 30)), axis=0)),
                     index=pd.Index(np.array(DAYS, dtype="datetime64[D]")),
                     columns=ids)  # fmt: skip
    c.iloc[:120, :5] = np.nan  # late listings
    ready = pd.DataFrame(np.array([[rel(d, 21).value] * 30 for d in DAYS]),
                         index=c.index, columns=ids)  # fmt: skip
    mem = pd.DataFrame(True, index=c.index, columns=ids)
    mem.iloc[300:, 25:] = False  # some leave the index
    return c, ready, mem


BC, BR, BM = _breadth()
BREADTH = market.breadth_table(BC, BR, BM, sma=200, hl=252)


def state(t: pd.Timestamp, macro=MACRO, prices=PRICES, breadth=BREADTH) -> market.MarketState:
    ms = MacroState(macro)
    ms.advance(t)
    return market.MarketState(t, ms, prices, SESSIONS, CFG, breadth=breadth)


# ---------------------------------------------------------------- naive "known at T"
class Known:
    def __init__(self, t: pd.Timestamp, macro=MACRO, prices=PRICES, breadth=BREADTH):
        self.t = t
        self.today = t.tz_convert("America/New_York").date()
        self.sess = [pd.Timestamp(x).date() for x in SESSIONS
                     if pd.Timestamp(x).date() <= self.today]  # fmt: skip
        self.m: dict[str, list] = {}
        best: dict = {}
        for r in macro.itertuples(index=False):
            if r.released_at > t or (isinstance(r.value, float) and math.isnan(r.value)):
                continue
            k = (r.series_id, r.period)
            if k not in best or r.vintage >= best[k][0]:
                best[k] = (r.vintage, r.value)
        for (sid, p), (_v, val) in best.items():
            self.m.setdefault(sid, []).append((p, float(val)))
        for v in self.m.values():
            v.sort()
        self.p = {}
        for tk, f in prices.items():
            ok = f[f["released_at"] <= t]
            self.p[tk] = sorted(zip(ok["date"], ok["value"].astype(float), strict=True))
        self.b = breadth[breadth["ready_ns"] <= t.value]

    def ser(self, src):
        return self.p.get(src[3:], []) if src.startswith("px:") else self.m.get(src, [])

    def on(self, src, d):
        v = math.nan
        for dd, x in self.ser(src):
            if dd <= d:
                v = x
            else:
                break
        return v

    def smp(self, src, n):
        if len(self.sess) < n:
            return [math.nan] * n
        return [self.on(src, d) for d in self.sess[-n:]]

    def last(self, src):
        s = self.ser(src)
        return s[-1][1] if s else math.nan


def _z(x):
    x = np.array(x, dtype=float)
    if not np.isfinite(x).all():
        return math.nan
    return (x[-1] - x.mean()) / x.std(ddof=1)


def _lr(x):
    return math.log(x[-1] / x[0]) if all(map(math.isfinite, (x[0], x[-1]))) else math.nan


def _d(x):
    return x[-1] - x[0]


def _rets(k, src, n):
    x = k.smp(src, n + 1)
    return [math.log(x[i + 1] / x[i]) for i in range(n)]


def _corr(a, b):
    a, b = np.array(a), np.array(b)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _month_back(s, months):
    last = s[-1][0]
    want = (pd.Timestamp(last) - pd.DateOffset(months=months)).date()
    for p, v in s:
        if p == want:
            return v
    return math.nan


def h_infl(sid):
    def g(k):
        s = k.ser(sid)
        c = CFG.infl_months
        now = s[-1][1]
        return (
            (now / _month_back(s, c.short)) ** (12 / c.short)
            - 1
            - (now / _month_back(s, c.long) - 1)
        )

    return g


def h_sahm(k):
    s = dict(k.ser("UNRATE"))
    c = CFG.sahm
    first, last = min(s), max(s)
    months = []
    d = first
    while d <= last:
        months.append(d)
        d = (pd.Timestamp(d) + pd.DateOffset(months=1)).date()
    avg = []
    for i in range(len(months)):
        win = [s[m] for m in months[max(0, i - c.avg_months + 1) : i + 1] if m in s]
        avg.append(sum(win) / len(win) if len(win) >= c.avg_months - 1 else math.nan)
    prior = [a for a in avg[-1 - c.lookback_months : -1] if math.isfinite(a)]
    return avg[-1] - min(prior)


def h_gdp(k):
    s = k.ser("GDPC1")
    return (s[-1][1] / s[-2][1]) ** 4 - 1


def h_claims(k):
    s = k.ser("ICSA")
    c = CFG.claims
    now = sum(v for _, v in s[-c.avg_weeks :]) / c.avg_weeks
    cut = s[-1][0] - timedelta(days=c.yoy_days)
    past = [x for x in s if x[0] <= cut]
    return now / (sum(v for _, v in past[-c.avg_weeks :]) / c.avg_weeks) - 1


def h_days_chg(sid, days):
    def g(k):
        s = k.ser(sid)
        return s[-1][1] - k.on(sid, s[-1][0] - timedelta(days=days))

    return g


def h_gold(k):
    c = CFG.gold_resid
    g = _rets(k, "px:" + CFG.tickers["gold"], c.fit_window)
    r = k.smp("DFII10", c.fit_window + 1)
    dr = [r[i + 1] - r[i] for i in range(c.fit_window)]
    b, a = np.polyfit(dr, g, 1)
    e = np.array(g) - (a + b * np.array(dr))
    return e[-c.horizon :].sum() / (e.std(ddof=2) * math.sqrt(c.horizon))


def gli_on(k, d):
    tot = 0.0
    for t in CFG.gli.terms:
        v = k.on(t.series, d)
        if t.fx:
            fx = k.on(t.fx, d)
            v = v * fx if t.fx_op == "mul" else v / fx
        tot += t.factor * v
    return tot


def h_breadth(col):
    return lambda k: float(k.b[col].iloc[-1])


def h_crypto(n, z):
    def g(k):
        e = dict(k.p[CFG.tickers["eth"]])
        b = dict(k.p[CFG.tickers["btc"]])
        both = sorted(set(e) & set(b))
        r = [math.log(e[d] / b[d]) for d in both]
        return _z(r[-n:]) if z else r[-1] - r[-1 - n]

    return g


def _hand() -> dict:
    n_c, n_r, n_z, n_k, n_q = (CFG.chg_window, CFG.ret_window, CFG.z_window, CFG.corr_window,
                               CFG.ratio_window)  # fmt: skip
    tk = {k: "px:" + v for k, v in CFG.tickers.items()}

    def spread(a, b, n=None):
        if n is None:
            return lambda k: k.last(a) - k.last(b)
        return lambda k: _d(np.array(k.smp(a, n + 1)) - np.array(k.smp(b, n + 1)))

    h = {
        "curve_10y2y": spread("DGS10", "DGS2"),
        f"curve_10y2y_chg{n_c}": spread("DGS10", "DGS2", n_c),
        "curve_10y3m": spread("DGS10", "DGS3MO"),
        f"curve_10y3m_chg{n_c}": spread("DGS10", "DGS3MO", n_c),
        "real_yield_10y": lambda k: k.last("DFII10"),
        f"real_yield_10y_chg{n_c}": lambda k: _d(k.smp("DFII10", n_c + 1)),
        "policy_path": spread("DGS2", "DFF"),
        "breakeven_10y": lambda k: k.last("T10YIE"),
        f"breakeven_10y_chg{n_c}": lambda k: _d(k.smp("T10YIE", n_c + 1)),
        "fwd_5y5y": lambda k: k.last("T5YIFR"),
        "term_premium_10y": lambda k: k.last("THREEFYTP10"),
        f"term_premium_10y_chg{n_c}": lambda k: _d(k.smp("THREEFYTP10", n_c + 1)),
        "infl_mom_core_cpi": h_infl("CPILFESL"),
        "infl_mom_core_pce": h_infl("PCEPILFE"),
        "sahm": h_sahm,
        "gdp_qoq_ann": h_gdp,
        "nfci": lambda k: k.last("NFCI"),
        "nfci_chg4w": h_days_chg("NFCI", CFG.nfci_chg_days),
        "claims_4w_yoy": h_claims,
        "credit_baa10y": lambda k: k.last("BAA10Y"),
        f"credit_baa10y_z{n_z}": lambda k: _z(k.smp("BAA10Y", n_z)),
        f"credit_baa10y_chg{n_c}": lambda k: _d(k.smp("BAA10Y", n_c + 1)),
        "vix_vix3m": lambda k: k.last(tk["vix"]) / k.last(tk["vix3m"]),
        f"vix_z{n_z}": lambda k: _z(k.smp(tk["vix"], n_z)),
        "move": lambda k: k.last(tk["move"]),
        f"move_z{n_z}": lambda k: _z(k.smp(tk["move"], n_z)),
        "move_vix": lambda k: k.last(tk["move"]) / k.last(tk["vix"]),
        f"skew_z{n_z}": lambda k: _z(k.smp(tk["skew"], n_z)),
        f"vvix_z{n_z}": lambda k: _z(k.smp(tk["vvix"], n_z)),
        "vvix_vix": lambda k: k.last(tk["vvix"]) / k.last(tk["vix"]),
        f"dxy_ret{n_r}": lambda k: _lr(k.smp(tk["dxy"], n_r + 1)),
        f"dxy_z{n_z}": lambda k: _z(np.log(k.smp(tk["dxy"], n_z))),
        f"spy_tlt_corr{n_k}": lambda k: _corr(_rets(k, tk["spy"], n_k), _rets(k, tk["tlt"], n_k)),
        "gold_resid_z": h_gold,
        f"vnq_spy_ret{n_q}": lambda k: _lr(k.smp(tk["vnq"], n_q + 1))
        - _lr(k.smp(tk["spy"], n_q + 1)),
        f"wti_ret{n_r}": lambda k: _lr(k.smp(tk["wti"], n_r + 1)),
        f"wti_z{n_z}": lambda k: _z(np.log(k.smp(tk["wti"], n_z))),
        f"copper_gold_chg{n_c}": lambda k: _d(np.log(np.array(k.smp(tk["copper"], n_c + 1))
                                                     / np.array(k.smp(tk["gold"], n_c + 1)))),
        f"copper_gold_z{n_z}": lambda k: _z(np.log(np.array(k.smp(tk["copper"], n_z))
                                                   / np.array(k.smp(tk["gold"], n_z)))),
        f"aud_ret{n_r}": lambda k: _lr(k.smp("DEXUSAL", n_r + 1)),
        "usdjpy_carry": spread("DGS3MO", "IR3TIB01JPM156N"),
        f"usdjpy_ret{n_r}": lambda k: _lr(k.smp("DEXJPUS", n_r + 1)),
        "breadth_above_sma200": h_breadth("above_sma"),
        "breadth_nh_nl": h_breadth("nh_nl"),
        f"breadth_adl_chg{n_c}": lambda k: float(k.b["adv_dec"].iloc[-n_c:].sum()),
        "gli": lambda k: gli_on(k, k.sess[-1]),
        "gli_chg13w": lambda k: gli_on(k, k.sess[-1])
        / gli_on(k, k.sess[-1] - timedelta(days=CFG.gli.window_days)) - 1,
        f"gli_z{n_z}": lambda k: _z([gli_on(k, d) for d in k.sess[-n_z:]]),
        f"eth_btc_chg{n_c}": h_crypto(n_c, False),
        f"eth_btc_z{n_z}": h_crypto(n_z, True),
    }  # fmt: skip
    for a, b in CFG.ratios:
        h[f"ratio_{a.lower()}_{b.lower()}_{n_q}d"] = lambda k, a=a, b=b: (
            _lr(k.smp(f"px:{a}", n_q + 1)) - _lr(k.smp(f"px:{b}", n_q + 1))
        )
    return h


HAND = _hand()
T_CHECKS = [rel(DAYS[i], 23) for i in (300, 420, 555, len(DAYS) - 1)]
# a weekend decision time too (crypto runs 7 days a week)
T_CHECKS.append(rel(DAYS[480] + timedelta(days=(5 - DAYS[480].weekday()) % 7 or 7), 23))
STATES = {t: state(t) for t in T_CHECKS}
KNOWN = {t: Known(t) for t in T_CHECKS}


def _same(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_market_hand_calcs(spec):
    assert spec.name in HAND, f"{spec.name} has no hand calculation"
    for t in T_CHECKS:
        got = market.compute(STATES[t], [spec])[spec.key]
        want = float(HAND[spec.name](KNOWN[t]))
        assert _same(got, want), (spec.name, str(t), got, want)
        assert math.isfinite(got), (spec.name, str(t))


def _mutate_after(t):
    m = MACRO.copy()
    late = m["released_at"] > t
    m.loc[late, "value"] = m.loc[late, "value"] * 1.5 + 1
    # revisions of old periods, released later
    extra = m[~late].groupby("series_id").head(3).assign(
        value=lambda x: x["value"] * 7, vintage=t.date() + timedelta(days=3),
        released_at=t + pd.Timedelta(minutes=1))  # fmt: skip
    m = pd.concat([m, extra], ignore_index=True)
    p = {}
    for tk, f in PRICES.items():
        f = f.copy()
        lt = f["released_at"] > t
        # per-row noise (a constant factor would leave returns inside the window unchanged)
        f.loc[lt, "value"] = f.loc[lt, "value"] * np.exp(RNG.normal(0, 0.05, lt.sum()))
        p[tk] = f
    b = BREADTH.copy()
    b.loc[b["ready_ns"] > t.value, ["above_sma", "nh_nl", "adv_dec"]] = 0.99
    return m, p, b


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_no_lookahead_market(spec):
    for t in T_CHECKS[:3]:
        clean = market.compute(STATES[t], [spec])[spec.key]
        dirty = market.compute(state(t, *_mutate_after(t)), [spec])[spec.key]
        assert _same(clean, dirty), (spec.name, str(t), clean, dirty)


def test_mutation_is_visible_without_the_cut():
    t = T_CHECKS[0]
    end = T_CHECKS[-2]
    clean = market.compute(state(end), SPECS)
    dirty = market.compute(state(end, *_mutate_after(t)), SPECS)
    changed = {k[0] for k in clean if not _same(clean[k], dirty[k])}
    assert len(changed) >= len({s.name for s in SPECS}) - 2, sorted({s.name for s in SPECS} - changed)


def test_breadth_table_counts():
    i = 400
    d = BC.index[i]
    m = BM.loc[d] & BC.loc[d].notna()
    win = BC.iloc[i - 199 : i + 1]
    above = (BC.loc[d] > win.mean())[m & win.notna().all()].mean()
    assert math.isclose(BREADTH.loc[d, "above_sma"], above)
    chg = BC.iloc[i] - BC.iloc[i - 1]
    ok = m & chg.notna()
    assert math.isclose(BREADTH.loc[d, "adv_dec"],
                        ((chg[ok] > 0).sum() - (chg[ok] < 0).sum()) / ok.sum())  # fmt: skip


def test_missing_month_keeps_sahm_defined():
    t = rel(date(2023, 1, 20), 23)  # after the unpublished October 2022
    v = market.compute(state(t), [next(s for s in SPECS if s.name == "sahm")])
    assert math.isfinite(next(iter(v.values())))
