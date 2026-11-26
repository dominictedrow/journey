"""Phase 3 step 5a: regime models (component doc 03 sections 6 and 11).

Synthetic, point-in-time inputs: a regime-switching SPY path with matching VIX / VIX3M, BAA10Y
daily vintages (released the next morning, some revised later), and monthly / quarterly macro.
Checked against naive re-implementations and against mutations of rows released after T.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qt.common import config
from qt.common.calendar import TradingCalendar
from qt.features import market, registry
from qt.features import regime as regime_mod
from qt.features.inputs import ET, MacroState
from qt.features.market import MarketState
from qt.features.regime import hmm, quadrant, stress

CFG = config.load().config.features
MCFG = CFG.market
RCFG = CFG.regime
# small enough for tests: a 3-regime path over ~8 years gives ~400 weekly samples
HCFG = RCFG.hmm.model_copy(update={"n_seeds": 4, "min_weeks": 150, "n_iter": 100,
                                   "first_obs": date(2015, 1, 1)})  # fmt: skip
SESSIONS = np.array(TradingCalendar().cal.sessions.date, dtype="datetime64[D]")
DAYS = [pd.Timestamp(x).date() for x in SESSIONS
        if np.datetime64("2014-10-01") <= x <= np.datetime64("2022-12-30")]  # fmt: skip
TK = MCFG.tickers


def rel(d: date, hh: int, mm: int = 0) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)).tz_convert(UTC)


def at18(d: date) -> pd.Timestamp:
    return rel(d, 18)


def _inputs(seed: int = 3):
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    # regimes: 0 calm, 1 normal, 2 stress, sticky
    trans = np.array([[0.995, 0.004, 0.001], [0.006, 0.99, 0.004], [0.002, 0.018, 0.98]])
    st = np.zeros(n, dtype=int)
    for i in range(1, n):
        st[i] = rng.choice(3, p=trans[st[i - 1]])
    st[300:360] = 2  # make sure stress appears
    vol = np.array([0.006, 0.011, 0.028])[st]
    drift = np.array([0.0006, 0.0002, -0.002])[st]
    spy = 200 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    vix = np.array([12.0, 18.0, 35.0])[st] * np.exp(0.05 * rng.standard_normal(n))
    vix3m = np.array([14.0, 19.0, 30.0])[st] * np.exp(0.03 * rng.standard_normal(n))
    baa = 2.0 + np.cumsum(np.array([-0.002, 0.0, 0.02])[st] + 0.01 * rng.standard_normal(n))

    def px(v, hh, mm):
        return pd.DataFrame({"date": DAYS, "value": v,
                             "released_at": [rel(d, hh, mm) for d in DAYS]})  # fmt: skip

    prices = {TK["spy"]: px(spy, 16, 30), TK["vix"]: px(vix, 16, 30),
              TK["vix3m"]: px(vix3m, 16, 30)}  # fmt: skip
    rows = []
    for i, d in enumerate(DAYS):
        nxt = DAYS[i + 1] if i + 1 < n else d + timedelta(days=1)
        if i % 41 == 7:
            continue  # a missing day
        rows.append(("BAA10Y", d, baa[i], nxt, rel(nxt, 8)))
        if i % 97 == 11 and i + 15 < n:  # revised 15 sessions later
            rows.append(("BAA10Y", d, baa[i] + 0.3, DAYS[i + 15], rel(DAYS[i + 15], 8)))
        rows.append(("DGS10", d, 2.0 + 0.002 * i % 1.5, nxt, rel(nxt, 8)))
        rows.append(("DGS3MO", d, 1.0 + rng.normal(0, 0.05), nxt, rel(nxt, 8)))
    macro = pd.DataFrame(rows, columns=["series_id", "period", "value", "vintage", "released_at"])
    macro["released_at"] = pd.to_datetime(macro["released_at"], utc=True)
    return prices, macro, st


PRICES, MACRO, STATES = _inputs()


def state(t: pd.Timestamp, macro=MACRO, prices=PRICES, regime=None) -> MarketState:
    ms = MacroState(macro)
    ms.advance(t)
    return MarketState(asof=t, macro=ms, prices=prices, sessions=SESSIONS, cfg=MCFG,
                       regime=regime)  # fmt: skip


# ---------------------------------------------------------------- quadrant
def _quad_macro(unrate: list[float], gdp: list[float], cpi: list[float]) -> pd.DataFrame:
    """Monthly UNRATE / CPILFESL from 2018-01 and quarterly GDPC1 from 2018Q1, released early."""
    rows = []
    for k, v in enumerate(unrate):
        p = date(2018 + k // 12, k % 12 + 1, 1)
        r = (pd.Timestamp(p) + pd.DateOffset(months=1, days=4)).date()
        rows.append(("UNRATE", p, v, r, rel(r, 8, 30)))
    for k, v in enumerate(cpi):
        p = date(2018 + k // 12, k % 12 + 1, 1)
        r = (pd.Timestamp(p) + pd.DateOffset(months=1, days=12)).date()
        rows.append(("CPILFESL", p, v, r, rel(r, 8, 30)))
    for k, v in enumerate(gdp):
        p = date(2018 + k // 4, 3 * (k % 4) + 1, 1)
        r = (pd.Timestamp(p) + pd.DateOffset(months=4)).date()
        rows.append(("GDPC1", p, v, r, rel(r, 8, 30)))
    df = pd.DataFrame(rows, columns=["series_id", "period", "value", "vintage", "released_at"])
    df["released_at"] = pd.to_datetime(df["released_at"], utc=True)
    return df


def _qstate(unrate, gdp, cpi, t=None):
    t = at18(date(2021, 12, 31)) if t is None else t
    return state(t, macro=_quad_macro(unrate, gdp, cpi), prices={})


FLAT_UNRATE = [4.0] * 47  # through Nov 2021 (released early Dec)
RISING_GDP = [100 * 1.005**k for k in range(13)]  # through 2021Q1 (released Aug 2021)
for k in (12,):
    RISING_GDP[k] = RISING_GDP[k - 1] * 1.02  # last quarter's growth above the 4-quarter mean
FALLING_GDP = RISING_GDP[:12] + [RISING_GDP[11] * 0.999]


def _cpi(accel: bool) -> list[float]:
    x, out = 250.0, []
    for k in range(47):
        g = 0.004 if (accel and k >= 44) else (0.001 if k >= 44 else 0.002)
        x *= 1 + g
        out.append(x)
    return out


def test_sahm_flat_within_tolerance_passes_growth_leg():
    # Sahm (= 0 for flat unemployment) then a bump that keeps Sahm within avg + 0.05
    u = FLAT_UNRATE[:-3] + [4.0, 4.0, 4.1]  # 3-mo avg 4.033 -> Sahm 0.033 vs its avg 0.011
    m = _qstate(u, RISING_GDP, _cpi(True))
    s = market.sahm_values(m).to_numpy()
    assert s[-1] == pytest.approx(0.1 / 3)
    assert s[-1] <= np.mean(s[-3:]) + 0.05
    assert quadrant.sahm_not_rising(m, RCFG.quadrant) is True


def test_sahm_above_avg_plus_tolerance_fails_growth_leg():
    u = FLAT_UNRATE[:-3] + [4.0, 4.1, 4.5]  # Sahm jumps to 0.2
    m = _qstate(u, RISING_GDP, _cpi(True))
    s = market.sahm_values(m).to_numpy()
    assert s[-1] > np.mean(s[-3:]) + 0.05
    assert quadrant.sahm_not_rising(m, RCFG.quadrant) is False
    assert quadrant.growth_up(m, RCFG.quadrant) is False


@pytest.mark.parametrize(
    ("gdp", "accel", "want"),
    [
        (RISING_GDP, False, quadrant.GOLDILOCKS),
        (RISING_GDP, True, quadrant.REFLATION),
        (FALLING_GDP, True, quadrant.STAGFLATION),
        (FALLING_GDP, False, quadrant.DEFLATION),
    ],
)
def test_each_quadrant_reached(gdp, accel, want):
    m = _qstate(FLAT_UNRATE, gdp, _cpi(accel))
    assert quadrant.quadrant(m, RCFG.quadrant) == want
    specs = regime_mod.register(MCFG, RCFG)
    hot = {s.name: s.fn(m) for s in specs if s.group == regime_mod.G_QUADRANT}
    assert hot == {f"quad_{q}": float(q == want) for q in quadrant.QUADRANTS}


def test_quadrant_gdp_rule_hand_calc():
    m = _qstate(FLAT_UNRATE, RISING_GDP, _cpi(False))
    g = RISING_GDP
    qoq = [(g[k] / g[k - 1]) ** 4 - 1 for k in range(1, len(g))]
    assert market.gdp_qoq_values(m).to_numpy()[1:] == pytest.approx(qoq)
    assert quadrant.gdp_above_avg(m, RCFG.quadrant) == (qoq[-1] > np.mean(qoq[-4:]))


def test_quadrant_unknown_without_inputs():
    m = _qstate(FLAT_UNRATE[:5], RISING_GDP[:2], _cpi(False)[:5])
    assert quadrant.quadrant(m, RCFG.quadrant) is None
    assert math.isnan(quadrant.one_hot(m, RCFG.quadrant, quadrant.GOLDILOCKS))


def test_quadrant_uses_vintage_known_at_t():
    # GDP's last quarter (2021Q1) is released 2021-05-01; before that 12 quarters are known
    t = at18(date(2021, 4, 20))
    before = _qstate(FLAT_UNRATE, RISING_GDP, _cpi(False), t=t)
    assert len(market.gdp_qoq_values(before)) == 12
    df = _quad_macro(FLAT_UNRATE, RISING_GDP, _cpi(False))
    assert (df["released_at"] > t).sum() > 10
    df.loc[df["released_at"] > t, "value"] = 1e9
    after = state(t, macro=df, prices={})
    assert quadrant.quadrant(after, RCFG.quadrant) == quadrant.quadrant(before, RCFG.quadrant)
    assert quadrant.quadrant(before, RCFG.quadrant) is not None


# ---------------------------------------------------------------- stress
def _z(x):
    x = np.asarray(x, dtype=float)
    return (x[-1] - x.mean()) / x.std(ddof=1)


def _naive_known(macro: pd.DataFrame, sid: str, t: pd.Timestamp) -> dict:
    out = {}
    for r in macro[(macro.series_id == sid) & (macro.released_at <= t)].itertuples():
        if r.period not in out or r.vintage >= out[r.period][0]:
            out[r.period] = (r.vintage, r.value)
    return {k: v[1] for k, v in out.items()}


def _naive_sampled(known: dict, sess: list[date]) -> list[float]:
    keys = sorted(known)
    out = []
    for d in sess:
        prior = [k for k in keys if k <= d]
        out.append(known[prior[-1]] if prior else math.nan)
    return out


@pytest.mark.parametrize("d", [DAYS[400], DAYS[1200], DAYS[-1]])
def test_stress_score_hand_calc(d):
    t = at18(d)
    m = state(t)
    n = RCFG.stress.z_window
    sess = [pd.Timestamp(x).date() for x in SESSIONS if x <= np.datetime64(d)][-n:]
    px = {k: dict(zip(PRICES[TK[k]]["date"], PRICES[TK[k]]["value"], strict=True))
          for k in ("vix", "vix3m")}  # fmt: skip
    vix = _naive_sampled(px["vix"], sess)
    baa = _naive_sampled(_naive_known(MACRO, "BAA10Y", t), sess)
    slope = np.subtract(_naive_sampled(_naive_known(MACRO, "DGS10", t), sess),
                        _naive_sampled(_naive_known(MACRO, "DGS3MO", t), sess))  # fmt: skip
    want = np.mean([_z(vix), _z(baa), _z(-slope), float(px["vix"][d] / px["vix3m"][d] > 1)])
    assert stress.stress_score(m, MCFG, RCFG.stress) == pytest.approx(want, rel=1e-9)


def test_stress_nan_before_window():
    assert math.isnan(stress.stress_score(state(at18(DAYS[100])), MCFG, RCFG.stress))


# ---------------------------------------------------------------- HMM observations
def test_week_end_sessions_handles_holidays():
    ends = hmm.week_end_sessions(SESSIONS)
    # Good Friday 2022-04-15: the week ends Thursday; Thanksgiving week ends Friday 2022-11-25
    assert np.datetime64("2022-04-14") in ends and np.datetime64("2022-04-15") not in ends
    assert np.datetime64("2022-11-25") in ends
    wk = pd.Series(pd.to_datetime(ends)).dt.isocalendar()
    assert not wk.duplicated(["year", "week"]).any()


def _naive_weekly(t: pd.Timestamp) -> pd.DataFrame:
    today = t.tz_convert(ET).date()
    sess = [pd.Timestamp(x).date() for x in SESSIONS if x <= np.datetime64(today)]
    known_px = {k: {r.date: r.value for r in PRICES[TK[k]].itertuples() if r.released_at <= t}
                for k in ("spy", "vix", "vix3m")}  # fmt: skip
    baa = _naive_known(MACRO, "BAA10Y", t)
    rows = {}
    ends = [pd.Timestamp(x).date() for x in hmm.week_end_sessions(SESSIONS)]
    for e in ends:
        if e < HCFG.first_obs or e > today or at18(e) > t:
            continue
        i = sess.index(e)
        win = sess[i - 21 : i + 1]
        spy = np.log(_naive_sampled(known_px["spy"], win))
        b = _naive_sampled(baa, win)
        vix = _naive_sampled(known_px["vix"], [e])[0]
        v3 = _naive_sampled(known_px["vix3m"], [e])[0]
        rows[e] = [spy[-1] - spy[0], np.std(np.diff(spy), ddof=1) * math.sqrt(252),
                   b[-1] - b[0], vix / v3]  # fmt: skip
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(hmm.OBS_COLS))


@pytest.mark.parametrize("t", [at18(DAYS[300]), rel(DAYS[700], 17), at18(DAYS[-1]),
                               at18(DAYS[-1]) + pd.Timedelta(days=2)])  # fmt: skip
def test_weekly_obs_hand_calc(t):
    got = hmm.weekly_obs(state(t), MCFG, HCFG)
    want = _naive_weekly(t)
    assert [pd.Timestamp(x).date() for x in got.index] == list(want.index)
    np.testing.assert_allclose(got.to_numpy(), want.to_numpy(), rtol=1e-10, equal_nan=True)


def test_weekly_sample_waits_for_decision_time():
    fri = next(d for d in DAYS[600:] if d.weekday() == 4)
    early = hmm.weekly_obs(state(rel(fri, 17, 59)), MCFG, HCFG)
    late = hmm.weekly_obs(state(at18(fri)), MCFG, HCFG)
    assert pd.Timestamp(late.index[-1]).date() == fri
    assert pd.Timestamp(early.index[-1]).date() < fri


# ---------------------------------------------------------------- HMM fit and filter
@pytest.fixture(scope="module")
def fitted():
    t = at18(DAYS[-1])
    x = hmm.training_set(hmm.weekly_obs(state(t), MCFG, HCFG))
    return hmm.fit(x, HCFG, DAYS[-1]), x


def test_fit_orders_states_by_vol(fitted):
    f, x = fitted
    assert np.all(np.diff(f.means[:, hmm.VOL_COL]) > 0)
    assert sorted(f.state_order) == [0, 1, 2]
    assert f.n_weeks == len(x) and 0 <= f.seed < HCFG.n_seeds
    # the forced stress block is labelled stress
    p = hmm.filtered(f, x)
    block = [i for i, d in enumerate(x.index) if DAYS[310] <= pd.Timestamp(d).date() <= DAYS[355]]
    assert p[block].argmax(axis=1).mean() > 1.5


def test_same_seed_same_fit(fitted):
    f, x = fitted
    g = hmm.fit(x, HCFG, DAYS[-1])
    assert g.seed == f.seed and g.log_likelihood == f.log_likelihood
    np.testing.assert_array_equal(g.means, f.means)
    np.testing.assert_array_equal(g.transmat, f.transmat)


def test_best_of_restarts(fitted):
    f, x = fitted
    one = hmm.fit(x, HCFG.model_copy(update={"n_seeds": 1}), DAYS[-1])
    assert f.log_likelihood >= one.log_likelihood


def test_too_few_weeks_raises(fitted):
    _, x = fitted
    with pytest.raises(ValueError, match="min_weeks"):
        hmm.fit(x.iloc[:10], HCFG, DAYS[-1])


def _naive_forward(f, x):
    xs = (x.to_numpy(dtype=float) - f.mu) / f.sd
    k = len(f.startprob)
    out, prev = [], None
    for t in range(len(xs)):
        lik = np.ones(k)
        if np.isfinite(xs[t]).all():
            for j in range(k):
                d = xs[t] - f.means[j]
                cov = f.covars[j]
                lik[j] = math.exp(-0.5 * d @ np.linalg.solve(cov, d)) / math.sqrt(
                    (2 * math.pi) ** len(d) * np.linalg.det(cov))
        pred = f.startprob if prev is None else prev @ f.transmat
        a = pred * lik
        prev = a / a.sum()
        out.append(prev)
    return np.array(out)


def test_filter_matches_naive_forward(fitted):
    f, x = fitted
    xx = x.iloc[:80].copy()
    xx.iloc[40, 2] = np.nan  # a missing observation: transition step only
    np.testing.assert_allclose(hmm.filtered(f, xx), _naive_forward(f, xx), rtol=1e-8, atol=1e-12)


def test_filtered_only_appending_future_changes_nothing(fitted):
    f, x = fitted
    full = hmm.filtered(f, x)
    for k in (50, 120, len(x) - 1):
        np.testing.assert_allclose(hmm.filtered(f, x.iloc[:k]), full[:k], rtol=1e-12)
    # the smoothed posterior would differ: make sure the test can see the difference
    from hmmlearn.hmm import GaussianHMM

    g = GaussianHMM(n_components=3, covariance_type="full")
    g.startprob_, g.transmat_, g.means_, g.covars_ = f.startprob, f.transmat, f.means, f.covars
    smooth = g.predict_proba((x.to_numpy() - f.mu) / f.sd)
    assert np.abs(smooth[:50] - full[:50]).max() > 1e-3


def test_save_load_roundtrip(fitted, tmp_path):
    f, x = fitted
    p = hmm.model_path(tmp_path, f)
    hmm.save(f, p)
    g = hmm.load(p)
    assert g.fit_date == f.fit_date and g.state_order == f.state_order
    np.testing.assert_array_equal(hmm.filtered(g, x), hmm.filtered(f, x))


def test_relabel_change(fitted):
    f, x = fitted
    assert hmm.relabel_change(f, f, x, 52) == 0.0
    perm = [2, 0, 1]  # swap labels: the same model with states renamed
    g = hmm.HmmFit(**{**{k: getattr(f, k) for k in hmm.HmmFit.__dataclass_fields__
                          if not k.startswith("_")},
                      "startprob": f.startprob[perm], "transmat": f.transmat[np.ix_(perm, perm)],
                      "means": f.means[perm], "covars": f.covars[perm]})  # fmt: skip
    ch = hmm.relabel_change(f, g, x, 52)
    assert ch > RCFG.hmm.stability_threshold


def test_fit_dates_first_sunday():
    d = hmm.fit_dates(date(2021, 12, 15), date(2022, 5, 31), 6)
    assert d == [date(2022, 1, 2), date(2022, 2, 6), date(2022, 3, 6), date(2022, 4, 3),
                 date(2022, 5, 1)]  # fmt: skip
    assert hmm.available_at(date(2022, 1, 2), RCFG.hmm) == rel(date(2022, 1, 2), 6)


# ---------------------------------------------------------------- HMM features at T
def _fits_monthly(until: date) -> list[hmm.HmmFit]:
    out = []
    for fd in hmm.fit_dates(date(2018, 1, 1), until, HCFG.refit_weekday):
        t = hmm.available_at(fd, HCFG)
        x = hmm.training_set(hmm.weekly_obs(state(t), MCFG, HCFG))
        if len(x) >= HCFG.min_weeks:
            out.append(hmm.fit(x, HCFG, fd))
        if len(out) == 3:
            break
    return out


@pytest.fixture(scope="module")
def fits():
    return _fits_monthly(DAYS[-1])


def test_fit_at_picks_newest_usable(fits):
    h = hmm.HmmInputs(fits, MCFG, HCFG)
    f0 = fits[0]
    assert h.fit_at(f0.available_ns - 1) is None
    assert h.fit_at(f0.available_ns) is f0
    assert h.fit_at(fits[1].available_ns + 1) is fits[1]


def test_probabilities_hand_calc_and_no_lookahead(fits):
    h = hmm.HmmInputs(fits, MCFG, HCFG)
    t = pd.Timestamp(fits[1].available_ns, tz=UTC) + pd.Timedelta(days=3, hours=14)
    m = state(t, regime=h)
    p = hmm.probabilities(m)
    x = _naive_weekly(t)
    want = _naive_forward(fits[1], x)[-1]
    np.testing.assert_allclose(p, want, rtol=1e-8)
    assert p.sum() == pytest.approx(1.0)
    # rows released after T, and fits not yet usable, change nothing
    macro = MACRO.copy()
    macro.loc[macro.released_at > t, "value"] = 99.0
    prices = {k: v.assign(value=np.where(v.released_at > t, 1e6, v.value))
              for k, v in PRICES.items()}  # fmt: skip
    later = hmm.HmmInputs(fits + [hmm.HmmFit(**{**{k: getattr(fits[2], k) for k in
                          hmm.HmmFit.__dataclass_fields__ if not k.startswith("_")},
                          "available_ns": int(t.value) + 1, "means": fits[2].means + 5})],
                          MCFG, HCFG)  # fmt: skip
    m2 = state(t, macro=macro, prices=prices, regime=later)
    np.testing.assert_array_equal(hmm.probabilities(m2), p)
    # and the mutation is visible later
    m3 = state(at18(DAYS[-1]), macro=macro, prices=prices, regime=later)
    assert not np.allclose(hmm.probabilities(m3), hmm.probabilities(state(at18(DAYS[-1]),
                                                                          regime=h)))


def test_probabilities_nan_without_fit():
    m = state(at18(DAYS[500]), regime=hmm.HmmInputs([], MCFG, HCFG))
    assert np.isnan(hmm.probabilities(m)).all()
    assert np.isnan(hmm.probabilities(state(at18(DAYS[500])))).all()


def test_daily_values_carry_the_weekly_step(fits):
    h = hmm.HmmInputs(fits[:1], MCFG, HCFG)
    fri = next(d for d in DAYS if d.weekday() == 4 and at18(d).value > fits[0].available_ns)
    mon = next(d for d in DAYS if d > fri)
    tue = next(d for d in DAYS if d > mon)
    pf = hmm.probabilities(state(at18(fri), regime=h))
    # Mon/Tue add no weekly sample; only revisions of the history could move the value
    for d in (mon, tue):
        np.testing.assert_allclose(hmm.probabilities(state(at18(d), regime=h)), pf, atol=1e-6)


def test_regime_features_and_row(fits):
    specs = regime_mod.register(MCFG, RCFG)
    names = {s.name for s in specs}
    assert {"stress_score", "hmm_p_calm", "hmm_p_normal", "hmm_p_stress"} <= names
    assert {f"quad_{q}" for q in quadrant.QUADRANTS} <= names
    t = at18(DAYS[-1])
    m = state(t, regime=hmm.HmmInputs(fits, MCFG, HCFG))
    vals = {s.name: s.fn(m) for s in specs}
    p = hmm.probabilities(m)
    assert [vals[f"hmm_p_{s}"] for s in hmm.STATES] == list(p)
    row = regime_mod.regime_row(m, MCFG, RCFG)
    assert row["hmm_stress"] == p[2] and row["stress_score"] == vals["stress_score"]
    assert row["quadrant"] is None  # no UNRATE / GDP / CPI in this fixture
    assert all(registry.get(s.name, s.version) is s for s in specs)
