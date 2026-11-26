"""Phase 3 step 4b: crypto structure features (D-039).

Synthetic CoinGecko dominance snapshots and market caps, DefiLlama supply, SoSoValue flows, venue
funding (hourly and 8-hourly, with a gap and a late-starting venue) and OI snapshots (one venue
missing some days). Every feature is checked against a loop-based "known at T" rebuild and against
a mutation of every row released after T.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qt.common import config
from qt.common.calendar import ET, TradingCalendar
from qt.features import crypto, market
from qt.features.inputs import MacroState

CFG = config.load().config.features.market
C = CFG.crypto
SPECS = crypto.register(CFG)
SESSIONS = np.array(TradingCalendar().cal.sessions.date, dtype="datetime64[D]")
RNG = np.random.default_rng(3)
EMPTY_MACRO = pd.DataFrame(columns=["series_id", "period", "value", "vintage", "released_at"])
START, END = date(2026, 1, 1), date(2026, 6, 30)
DAYS = [START + timedelta(days=i) for i in range((END - START).days + 1)]


def utc(d: date, hh=0, mm=0) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, hh, mm, tzinfo=UTC))


def _frame(rows):
    df = pd.DataFrame(rows)
    for c in ("ts", "released_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def _daily(subjects, lvl, sd, released, skip=()):
    rows = []
    for s in subjects:
        x = lvl
        for d in DAYS:
            x *= math.exp(RNG.normal(0, sd))
            if d in skip:
                continue
            ts = utc(d)
            rows.append({"subject": s, "ts": ts, "value": x, "released_at": released(ts)})
    return _frame(rows)


GAP = {date(2026, 3, 10) + timedelta(days=i) for i in range(5)}
DOM = _daily([""], 55.0, 0.01, lambda ts: ts + pd.Timedelta(hours=6, minutes=45), skip=GAP)
DOM["ts"] = DOM["released_at"]  # /global snapshots: ts = updated_at ~ fetch time
MCAP = _daily(["BTC", "ETH"], 1e12, 0.03, lambda ts: ts + pd.Timedelta(days=1))
STABLE = _daily([""], 2e11, 0.002, lambda ts: ts + pd.Timedelta(days=1), skip={date(2026, 2, 3)})


def _etf():
    rows = []
    holiday = {date(2026, 4, 3)} | {date(2026, 5, 4) + timedelta(days=i) for i in range(10)}
    for s in ("US-BTC", "US-ETH"):
        for d in DAYS:
            if d.weekday() >= 5 or d in holiday:
                continue
            rel = pd.Timestamp(datetime(d.year, d.month, d.day, 9, tzinfo=ET))
            rel += pd.Timedelta(days=1)
            rows.append({"subject": s, "ts": utc(d), "value": RNG.normal(1e8, 4e8),
                         "released_at": rel.tz_convert(UTC)})  # fmt: skip
    return _frame(rows)


ETF = _etf()


def _funding():
    rows = []
    for sym in ("BTC", "ETH"):
        t = utc(START)
        while t <= utc(END):
            if not (utc(date(2026, 4, 10)) <= t < utc(date(2026, 4, 11), 8)):  # a 32 h gap
                rows.append({"venue": "hyperliquid", "symbol": sym, "ts": t,
                             "funding_rate": RNG.normal(1e-5, 2e-5), "funding_interval_h": 1.0,
                             "oi_usd": math.nan, "released_at": t})  # fmt: skip
            t += pd.Timedelta(hours=1)
        t = utc(date(2026, 3, 1))  # OKX starts later
        while t <= utc(END):
            rows.append({"venue": "okx", "symbol": sym, "ts": t,
                         "funding_rate": RNG.normal(1e-4, 1e-4), "funding_interval_h": 8.0,
                         "oi_usd": math.nan, "released_at": t})  # fmt: skip
            t += pd.Timedelta(hours=8)
    rows.append({"venue": "binance", "symbol": "BTC", "ts": utc(date(2026, 5, 1)),
                 "funding_rate": 0.5, "funding_interval_h": 8.0, "oi_usd": math.nan,
                 "released_at": utc(date(2026, 5, 1))})  # an unconfigured venue: ignored
    return _frame(rows)


FUND = _funding()


def _oi():
    rows = []
    for sym in ("BTC", "ETH"):
        for d in DAYS:
            for v in C.oi_venues:
                if v == "kraken" and d.day % 9 == 0:
                    continue  # a venue failing some days -> NaN that day
                t = utc(d, 6, 45)
                oi = float(RNG.uniform(1e9, 1e10))
                rows.append({"venue": v, "symbol": sym, "ts": t, "funding_rate": math.nan,
                             "funding_interval_h": math.nan, "oi_usd": oi,
                             "released_at": t})  # fmt: skip
    return _frame(rows)


OI = _oi()
INPUTS = dict(dominance=DOM, mcap=MCAP, stablecoins=STABLE, etf=ETF, funding=FUND, oi=OI)


def state(t, **over) -> market.MarketState:
    ms = MacroState(EMPTY_MACRO)
    ms.advance(t)
    cx = crypto.CryptoInputs(**{**INPUTS, **over})
    return market.MarketState(t, ms, {}, SESSIONS, CFG, crypto=cx)


# ---------------------------------------------------------------- naive "known at T"
def _rows(df, t, subject=None, col="subject"):
    out = [r for r in df.itertuples(index=False) if r.released_at <= t
           and (subject is None or getattr(r, col) == subject)]  # fmt: skip
    return sorted(out, key=lambda r: r.ts)


def n_level(df, t, subject=None):
    rs = _rows(df, t, subject)
    if not rs or t - rs[-1].released_at > pd.Timedelta(days=C.max_age_days):
        return math.nan
    return rs[-1].value


def n_change(df, t, days, pct):
    rs = _rows(df, t)
    if not rs or t - rs[-1].released_at > pd.Timedelta(days=C.max_age_days):
        return math.nan
    last = rs[-1]
    past = math.nan
    for r in rs:
        if last.ts - pd.Timedelta(days=days + C.match_tolerance_days) <= r.ts <= \
                last.ts - pd.Timedelta(days=days):  # fmt: skip
            past = r.value
    if pct:
        return last.value / past - 1 if past > 0 else math.nan
    return last.value - past


def n_funding(t, sym):
    vals = []
    for v in C.funding_venues:
        rs = [r for r in _rows(FUND, t, sym, "symbol") if r.venue == v
              and r.ts > t - pd.Timedelta(days=C.funding_days)]  # fmt: skip
        hours = sum(r.funding_interval_h for r in rs)
        if hours >= C.funding_min_coverage * 24 * C.funding_days:
            vals.append(sum(r.funding_rate for r in rs) * 8760 / hours)
    return sum(vals) / len(vals) if vals else math.nan


def n_oi(t, sym):
    tot = 0.0
    for v in C.oi_venues:
        rs = [r for r in _rows(OI, t, sym, "symbol") if r.venue == v
              and r.ts >= t - pd.Timedelta(hours=C.oi_max_age_h)]  # fmt: skip
        if not rs:
            return math.nan
        tot += rs[-1].oi_usd
    return tot / n_level(MCAP, t, sym)


def n_etf(t, sym):
    rs = _rows(ETF, t, C.etf_subject.format(symbol=sym))
    if len(rs) < C.etf_days:
        return math.nan
    last = rs[-C.etf_days :]
    if (t.tz_convert(ET).date() - last[0].ts.date()).days > C.etf_window_days:
        return math.nan
    return sum(r.value for r in last) / 1e9


HAND = {
    "btc_dominance": lambda t: n_level(DOM, t),
    f"btc_dominance_chg{C.dominance_chg_days}": lambda t: n_change(DOM, t, C.dominance_chg_days,
                                                                   False),
    f"stablecoin_supply_chg{C.stable_chg_days}": lambda t: n_change(STABLE, t, C.stable_chg_days,
                                                                    True),
}  # fmt: skip
for _s in C.symbols:
    HAND[f"funding_{C.funding_days}d_ann_{_s.lower()}"] = lambda t, s=_s: n_funding(t, s)
    HAND[f"oi_mcap_{_s.lower()}"] = lambda t, s=_s: n_oi(t, s)
    HAND[f"etf_flow_{C.etf_days}d_{_s.lower()}"] = lambda t, s=_s: n_etf(t, s)

TIMES = [utc(d, 0, 15) for d in DAYS[::2]] + [utc(d, 23) for d in DAYS[1::3]]


def _eq(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-15)


def test_every_feature_has_a_hand_case():
    assert {s.name for s in SPECS} == set(HAND)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_hand_calc(spec):
    finite = nan = 0
    for t in TIMES:
        got = market.compute(state(t), [spec])[spec.key]
        want = HAND[spec.name](t)
        assert _eq(got, want), (spec.name, t, got, want)
        finite += math.isfinite(want)
        nan += not math.isfinite(want)
    assert finite > len(TIMES) // 2, (spec.name, finite)
    assert nan > 0, spec.name  # each feature hits a gap / short history somewhere


def test_known_cases():
    k = ("btc_dominance", 1)
    in_gap = utc(date(2026, 3, 13), 12)  # last snapshot 3 days old -> stale
    assert math.isnan(market.compute(state(in_gap), SPECS)[k])
    f = (f"funding_{C.funding_days}d_ann_btc", 1)
    # before OKX has a full window, hyperliquid alone; across its gap, OKX alone
    t1 = utc(date(2026, 3, 3))
    assert market.compute(state(t1), SPECS)[f] == pytest.approx(n_funding(t1, "BTC"))
    t2 = utc(date(2026, 4, 12))
    hl_only = crypto.funding_ann(state(t2, funding=FUND[FUND.venue == "hyperliquid"]), "BTC")
    assert math.isnan(hl_only)  # 32 of 168 hours missing: below 90% coverage
    empty = {name: df.iloc[0:0] for name, df in INPUTS.items()}
    assert all(math.isnan(v) for v in market.compute(state(t2, **empty), SPECS).values())


def _mutate_after(t):
    out = {}
    for name, df in INPUTS.items():
        df = df.copy()
        late = df["released_at"] > t
        for col in ("value", "funding_rate", "oi_usd"):
            if col in df:
                df.loc[late, col] = df.loc[late, col] * RNG.uniform(1.5, 3, late.sum()) + 1
        extra = df.iloc[-1:].copy()  # a row dated before T but released after it
        extra["ts"] = t - pd.Timedelta(hours=1)
        extra["released_at"] = t + pd.Timedelta(seconds=1)
        out[name] = pd.concat([df, extra], ignore_index=True)
    return out


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_no_lookahead(spec):
    for t in TIMES[::6]:
        clean = market.compute(state(t), [spec])[spec.key]
        dirty = market.compute(state(t, **_mutate_after(t)), [spec])[spec.key]
        assert _eq(clean, dirty), (spec.name, t, clean, dirty)


def test_mutation_is_strong():
    changed = set()
    for t in TIMES[5:60:7]:
        late = _mutate_after(t)
        t2 = t + pd.Timedelta(days=4)  # spans a weekday flow release
        a = market.compute(state(t2), SPECS)
        b = market.compute(state(t2, **late), SPECS)
        changed |= {s.name for s in SPECS if not _eq(a[s.key], b[s.key])}
    assert changed == {s.name for s in SPECS}


def test_valid_from_drops_early_observations():
    k = (f"stablecoin_supply_chg{C.stable_chg_days}", 1)
    cfg = CFG.model_copy(update={"crypto": C.model_copy(
        update={"valid_from": {"stablecoins": date(2026, 3, 1)}})})  # fmt: skip
    late = STABLE[STABLE["ts"] >= utc(date(2026, 3, 1))]

    def st(t):
        ms = MacroState(EMPTY_MACRO)
        ms.advance(t)
        return market.MarketState(t, ms, {}, SESSIONS, cfg, crypto=crypto.CryptoInputs(**INPUTS))

    early = utc(date(2026, 3, 20))  # its 30-day base is before the cutoff
    assert math.isfinite(market.compute(state(early), SPECS)[k])
    assert math.isnan(market.compute(st(early), SPECS)[k])
    t = utc(date(2026, 5, 2))
    assert market.compute(st(t), SPECS)[k] == pytest.approx(
        n_change(late, t, C.stable_chg_days, True))
