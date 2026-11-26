"""Phase 3 step 4b: event clock (FOMC, CPI, earnings-heavy week; D-039).

Synthetic schedule snapshots (a moved date and a cancellation released later), an earnings
calendar with moved dates and live/backfilled rows, and a membership change. Every feature is
checked against a loop-based "known at T" rebuild and against a mutation of every row released
after T.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from qt.common import config
from qt.common.calendar import ET, TradingCalendar
from qt.features import events, market
from qt.features.inputs import MacroState

CFG = config.load().config.features.market
SPECS = events.register(CFG)
SESSIONS = np.array(TradingCalendar().cal.sessions.date, dtype="datetime64[D]")
SESS_DATES = [pd.Timestamp(x).date() for x in SESSIONS]
RNG = np.random.default_rng(5)
D = date.fromisoformat
EMPTY_MACRO = pd.DataFrame(columns=["series_id", "period", "value", "vintage", "released_at"])


def et(d: date, hh=0, mm=0) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)).tz_convert(UTC)


# ---------------------------------------------------------------- synthetic inputs
def _schedules(first_day: int, step: int, hh_known=0) -> pd.DataFrame:
    rows = []
    for y in (2021, 2022, 2023):
        ds = [date(y, m, first_day + (m % 3)) for m in range(1, 13, step)]
        rows.append({"year": y, "dates": ds, "released_at": et(date(y - 1, 12, 1), hh_known)})
    return pd.DataFrame(rows)


FOMC = _schedules(15, 2)
# 2022: a meeting moved (known 2022-04-20) and one cancelled (known 2022-08-30)
_d22 = list(FOMC.loc[FOMC.year == 2022, "dates"].iloc[0])
_moved = [d if d != _d22[2] else _d22[2] + timedelta(days=7) for d in _d22]
FOMC = pd.concat([FOMC, pd.DataFrame([
    {"year": 2022, "dates": _moved, "released_at": et(D("2022-04-20"), 10)},
    {"year": 2022, "dates": [d for d in _moved if d != _moved[4]],
     "released_at": et(D("2022-08-30"))},
])], ignore_index=True)  # fmt: skip
CPI = _schedules(10, 1)


def _members():
    snap1 = pd.DataFrame({"snapshot_at": et(D("2020-12-01")), "ticker": [f"T{i}" for i in
                          range(60)], "source": "w", "asset_id": range(60),
                          "released_at": et(D("2020-12-01"))})  # fmt: skip
    snap2 = pd.DataFrame({"snapshot_at": et(D("2022-06-01")), "ticker": [f"T{i}" for i in
                          range(50)], "source": "w", "asset_id": range(50),
                          "released_at": et(D("2022-06-01"))})  # fmt: skip
    return pd.concat([snap1, snap2], ignore_index=True)


MEMBERS = _members()


def _earnings():
    rows = []
    for a in range(70):  # 60-69 never members: ignored
        for q, (m, d0) in enumerate([(1, 20), (4, 18), (7, 18), (10, 17)] * 3):
            y = 2021 + q // 4
            rep = date(y, m, d0) + timedelta(days=int(RNG.integers(0, 30)))
            key = f"T{a}:{y}Q{q % 4}"
            live = RNG.random() < 0.3
            rel = et(rep - timedelta(days=21 if live else 7))
            rows.append({"asset_id": a, "quarter_key": key, "report_date": rep,
                         "released_at": rel})  # fmt: skip
            if RNG.random() < 0.15:  # moved later; the new date known 3 days before the old one
                rows.append({"asset_id": a, "quarter_key": key,
                             "report_date": rep + timedelta(days=int(RNG.integers(1, 10))),
                             "released_at": et(rep - timedelta(days=3), 7)})  # fmt: skip
    df = pd.DataFrame(rows)
    df["released_at"] = pd.to_datetime(df["released_at"], utc=True)
    return df


EARN = _earnings()


def state(t, fomc=FOMC, cpi=CPI, earn=EARN, members=MEMBERS) -> market.MarketState:
    ms = MacroState(EMPTY_MACRO)
    ms.advance(t)
    ev = events.EventInputs(fomc=fomc, cpi=cpi, earnings=earn, membership=members)
    return market.MarketState(t, ms, {}, SESSIONS, CFG, events=ev)


# ---------------------------------------------------------------- naive "known at T"
def n_schedule(df, t):
    best = {}
    for r in df.itertuples(index=False):
        if r.released_at <= t and (r.year not in best or r.released_at >= best[r.year][0]):
            best[r.year] = (r.released_at, r.dates)
    return sorted(d for _, ds in best.values() for d in ds)


def n_days(df, t, hhmm):
    h, m = map(int, hhmm.split(":"))
    today = t.tz_convert(ET).date()
    for d in n_schedule(df, t):
        if et(d, h, m) > t:
            return float((d - today).days)
    return math.nan


def n_weeks(t, earn=EARN, members=MEMBERS):
    c = CFG.events
    snaps = members[members["released_at"] <= t]
    last = snaps["snapshot_at"].max()
    mem = set(snaps.loc[snaps["snapshot_at"] == last, "asset_id"])
    today = t.tz_convert(ET).date()
    horizon = today + timedelta(days=c.earnings_visible_days)
    best = {}
    for r in earn.itertuples(index=False):
        k = (r.asset_id, r.quarter_key)
        if r.released_at <= t and (k not in best or r.released_at >= best[k][0]):
            best[k] = (r.released_at, r.report_date)
    reps = [(a, d) for (a, _), (_, d) in best.items() if a in mem and d <= horizon]
    nxt = [s for s in SESS_DATES if s > today]
    mon = nxt[0] - timedelta(days=nxt[0].weekday())
    out = {"sessions": math.nan, "share": math.nan}
    w = 0
    while mon <= horizon:
        share = len({a for a, d in reps if mon <= d <= mon + timedelta(days=6)}) / len(mem)
        if w == 0:
            out["share"] = share
        if share >= c.earnings_heavy_share:
            out["sessions"] = float(sum(1 for s in nxt if s < mon))
            break
        mon += timedelta(weeks=1)
        w += 1
    return out


HAND = {
    "fomc_days_next": lambda t: n_days(FOMC, t, CFG.events.fomc_statement_et),
    "cpi_days_next": lambda t: n_days(CPI, t, CFG.events.cpi_release_et),
    "earn_heavy_sessions_next": lambda t: n_weeks(t)["sessions"],
    "earn_week_share": lambda t: n_weeks(t)["share"],
}

_days = [D("2021-01-04") + timedelta(days=3 * i) for i in range(330)]
TIMES = [et(d, 18) for d in _days] + [et(d, 20, 15) for d in _days[::5]]  # + crypto 00:15 UTC
# around the statement/release instants: just before and after
TIMES += [et(d, 13, 59) for d in n_schedule(FOMC, et(D("2023-12-31")))]
TIMES += [et(d, 14, 0) for d in n_schedule(FOMC, et(D("2023-12-31")))]
TIMES += [et(d, 8, 29) for d in n_schedule(CPI, et(D("2023-12-31")))]
TIMES += [et(D("2022-04-20"), 9, 59), et(D("2022-04-20"), 10, 0)]


def _eq(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-12)


def test_every_feature_has_a_hand_case():
    assert {s.name for s in SPECS} == set(HAND)


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_hand_calc(spec):
    finite = 0
    for t in TIMES:
        got = market.compute(state(t), [spec])[spec.key]
        want = HAND[spec.name](t)
        assert _eq(got, want), (spec.name, t, got, want)
        finite += math.isfinite(want)
    assert finite > len(TIMES) // 8, (spec.name, finite)  # heavy weeks: ~20% of dates


def test_known_values():
    # 2022 schedule: the third meeting moved +7 days, known from 2022-04-20 10:00 ET
    before, after = et(D("2022-04-20"), 9, 59), et(D("2022-04-20"), 10, 0)
    v0 = market.compute(state(before), SPECS)[("fomc_days_next", 1)]
    v1 = market.compute(state(after), SPECS)[("fomc_days_next", 1)]
    assert v1 == v0 + 7
    # on a decision day: before 14:00 ET it's 0 days away; from 14:00 the next meeting
    d = n_schedule(FOMC, et(D("2021-12-31")))[1]
    assert market.compute(state(et(d, 13, 59)), SPECS)[("fomc_days_next", 1)] == 0
    assert market.compute(state(et(d, 14, 0)), SPECS)[("fomc_days_next", 1)] > 0
    # no events and no earnings: NaN, not an error
    ms = MacroState(EMPTY_MACRO)
    ms.advance(after)
    bare = market.compute(market.MarketState(after, ms, {}, SESSIONS, CFG), SPECS)
    assert all(math.isnan(v) for v in bare.values())


def _mutate_after(t):
    def sched(df):
        df = df.copy()
        late = df["released_at"] > t
        df.loc[late, "dates"] = df.loc[late, "dates"].map(
            lambda ds: [d + timedelta(days=int(RNG.integers(-5, 6))) for d in ds])
        extra = pd.DataFrame([{"year": 2022, "dates": [D("2022-01-02")] * 3,
                               "released_at": t + pd.Timedelta(seconds=1)}])  # fmt: skip
        return pd.concat([df, extra], ignore_index=True)

    e = EARN.copy()
    late = e["released_at"] > t
    e.loc[late, "report_date"] = e.loc[late, "report_date"].map(lambda d: d - timedelta(days=5))
    flood = pd.DataFrame({"asset_id": range(60), "quarter_key": [f"T{a}:x" for a in range(60)],
                          "report_date": t.tz_convert(ET).date() + timedelta(days=1),
                          "released_at": t + pd.Timedelta(minutes=1)})  # fmt: skip
    m = MEMBERS.copy()
    m.loc[m["released_at"] > t, "asset_id"] = 0
    extra_m = pd.DataFrame({"snapshot_at": t + pd.Timedelta(hours=1), "ticker": ["X"],
                            "source": "w", "asset_id": [0],
                            "released_at": t + pd.Timedelta(hours=1)})  # fmt: skip
    return (sched(FOMC), sched(CPI), pd.concat([e, flood], ignore_index=True),
            pd.concat([m, extra_m], ignore_index=True))  # fmt: skip


@pytest.mark.parametrize("spec", SPECS, ids=[s.name for s in SPECS])
def test_no_lookahead(spec):
    for t in TIMES[::7]:
        clean = market.compute(state(t), [spec])[spec.key]
        dirty = market.compute(state(t, *_mutate_after(t)), [spec])[spec.key]
        assert _eq(clean, dirty), (spec.name, t, clean, dirty)


def test_mutation_is_strong():
    """The mutation changes values when the rows are released (guards a no-op mutation)."""
    changed = set()
    for t in TIMES[:200:9]:
        late = _mutate_after(t)
        t2 = t + pd.Timedelta(hours=2)
        a = market.compute(state(t2), SPECS)
        b = market.compute(state(t2, *late), SPECS)
        changed |= {s.name for s in SPECS if not _eq(a[s.key], b[s.key])}
    assert changed == {s.name for s in SPECS}
