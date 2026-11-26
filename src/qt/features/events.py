"""Event clock (component doc 03 section 5.3): days to the next FOMC decision and CPI release, and
the next earnings-heavy week (Phase 3 step 4b, D-039).

Features are functions of a :class:`qt.features.market.MarketState` whose ``events`` field holds
an :class:`EventInputs`. Everything is cut at T inside the feature:

- FOMC / CPI: each year's schedule is its newest snapshot released by T
  (``qt.collect.calendars``), so a moved or cancelled date is seen only from when it was known.
  The next event is the first whose release instant (14:00 / 08:30 ET) is after T; days are
  calendar days from T's New York date.
- Earnings: the newest observation per (asset, quarter) released by T, PIT members at T, and only
  report dates up to ``T + earnings_visible_days`` (the backfill's horizon, so training and live
  see the same thing). Decision times are after the close, so the current week is the week of
  the next XNYS session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from qt.common.config import MarketCfg
from qt.features import registry
from qt.features.inputs import ET, members_at
from qt.features.market import MarketState

G_EVENTS = "event_clock"
FOMC = "fomc_calendar"
CPI = "cpi_calendar"
EARNINGS = "earnings_calendar"


@dataclass
class EventInputs:
    fomc: pd.DataFrame  # inputs.load_schedules: year, dates, released_at
    cpi: pd.DataFrame
    earnings: pd.DataFrame | None = None  # inputs.load_earnings rows
    membership: pd.DataFrame | None = None  # inputs.load_membership rows


def schedule_at(df: pd.DataFrame | None, asof: pd.Timestamp) -> list[date]:
    """Union of each year's newest snapshot released by ``asof``."""
    if df is None or df.empty:
        return []
    k = df[df["released_at"] <= asof].sort_values("released_at", kind="stable")
    out: set[date] = set()
    for ds in k.groupby("year", sort=False)["dates"].last():
        out.update(ds)
    return sorted(out)


def _at_et(d: date, hhmm: str) -> pd.Timestamp:
    h, m = (int(x) for x in hhmm.split(":")[:2])
    return pd.Timestamp(datetime(d.year, d.month, d.day, h, m, tzinfo=ET))


def days_to_next(m: MarketState, df: pd.DataFrame | None, hhmm: str) -> float:
    today = m.asof.tz_convert(ET).date()
    for d in schedule_at(df, m.asof):
        if _at_et(d, hhmm) > m.asof:
            return float((d - today).days)
    return math.nan


def _week_start(d: np.datetime64) -> date:
    x = pd.Timestamp(d).date()
    return x - timedelta(days=x.weekday())


def earnings_weeks(m: MarketState) -> dict[str, float]:
    """``sessions``: XNYS sessions from the next session to the first session of the first
    earnings-heavy week (0 = the current week); ``share``: share of members reporting this week."""
    key = ("earn_weeks",)
    if key in m._cache:
        return m._cache[key]
    out = {"sessions": math.nan, "share": math.nan}
    m._cache[key] = out
    ev = m.events
    if ev is None or ev.earnings is None or ev.membership is None or ev.earnings.empty:
        return out
    members = members_at(ev.membership, m.asof)
    if not members:
        return out
    c = m.cfg.events
    today = m.asof.tz_convert(ET).date()
    horizon = today + timedelta(days=c.earnings_visible_days)
    e = ev.earnings[ev.earnings["released_at"] <= m.asof]
    e = e.sort_values("released_at", kind="stable").groupby(["asset_id", "quarter_key"]).last()
    e = e.reset_index()
    e = e[e["asset_id"].isin(members) & (e["report_date"] <= horizon)]
    nxt = m.sessions[m.sessions > np.datetime64(today, "D")]
    if not len(nxt):
        return out
    anchor = _week_start(nxt[0])
    n = len(members)
    w = 0
    while anchor + timedelta(weeks=w) <= horizon:
        mon = anchor + timedelta(weeks=w)
        sun = mon + timedelta(days=6)
        share = e.loc[(e["report_date"] >= mon) & (e["report_date"] <= sun), "asset_id"].nunique()
        share /= n
        if w == 0:
            out["share"] = float(share)
        if share >= c.earnings_heavy_share:
            first = int(np.searchsorted(nxt, np.datetime64(mon, "D"), side="left"))
            out["sessions"] = float(first)
            break
        w += 1
    return out


def register(cfg: MarketCfg) -> list[registry.FeatureSpec]:
    """Register the event-clock features (idempotent). Returns the specs."""
    c = cfg.events
    keys: list[tuple[str, int]] = []

    def reg(name, formula, sources, fn):
        registry.feature(name, 1, kind="market", group=G_EVENTS, formula=formula,
                         sources=tuple(sources))(fn)  # fmt: skip
        keys.append((name, 1))

    def ev(m, name):
        return getattr(m.events, name) if m.events is not None else None

    reg("fomc_days_next", f"calendar days to the next scheduled FOMC decision day "
        f"(statement {c.fomc_statement_et} ET after T)", [FOMC],
        lambda m: days_to_next(m, ev(m, "fomc"), c.fomc_statement_et))  # fmt: skip
    reg("cpi_days_next", f"calendar days to the next CPI release ({c.cpi_release_et} ET after T)",
        [CPI], lambda m: days_to_next(m, ev(m, "cpi"), c.cpi_release_et))  # fmt: skip
    reg("earn_heavy_sessions_next",
        f"sessions from the next session to the first week where >= "
        f"{c.earnings_heavy_share:.0%} of members report (0 = this week; report dates up to "
        f"T + {c.earnings_visible_days} d)", [EARNINGS, "membership"],
        lambda m: earnings_weeks(m)["sessions"])  # fmt: skip
    reg("earn_week_share", f"share of members reporting in the next session's week (report dates "
        f"up to T + {c.earnings_visible_days} d)", [EARNINGS, "membership"],
        lambda m: earnings_weeks(m)["share"])  # fmt: skip
    return [registry.get(*k) for k in keys]
