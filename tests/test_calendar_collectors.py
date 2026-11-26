"""Phase 3 step 4b: FOMC and CPI calendar collectors (D-039), against fakes (no network)."""

from __future__ import annotations

import dataclasses
from datetime import UTC, date, datetime

import httpx
import pytest
from pydantic import SecretStr

from qt.collect import calendars as cal_mod
from qt.collect import pit
from qt.collect.base import run_collector
from qt.collect.calendars import (
    CpiCalendarCollector,
    FomcCalendarCollector,
    Meeting,
    monthly_last,
    parse_current,
    parse_historical,
    parse_meeting,
    year_rows,
)
from qt.collect.http import HttpClient
from qt.common import config
from qt.common.calendar import ET, TradingCalendar
from qt.db.conn import connect

D = date.fromisoformat
NOW = datetime(2026, 11, 26, 12, 0, tzinfo=ET).astimezone(UTC)
PIT = config.load().config.collect.pit


def et(d, hh=0, mm=0, ss=0):
    return datetime(*D(d).timetuple()[:3], hh, mm, ss, tzinfo=ET).astimezone(UTC)


# ---------------------------------------------------------------- PIT rules
def test_pit_rules():
    assert pit.cpi_schedule(PIT, D("2025-10-24")) == et("2024-12-01")
    assert pit.fomc_schedule(PIT, D("2026-03-18")) == et("2025-12-01")
    assert pit.fomc_unscheduled(PIT, D("2020-03-15")) == et("2020-03-15", 23, 59, 59)


def test_earnings_visibility_matches_backfill_rule():
    c = config.load().config
    lead = c.collect.pit.earnings_backfill_lead_days
    assert c.features.market.events.earnings_visible_days == lead


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize(
    "month,days,year,want",
    [
        ("January", "27-28", 2026, Meeting(D("2026-01-27"), D("2026-01-28"), "scheduled", False)),
        ("March", "17-18*", 2026, Meeting(D("2026-03-17"), D("2026-03-18"), "scheduled", True)),
        ("Apr/May", "30-1", 2024, Meeting(D("2024-04-30"), D("2024-05-01"), "scheduled", False)),
        ("April/May", "30-1", 2019, Meeting(D("2019-04-30"), D("2019-05-01"), "scheduled", False)),
        ("Jan/Feb", "31-1*", 2023, Meeting(D("2023-01-31"), D("2023-02-01"), "scheduled", True)),
        ("August", "22 (notation vote)", 2025,
         Meeting(D("2025-08-22"), D("2025-08-22"), "notation", False)),
        ("October", "4 (unscheduled)", 2019,
         Meeting(D("2019-10-04"), D("2019-10-04"), "unscheduled", False)),
        ("March", "17-18 (cancelled)", 2020,
         Meeting(D("2020-03-17"), D("2020-03-18"), "cancelled", False)),
        ("Dec/Jan", "31-1", 2030, Meeting(D("2030-12-31"), D("2031-01-01"), "scheduled", False)),
    ],
)  # fmt: skip
def test_parse_meeting(month, days, year, want):
    assert parse_meeting(month, days, year) == want


@pytest.mark.parametrize(
    "month,days",
    [("Smarch", "1-2"), ("March", "x"), ("March", "3 (postponed)"), ("March", "18-17")],
)
def test_parse_meeting_rejects(month, days):
    with pytest.raises(ValueError):
        parse_meeting(month, days, 2026)


def _row(month, days):
    return (f'<div class="row fomc-meeting"><div class="fomc-meeting__month col-xs-5">'
            f"<strong>{month}</strong></div>\n  <div class=\"fomc-meeting__date col-xs-4\">{days}"
            "</div><div>Statement</div></div>")  # fmt: skip


def _panel(year, rows):
    return (f'<div class="panel panel-default"><div class="panel-heading"><h4><a id="1">{year} '
            f"FOMC Meetings</a></h4></div>{''.join(_row(*r) for r in rows)}</div>")  # fmt: skip


CURRENT = (
    "<html><p>Next year</p>"
    + _panel(2026, [("January", "27-28"), ("March", "17-18*"), ("December", "8-9*")])
    + _panel(2025, [("January", "28-29"), ("August", "22 (notation vote)"),
                    ("December", "9-10*")])
    + _panel(2027, [("January", "26-27"), ("March", "16-17")])
    + '<div class="panel-heading"><h4>Other things</h4></div></html>'
)  # fmt: skip


def _hist(year, heads):
    return "<html>" + "".join(f"<h5>{h} - {year}</h5><p>Minutes</p>" for h in heads) + "</html>"


HIST = {
    2024: _hist(2024, ["January 30-31 Meeting", "Apr/May 30-1 Meeting", "December 17-18 Meeting"]),
    2023: _hist(2023, ["March 2 (unscheduled) Meeting", "March 17-18 (cancelled) Meeting",
                       "March 19 (notation vote)", "April 28-29 Meeting"]),
}  # fmt: skip


def test_parse_pages():
    y = parse_current(CURRENT)
    assert sorted(y) == [2025, 2026, 2027]
    assert [m.end for m in y[2026]] == [D("2026-01-28"), D("2026-03-18"), D("2026-12-09")]
    assert [m.label for m in y[2025]] == ["scheduled", "notation", "scheduled"]
    h = parse_historical(HIST[2023], 2023)
    assert [(m.end, m.label) for m in h] == [
        (D("2023-03-02"), "unscheduled"), (D("2023-03-18"), "cancelled"),
        (D("2023-03-19"), "notation"), (D("2023-04-29"), "scheduled"),
    ]  # fmt: skip
    assert parse_historical(HIST[2023], 2022) == []  # headings for another year are ignored


def test_monthly_last_drops_february_seasonal_posts():
    ds = [D("2019-01-11"), D("2019-02-11"), D("2019-02-13"), D("2019-03-12")]
    assert monthly_last(ds) == [D("2019-01-11"), D("2019-02-13"), D("2019-03-12")]


# ---------------------------------------------------------------- snapshot logic
def _yr(existing, dates, now, steps=(), known=None):
    known = known or et("2025-12-01")
    return year_rows(existing, source="s", kind="fomc_meetings", year=2026, dates=dates,
                     known_at=known, now=now, url="u", title="t", steps=steps)  # fmt: skip


def test_year_rows_first_fetch_backfill_and_live():
    ds = [D("2026-01-28"), D("2026-03-18")]
    (r,) = _yr({}, ds, NOW)
    assert r[1] == "fomc:2026" and r[8] == et("2025-12-01") and r[7]["approx"] is True
    early = et("2025-08-01", 9)  # fetched before Dec 1: known from the fetch
    (r,) = _yr({}, ds, early)
    assert r[8] == early and r[7]["approx"] is False


def test_year_rows_cancellation_and_changes():
    ds = [D("2026-01-28"), D("2026-03-18"), D("2026-04-29")]
    after = [D("2026-01-28"), D("2026-04-29")]
    rows = _yr({}, ds, NOW, steps=[(et("2026-03-17"), after)])
    assert [(r[8], r[7]["dates"]) for r in rows] == [
        (et("2025-12-01"), [d.isoformat() for d in ds]),
        (et("2026-03-17"), [d.isoformat() for d in after]),
    ]
    existing = {"fomc:2026": [(r[2], r[8]) for r in rows]}
    assert _yr(existing, ds, NOW, steps=[(et("2026-03-17"), after)]) == []  # unchanged
    moved = [D("2026-01-28"), D("2026-05-06")]
    (r,) = _yr(existing, moved, NOW)
    assert r[8] == NOW and r[7]["approx"] is False  # a later change is first-seen
    existing["fomc:2026"].append((r[2], r[8]))
    (back,) = _yr(existing, after, NOW.replace(year=2027))  # A -> B -> A is a new row
    assert back[2] not in {h for h, _ in existing["fomc:2026"]}


def test_year_rows_cancellation_already_known_live():
    ds = [D("2026-01-28"), D("2026-03-18")]
    early = et("2025-08-01", 9)
    rows = _yr({}, ds, early, steps=[(et("2026-03-17"), [D("2026-01-28")])])
    assert len(rows) == 1 and rows[0][7]["dates"] == ["2026-01-28"] and rows[0][8] == early


# ---------------------------------------------------------------- collectors end to end
@pytest.fixture
def env(settings, db):
    s = dataclasses.replace(
        settings, secrets=settings.secrets.model_copy(update={"fred_api_key": SecretStr("k")})
    )
    c = s.config
    cals = c.collect.calendars.model_copy(update={
        "fomc": c.collect.calendars.fomc.model_copy(update={"start_year": 2023}),
    })  # fmt: skip
    s = dataclasses.replace(s, config=c.model_copy(update={
        "collect": c.collect.model_copy(update={"calendars": cals})}))  # fmt: skip
    pages = {"/monetarypolicy/fomccalendars.htm": CURRENT}
    pages |= {f"/monetarypolicy/fomchistorical{y}.htm": h for y, h in HIST.items()}
    fred_dates = ["2025-01-15", "2025-02-11", "2025-02-12", "2025-10-24", "2025-12-18",
                  "2026-01-13", "2026-12-10"]  # fmt: skip
    calls = []

    def handler(req):
        calls.append(req.url.path)
        if req.url.host == "www.federalreserve.gov" and req.url.path in pages:
            return httpx.Response(200, text=pages[req.url.path])
        if req.url.path.endswith("/release/dates"):
            assert req.url.params["release_id"] == "10"
            ds = [{"release_id": 10, "date": d} for d in fred_dates]
            return httpx.Response(200, json={"count": len(ds), "release_dates": ds})
        return httpx.Response(404)

    http = HttpClient(s.config.collect.http, transport=httpx.MockTransport(handler),
                      sleep=lambda _: None)  # fmt: skip
    kw = dict(settings=s, connect=lambda: connect(s, "primary", db), http=http,
              cal=TradingCalendar(s.config.calendar))  # fmt: skip
    yield kw, pages, fred_dates, calls
    http.close()


def _q(kw, q, *p):
    with kw["connect"]() as c:
        return c.execute(q, p).fetchall()


def test_fomc_collector(env):
    kw, pages, _, calls = env
    out = run_collector(FomcCalendarCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    rows = _q(kw, "SELECT external_id, payload->'dates', released_at FROM raw_events "
              "WHERE kind = 'fomc_meetings' ORDER BY external_id, released_at")  # fmt: skip
    assert rows == [
        ("fomc:2023", ["2023-03-18", "2023-04-29"], et("2022-12-01")),
        ("fomc:2023", ["2023-04-29"], et("2023-03-17")),  # cancelled: gone from its first day
        ("fomc:2024", ["2024-01-31", "2024-05-01", "2024-12-18"], et("2023-12-01")),
        ("fomc:2025", ["2025-01-29", "2025-12-10"], et("2024-12-01")),
        ("fomc:2026", ["2026-01-28", "2026-03-18", "2026-12-09"], et("2025-12-01")),
        ("fomc:2027", ["2027-01-27", "2027-03-17"], NOW),  # published before Dec 1: first seen
    ]
    u = _q(kw, "SELECT external_id, payload->>'label', released_at FROM raw_events "
           "WHERE kind = 'fomc_unscheduled' ORDER BY 1")  # fmt: skip
    assert u == [
        ("fomc_u:2023-03-02", "unscheduled", et("2023-03-02", 23, 59, 59)),
        ("fomc_u:2023-03-19", "notation", et("2023-03-19", 23, 59, 59)),
        ("fomc_u:2025-08-22", "notation", et("2025-08-22", 23, 59, 59)),
    ]
    assert out.result.cursor["historical_done"] == [2023, 2024]

    # rerun: historical pages are not fetched again, nothing new is stored
    calls.clear()
    out = run_collector(FomcCalendarCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok" and out.result.total == 0
    assert calls == ["/monetarypolicy/fomccalendars.htm"]

    # the Fed moves a 2026 meeting: a first-seen snapshot; the old one stays
    later = NOW.replace(day=27)
    pages["/monetarypolicy/fomccalendars.htm"] = CURRENT.replace("8-9*", "15-16*")
    out = run_collector(FomcCalendarCollector(clock=lambda: later), **kw)
    rows = _q(kw, "SELECT payload->'dates', released_at FROM raw_events "
              "WHERE external_id = 'fomc:2026' ORDER BY released_at")  # fmt: skip
    assert rows == [(["2026-01-28", "2026-03-18", "2026-12-09"], et("2025-12-01")),
                    (["2026-01-28", "2026-03-18", "2026-12-16"], later)]  # fmt: skip


def test_fomc_collector_fails_loudly_on_layout_change(env):
    kw, pages, _, _ = env
    pages["/monetarypolicy/fomccalendars.htm"] = "<html>redesigned</html>"
    out = run_collector(FomcCalendarCollector(clock=lambda: NOW), **kw)
    assert out.status == "failed" and "layout" in out.error


def test_cpi_collector(env):
    kw, _, fred_dates, _ = env
    out = run_collector(CpiCalendarCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    rows = _q(kw, "SELECT external_id, payload->'dates', (payload->>'approx')::bool, released_at "
              "FROM raw_events WHERE kind = 'cpi_releases' ORDER BY 1")  # fmt: skip
    assert rows == [
        ("cpi:2025", ["2025-01-15", "2025-02-12", "2025-10-24", "2025-12-18"], True,
         et("2024-12-01")),
        ("cpi:2026", ["2026-01-13", "2026-12-10"], True, et("2025-12-01")),
    ]  # fmt: skip
    out = run_collector(CpiCalendarCollector(clock=lambda: NOW), **kw)
    assert out.result.total == 0
    fred_dates.append("2027-01-14")  # next year's schedule appears: first seen
    out = run_collector(CpiCalendarCollector(clock=lambda: NOW), **kw)
    assert _q(kw, "SELECT released_at FROM raw_events WHERE external_id = 'cpi:2027'") == [(NOW,)]


def test_cpi_collector_needs_key(env, settings):
    kw, *_ = env
    s = dataclasses.replace(kw["settings"], secrets=kw["settings"].secrets.model_copy(
        update={"fred_api_key": None}))  # fmt: skip
    out = run_collector(CpiCalendarCollector(clock=lambda: NOW), **dict(kw, settings=s))
    assert out.status == "skipped"


def test_collectors_scheduled_weekly():
    from qt.orchestrator.jobs import JOB_COLLECTORS

    names = [c.name for c in JOB_COLLECTORS["weekly"]()]
    assert "fomc_cal" in names and "cpi_cal" in names
    fr = config.load().config.collect.freshness.collectors
    assert {"fomc_cal", "cpi_cal"} <= set(fr)
    assert cal_mod.FOMC_SOURCE != "fed"  # never mixed with the Fed RSS rows
