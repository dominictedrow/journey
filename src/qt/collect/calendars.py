"""Event calendars -> raw_events: FOMC meetings and CPI release dates (Phase 3 step 4b, D-039).

Each year's schedule is stored as one **snapshot** row (``external_id`` = ``fomc:2026`` /
``cpi:2026``, ``payload.dates`` = the decision / release dates). A changed schedule is a new row
with a new ``content_hash``, so the schedule as known at T is the newest snapshot per year with
``released_at <= T``, and a moved or cancelled date never leaks backwards.

``released_at`` of a year's **first** snapshot is ``min(fetch time, Dec 1 of the preceding
year)`` (``pit.fomc_schedule`` / ``pit.cpi_schedule``): the schedules are published well before
then. A later changed snapshot is first-seen (fetch time).

**FOMC (``fomc_cal``).** Scraped from federalreserve.gov (no API): the calendars page carries the
last ~5 years and the next one; years before that come from the per-year historical pages
(fetched once each). Scheduled meetings only go in the snapshot; their decision day is the last
day of the meeting. A meeting the page marks *cancelled* (March 2020) was in the schedule known the
December before, so a first fetch writes the schedule with it, then the schedule without it from
00:00 ET of its first day (no later than the Fed announced it). Unscheduled meetings and notation
votes are stored as their own rows (kind ``fomc_unscheduled``), released at the end of their day.

**CPI (``cpi_cal``).** FRED ``release/dates`` for the BLS CPI release, including scheduled dates
with no data yet. FRED also lists the February seasonal-factor posts, so only the last date in
each month is kept; that matched the first ALFRED vintage of every CPIAUCSL month 2015-2026
(probe 2026-11-26). FRED serves only the dates that happened, so a backfilled year includes any
reschedule (the 2025 shutdown) from the December before: ``payload.approx`` is true there.
"""

from __future__ import annotations

import calendar as _calendar
import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from qt.collect import pit
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.collect.fed import EVENT_COLS
from qt.collect.fred import _api

log = logging.getLogger("qt.collect.calendars")

FOMC_SOURCE, FOMC_KIND, FOMC_U_KIND = "fed_calendar", "fomc_meetings", "fomc_unscheduled"
CPI_SOURCE, CPI_KIND = "fred_calendar", "cpi_releases"

_MONTHS = {m.lower(): i for i, m in enumerate(_calendar.month_name) if m}
_MONTHS |= {m.lower(): i for i, m in enumerate(_calendar.month_abbr) if m}
_MONTHS["sept"] = 9
_LABELS = {"": "scheduled", "unscheduled": "unscheduled", "notation vote": "notation",
           "cancelled": "cancelled", "canceled": "cancelled"}  # fmt: skip


def _now() -> datetime:
    return datetime.now(UTC)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


# ---------------------------------------------------------------- FOMC parsing
@dataclass(frozen=True)
class Meeting:
    start: date
    end: date  # the decision (statement) day
    label: str  # scheduled | unscheduled | notation | cancelled
    sep: bool  # Summary of Economic Projections (marked * on the page)


_DAYS_RE = re.compile(r"^(\d{1,2})(?:\s*-\s*(\d{1,2}))?\s*(\*)?\s*(?:\(([^)]*)\))?")


def parse_meeting(month: str, days: str, year: int) -> Meeting:
    """``("Apr/May", "30-1", 2024)``, ``("March", "17-18*", 2026)``, ``("August", "22 (notation
    vote)", 2025)``. Raises ValueError on anything it doesn't recognise."""
    parts = [p.strip().lower().rstrip(".") for p in month.split("/")]
    try:
        m1, m2 = _MONTHS[parts[0]], _MONTHS[parts[-1]]
    except KeyError as e:
        raise ValueError(f"unknown FOMC month {month!r}") from e
    m = _DAYS_RE.match(days.strip())
    if not m:
        raise ValueError(f"unknown FOMC days {days!r}")
    d1, d2 = int(m.group(1)), int(m.group(2) or m.group(1))
    note = (m.group(4) or "").strip().lower()
    if note not in _LABELS:
        raise ValueError(f"unknown FOMC meeting note {note!r} in {days!r}")
    start = date(year, m1, d1)
    end = date(year + (1 if m2 < m1 else 0), m2, d2)
    if end < start:
        raise ValueError(f"FOMC meeting ends before it starts: {month} {days} {year}")
    return Meeting(start, end, _LABELS[note], bool(m.group(3)))


_PANEL_RE = re.compile(r'<div class="panel-heading">\s*<h\d[^>]*>(.*?)</h\d>', re.S)
_ROW_RE = re.compile(
    r'fomc-meeting__month[^>]*>(.*?)</div>\s*<div class="fomc-meeting__date[^>]*>(.*?)</div>', re.S
)
_HIST_RE = re.compile(r"<h5[^>]*>(.*?)</h5>", re.S)
_HIST_HEAD = re.compile(r"^(.+?)\s+(\d{1,2}(?:-\d{1,2})?\*?(?:\s*\([^)]*\))?)\s*(?:Meeting)?\s*-\s*"
                        r"(\d{4})$")  # fmt: skip


def parse_current(html: str) -> dict[int, list[Meeting]]:
    """The calendars page: one panel per year ("2026 FOMC Meetings")."""
    heads = [(m.start(), _text(m.group(1))) for m in _PANEL_RE.finditer(html)]
    out: dict[int, list[Meeting]] = {}
    for i, (pos, head) in enumerate(heads):
        ym = re.match(r"^(\d{4}) FOMC Meetings$", head)
        if not ym:
            continue
        end = heads[i + 1][0] if i + 1 < len(heads) else len(html)
        year = int(ym.group(1))
        out[year] = [parse_meeting(_text(a), _text(b), year)
                     for a, b in _ROW_RE.findall(html[pos:end])]  # fmt: skip
    return {y: v for y, v in out.items() if v}


def parse_historical(html: str, year: int) -> list[Meeting]:
    """A historical year page: one ``<h5>`` per meeting ("April/May 30-1 Meeting - 2019")."""
    out = []
    for m in _HIST_RE.finditer(html):
        h = _HIST_HEAD.match(_text(m.group(1)))
        if h and int(h.group(3)) == year:
            out.append(parse_meeting(h.group(1), h.group(2), year))
    return out


# ---------------------------------------------------------------- snapshots
def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


Existing = dict[str, list[tuple[str, datetime]]]  # external_id -> [(content_hash, released_at)]


def existing_rows(ctx: RunCtx, source: str) -> Existing:
    out: Existing = {}
    for eid, h, rel in ctx.conn.execute(
        "SELECT external_id, content_hash, released_at FROM raw_events WHERE source = %s "
        "ORDER BY released_at",
        (source,),
    ).fetchall():
        out.setdefault(eid, []).append((h, rel))
    return out


def year_rows(
    existing: Existing,
    *,
    source: str,
    kind: str,
    year: int,
    dates: list[date],
    known_at: datetime,
    now: datetime,
    url: str,
    title: str,
    steps: Iterable[tuple[datetime, list[date]]] = (),
    extra: dict[str, Any] | None = None,
) -> list[tuple]:
    """Snapshot rows for one year's schedule.

    First fetch of the year: ``dates`` released at ``min(now, known_at)``, then each of ``steps``
    (``(when, dates)``, e.g. a cancellation) at ``min(now, when)``. Later fetches: ``steps[-1]``
    dates (or ``dates``) as one first-seen row, unless that is already the newest snapshot.
    """
    eid = f"{kind.split('_')[0]}:{year}"
    steps = list(steps)
    have = existing.get(eid, [])
    if not have:
        base = min(now, known_at)
        snaps = [(base, dates)]
        for when, ds in steps:
            at = min(now, when)
            if at > snaps[-1][0]:
                snaps.append((at, ds))
            else:  # already known by the previous snapshot's time: replace it
                snaps[-1] = (snaps[-1][0], ds)
        approx = now > known_at
    else:
        snaps = [(now, steps[-1][1] if steps else dates)]
        approx = False
    rows = []
    latest_hash = have[-1][0] if have else None
    seen = {h for h, _ in have}
    for rel, ds in snaps:
        iso = sorted(d.isoformat() for d in ds)
        h = _hash(iso)
        if h == latest_hash:
            continue  # unchanged
        if h in seen:  # back to an older schedule (A -> B -> A): a new row, not a no-op
            h = _hash([*iso, rel.isoformat()])
        payload = {"year": year, "dates": iso, "approx": approx, **(extra or {})}
        first = min(ds) if ds else date(year, 1, 1)
        rows.append((source, eid, h, kind, pit.at_et(first, "00:00"), title, url, payload, rel))
        latest_hash = h
        seen.add(h)
    return rows


# ---------------------------------------------------------------- collectors
class FomcCalendarCollector:
    name = "fomc_cal"
    required_secrets: tuple[str, ...] = ()

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.calendars.fomc
        pcfg = ctx.settings.config.collect.pit
        cursor = dict(ctx.cursor)
        res = CollectResult(cursor=cursor)
        now = self.clock()

        years = parse_current(ctx.http.get(cfg.current_url).text)
        if len(years) < 2:
            raise RuntimeError(f"FOMC page parsed into {len(years)} years; layout changed?")
        urls = {y: cfg.current_url for y in years}
        done = set(cursor.get("historical_done", []))
        for y in range(cfg.start_year, min(years)):
            if y in done:
                continue
            url = cfg.historical_url.format(year=y)
            ms = parse_historical(ctx.http.get(url).text, y)
            if not ms:
                raise RuntimeError(f"FOMC historical page {y} has no meetings; layout changed?")
            years[y], urls[y] = ms, url

        existing = existing_rows(ctx, FOMC_SOURCE)
        rows: list[tuple] = []
        for y in sorted(years):
            ms = years[y]
            sched = [m for m in ms if m.label in ("scheduled", "cancelled")]
            if sched:
                steps, cur = [], [m.end for m in sched]
                cancelled = sorted((m for m in sched if m.label == "cancelled"),
                                   key=lambda m: m.start)  # fmt: skip
                for c in cancelled:
                    cur = [d for d in cur if d != c.end]
                    steps.append((pit.at_et(c.start, "00:00"), list(cur)))
                rows += year_rows(
                    existing, source=FOMC_SOURCE, kind=FOMC_KIND, year=y,
                    dates=[m.end for m in sched], known_at=pit.fomc_schedule(pcfg, sched[0].end),
                    now=now, url=urls[y], title=f"{y} FOMC meetings", steps=steps,
                    extra={"sep": sorted(m.end.isoformat() for m in sched if m.sep),
                           "cancelled": sorted(m.end.isoformat() for m in sched
                                               if m.label == "cancelled")},
                )  # fmt: skip
            for m in ms:
                if m.label in ("unscheduled", "notation"):
                    p = {"start": m.start.isoformat(), "end": m.end.isoformat(), "label": m.label}
                    rows.append((
                        FOMC_SOURCE, f"fomc_u:{m.end.isoformat()}", _hash(p), FOMC_U_KIND,
                        pit.at_et(m.end, "00:00"), f"FOMC {m.label} {m.end.isoformat()}",
                        urls[y], p, min(now, pit.fomc_unscheduled(pcfg, m.end)),
                    ))  # fmt: skip
        res.add("raw_events", insert_rows(ctx.conn, "raw_events", EVENT_COLS, rows))
        cursor["historical_done"] = sorted(done | {y for y in years if urls[y] != cfg.current_url})
        cursor["years_on_current_page"] = sorted(y for y in years if urls[y] == cfg.current_url)
        return res


def monthly_last(dates: Iterable[date]) -> list[date]:
    """The last date in each calendar month (drops FRED's February seasonal-factor posts)."""
    last: dict[tuple[int, int], date] = {}
    for d in sorted(dates):
        last[(d.year, d.month)] = d
    return sorted(last.values())


class CpiCalendarCollector:
    name = "cpi_cal"
    required_secrets: tuple[str, ...] = ("fred_api_key",)

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.calendars.cpi
        pcfg = ctx.settings.config.collect.pit
        res = CollectResult(cursor=dict(ctx.cursor))
        now = self.clock()
        dates = monthly_last(_api(ctx).release_dates(cfg.release_id, cfg.start))
        if not dates:
            raise RuntimeError(f"FRED release {cfg.release_id} returned no dates")
        by_year: dict[int, list[date]] = {}
        for d in dates:
            by_year.setdefault(d.year, []).append(d)
        existing = existing_rows(ctx, CPI_SOURCE)
        base = ctx.settings.config.collect.fred.base_url
        url = f"{base}/release/dates?release_id={cfg.release_id}"
        rows: list[tuple] = []
        for y, ds in sorted(by_year.items()):
            rows += year_rows(
                existing, source=CPI_SOURCE, kind=CPI_KIND, year=y, dates=ds,
                known_at=pit.cpi_schedule(pcfg, ds[0]), now=now, url=url,
                title=f"{y} CPI release dates",
            )  # fmt: skip
        res.add("raw_events", insert_rows(ctx.conn, "raw_events", EVENT_COLS, rows))
        res.cursor["last_date"] = dates[-1].isoformat()
        return res
