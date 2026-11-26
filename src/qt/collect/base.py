"""Collector contract, run wrapper, insert helper and freshness (component doc sections 5.1, 8).

Each collector run:

1. is skipped, loudly, if a required secret is missing (``collector_state.last_status =
   'skipped'``), never reported as a fake "ok"
2. runs inside :func:`qt.common.runs.run_context` (stage ``collect_<name>``), so it gets a
   ``runs`` row, run-tagged logs and, on failure, a critical alert
3. records ``collector_state`` (last ok / attempt / error / cursor / rows)

A failing collector returns a ``failed`` outcome instead of raising, so the other collectors in
the same scheduler job keep running.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from qt.collect.http import HttpClient
from qt.common.calendar import ET, TradingCalendar
from qt.common.config import FreshnessCfg, FreshnessRule, Settings
from qt.common.runs import run_context

log = logging.getLogger("qt.collect")

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,12}$")  # stage collect_<name> must fit runs' 21 chars
CHUNK = 1000

Status = Literal["ok", "failed", "skipped"]
Connect = Callable[[], psycopg.Connection]


# ---------------------------------------------------------------- contract
@dataclass
class RunCtx:
    run_id: str
    settings: Settings
    conn: psycopg.Connection
    http: HttpClient
    cal: TradingCalendar
    cursor: dict[str, Any]
    started_at: datetime  # tz-aware UTC; the fetch time for first-seen rules


@dataclass
class CollectResult:
    rows: dict[str, int] = field(default_factory=dict)  # table -> rows inserted
    rejected: int = 0  # rows dropped because they could not be timed
    cursor: dict[str, Any] | None = None  # new resume cursor; None keeps the old one

    @property
    def total(self) -> int:
        return sum(self.rows.values())

    def add(self, table: str, stats: InsertStats) -> None:
        self.rows[table] = self.rows.get(table, 0) + stats.inserted
        self.rejected += stats.rejected


class Collector(Protocol):
    name: str
    required_secrets: tuple[str, ...]  # Secrets field names, e.g. ("fred_api_key",)

    def collect(
        self, ctx: RunCtx, *, since: datetime | None, until: datetime
    ) -> CollectResult: ...


@dataclass
class Outcome:
    collector: str
    status: Status
    result: CollectResult | None = None
    error: str | None = None
    run_id: str | None = None


# ---------------------------------------------------------------- inserts
@dataclass
class InsertStats:
    inserted: int = 0
    rejected: int = 0


def _untimed(row: Sequence[Any], idx: int) -> bool:
    v = row[idx]
    return not isinstance(v, datetime) or v.tzinfo is None or v.utcoffset() is None


def insert_rows(
    conn: psycopg.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> InsertStats:
    """``INSERT ... ON CONFLICT DO NOTHING`` in chunks. Re-runs and overlapping backfills are safe.

    If ``released_at`` is among the columns, rows where it's missing or naive are rejected and
    logged (section 3: never silently default to now()).
    """
    cols = list(columns)
    rel = cols.index("released_at") if "released_at" in cols else None
    stats = InsertStats()
    good: list[Sequence[Any]] = []
    for r in rows:
        if len(r) != len(cols):
            raise ValueError(f"{table}: row has {len(r)} values, expected {len(cols)}")
        if rel is not None and _untimed(r, rel):
            stats.rejected += 1
            continue
        good.append(r)
    if stats.rejected:
        log.warning("rows_rejected_untimed", extra={"table": table, "rows": stats.rejected})

    head = sql.SQL("INSERT INTO {} ({}) VALUES ").format(
        sql.Identifier(table), sql.SQL(", ").join(map(sql.Identifier, cols))
    )
    one = sql.SQL("({})").format(sql.SQL(", ").join(sql.Placeholder() * len(cols)))
    for i in range(0, len(good), CHUNK):
        chunk = good[i : i + CHUNK]
        stmt = head + sql.SQL(", ").join([one] * len(chunk)) + sql.SQL(" ON CONFLICT DO NOTHING")
        params = [_adapt(v) for r in chunk for v in r]
        stats.inserted += conn.execute(stmt, params).rowcount
    return stats


def _adapt(v: Any) -> Any:
    return Jsonb(v) if isinstance(v, dict) else v


# ---------------------------------------------------------------- collector_state
def load_cursor(conn: psycopg.Connection, collector: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT cursor FROM collector_state WHERE collector = %s", (collector,)
    ).fetchone()
    return dict(row[0]) if row else {}


def record_state(
    conn: psycopg.Connection,
    collector: str,
    status: Status,
    at: datetime,
    *,
    error: str | None = None,
    rows: int | None = None,
    cursor: dict[str, Any] | None = None,
) -> None:
    """Upsert ``collector_state``. Only an ok run moves ``last_ok_at`` and the cursor."""
    ok = status == "ok"
    conn.execute(
        """
        INSERT INTO collector_state
            (collector, last_ok_at, last_attempt_at, last_status, last_error, cursor, rows_last_run)
        VALUES (%(c)s, %(ok_at)s, %(at)s, %(st)s, %(err)s,
                coalesce(%(cur)s::jsonb, '{}'::jsonb), %(rows)s)
        ON CONFLICT (collector) DO UPDATE SET
            last_ok_at      = coalesce(EXCLUDED.last_ok_at, collector_state.last_ok_at),
            last_attempt_at = EXCLUDED.last_attempt_at,
            last_status     = EXCLUDED.last_status,
            last_error      = EXCLUDED.last_error,
            cursor          = CASE WHEN %(cur)s::jsonb IS NULL THEN collector_state.cursor
                                   ELSE EXCLUDED.cursor END,
            rows_last_run   = EXCLUDED.rows_last_run
        """,
        {
            "c": collector,
            "ok_at": at if ok else None,
            "at": at,
            "st": status,
            "err": error,
            "cur": Jsonb(cursor) if (ok and cursor is not None) else None,
            "rows": rows,
        },
    )


# ---------------------------------------------------------------- run wrapper
def _now() -> datetime:
    return datetime.now(UTC)


def run_collector(
    collector: Collector,
    *,
    settings: Settings,
    connect: Connect,
    http: HttpClient,
    cal: TradingCalendar,
    since: datetime | None = None,
    until: datetime | None = None,
    clock: Callable[[], datetime] = _now,
) -> Outcome:
    name = collector.name
    if not NAME_RE.match(name):
        raise ValueError(f"collector name {name!r} must match {NAME_RE.pattern}")
    started = clock()

    missing = [s for s in collector.required_secrets if not settings.secrets.has(s)]
    if missing:
        err = "missing secret(s): " + ", ".join(m.upper() for m in missing)
        log.warning("collector_skipped", extra={"collector": name, "reason": err})
        with connect() as conn:
            record_state(conn, name, "skipped", started, error=err)
        return Outcome(name, "skipped", error=err)

    run_id: str | None = None
    try:
        with run_context(f"collect_{name}", settings, connect) as run:
            run_id = run.run_id
            with connect() as conn:
                ctx = RunCtx(
                    run_id=run.run_id,
                    settings=settings,
                    conn=conn,
                    http=http,
                    cal=cal,
                    cursor=load_cursor(conn, name),
                    started_at=started,
                )
                result = collector.collect(ctx, since=since, until=until or started)
                record_state(
                    conn, name, "ok", started, rows=result.total, cursor=result.cursor
                )
    except Exception as e:  # noqa: BLE001  (isolation: one collector never stops the job)
        err = f"{type(e).__name__}: {e}"
        try:
            with connect() as conn:
                record_state(conn, name, "failed", started, error=err[:2000])
        except Exception:  # noqa: BLE001
            log.exception("collector_state_write_failed", extra={"collector": name})
        return Outcome(name, "failed", error=err, run_id=run_id)

    log.info(
        "collected",
        extra={"collector": name, "rows": result.rows, "rejected": result.rejected},
    )
    return Outcome(name, "ok", result=result, run_id=run_id)


def run_all(collectors: Iterable[Collector], **kwargs: Any) -> list[Outcome]:
    """Run collectors in order; a failure in one never stops the rest."""
    return [run_collector(c, **kwargs) for c in collectors]


# ---------------------------------------------------------------- freshness
@dataclass(frozen=True)
class Freshness:
    collector: str
    critical: bool
    max_hours: float
    last_ok_at: datetime | None
    last_status: str | None
    age_hours: float | None
    stale_after: datetime | None  # the instant this feed counts as stale (None: never ran)
    now: datetime

    @property
    def stale(self) -> bool:
        return self.stale_after is None or self.now > self.stale_after


def stale_after(
    last_ok: datetime, rule: FreshnessRule, cal: TradingCalendar | None
) -> datetime:
    """When a feed whose last ok run was ``last_ok`` becomes stale.

    Session-paced feeds are due at the first session decision time after ``last_ok``; they're
    stale ``max_hours`` after that. Others are stale ``max_hours`` after ``last_ok``.
    """
    grace = timedelta(hours=rule.max_hours)
    if not rule.session_paced:
        return last_ok + grace
    if cal is None:
        raise ValueError("session-paced freshness needs a TradingCalendar")
    d = last_ok.astimezone(ET).date()
    due = cal.decision_time(d) if cal.is_session(d) else None
    if due is None or due <= last_ok:
        due = cal.decision_time(cal.next_session(d))
    return due + grace


def freshness(
    conn: psycopg.Connection,
    cfg: FreshnessCfg,
    collectors: Iterable[str],
    now: datetime,
    cal: TradingCalendar | None = None,
) -> list[Freshness]:
    names = list(collectors)
    state = {
        r[0]: (r[1], r[2])
        for r in conn.execute(
            "SELECT collector, last_ok_at, last_status FROM collector_state "
            "WHERE collector = ANY(%s)",
            (names,),
        ).fetchall()
    }
    out = []
    for n in names:
        rule = cfg.rule(n)
        last_ok, status = state.get(n, (None, None))
        age = (now - last_ok).total_seconds() / 3600 if last_ok else None
        due = stale_after(last_ok, rule, cal) if last_ok else None
        out.append(Freshness(n, rule.critical, rule.max_hours, last_ok, status, age, due, now))
    return out


def alert_stale(report: Iterable[Freshness]) -> list[str]:
    """Log stale feeds: critical ones at ERROR (critical push, coalesced), others at WARNING."""
    stale = []
    for f in report:
        if not f.stale:
            continue
        stale.append(f.collector)
        extra = {"collector": f.collector, "age_hours": f.age_hours, "max_hours": f.max_hours}
        if f.critical:
            log.error("feed_stale_critical: %s", f.collector, extra=extra)
        else:
            log.warning("feed_stale", extra=extra)
    return stale
