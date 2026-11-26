"""The collector service: one long-running APScheduler 3.x process (component doc section 7).

    QT_PROFILE=paper uv run python -m qt.orchestrator.scheduler

- timezone America/New_York; in-memory job store (collectors resume from ``collector_state``)
- ``coalesce=True``, ``max_instances=1``, ``misfire_grace_time`` from config
- a missed, erroring or overlapping job logs ERROR, which pushes a critical alert (D-016)
- ``stock_daily`` can't be a cron (close + 30 differs on early-close days), so it re-plans each
  morning for that day's session; on startup after today's decision time it catches up at once
"""

from __future__ import annotations

import logging
import signal
import sys
from datetime import UTC, date, datetime, timedelta
from typing import Any

from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MAX_INSTANCES, EVENT_JOB_MISSED
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from qt.collect.pit import parse_hhmm
from qt.common.calendar import ET, TradingCalendar
from qt.common.config import Settings
from qt.common.paths import repo_path
from qt.orchestrator.jobs import JobRunner, heartbeat

log = logging.getLogger("qt.scheduler")


def stock_daily_time(cal: TradingCalendar, d: date) -> datetime | None:
    """UTC run time of ``stock_daily`` on ET date ``d`` (decision time), or None if no session."""
    return cal.decision_time(d) if cal.is_session(d) else None


def plan_stock_daily(
    sched: BaseScheduler,
    runner: JobRunner,
    cal: TradingCalendar,
    now: datetime | None = None,
    catch_up: bool = False,
) -> datetime | None:
    """Schedule today's ``stock_daily``. Returns its UTC run time, or None (no session/passed)."""
    now = now or datetime.now(UTC)
    d = now.astimezone(ET).date()
    t = stock_daily_time(cal, d)
    if t is None:
        log.info("stock_daily_no_session", extra={"date": d.isoformat()})
        return None
    if t <= now:
        if not catch_up:
            return None
        t = now  # started late today: run now (collectors are idempotent)
    sched.add_job(
        runner.run_job, DateTrigger(t), args=["stock_daily"], id="stock_daily",
        name="stock_daily", replace_existing=True,
    )  # fmt: skip
    log.info("stock_daily_planned", extra={"at": t.isoformat()})
    return t


def _cron_et(hhmm: str, **kw: Any) -> CronTrigger:
    t = parse_hhmm(hhmm)
    return CronTrigger(hour=t.hour, minute=t.minute, timezone=ET, **kw)


def build_scheduler(
    settings: Settings,
    runner: JobRunner,
    cal: TradingCalendar,
    scheduler: BaseScheduler | None = None,
    now: datetime | None = None,
) -> BaseScheduler:
    sc = settings.config.collect.schedule
    now = now or datetime.now(UTC)
    sched = scheduler or BlockingScheduler()
    sched.configure(
        timezone=ET,
        job_defaults={
            "coalesce": True,
            "max_instances": 1,
            "misfire_grace_time": sc.misfire_grace_s,
        },
    )
    run_dir = repo_path(sc.run_dir)
    crypto = parse_hhmm(sc.crypto_utc)

    def add(func, trigger, job_id, args=(), first=None):
        kw = {"next_run_time": first} if first is not None else {}
        sched.add_job(func, trigger, args=list(args), id=job_id, name=job_id,
                      replace_existing=True, **kw)  # fmt: skip

    add(runner.run_job, IntervalTrigger(minutes=sc.news_every_min), "news_loop",
        ["news_loop"], first=now)  # fmt: skip
    add(heartbeat, IntervalTrigger(minutes=sc.heartbeat_min), "heartbeat", [run_dir], first=now)
    add(runner.check_freshness, IntervalTrigger(minutes=sc.freshness_check_min), "freshness",
        first=now + timedelta(minutes=sc.freshness_check_min))  # fmt: skip
    add(runner.run_job, CronTrigger(hour=crypto.hour, minute=crypto.minute, timezone=UTC),
        "crypto_daily", ["crypto_daily"])  # fmt: skip
    add(runner.run_job, _cron_et(sc.congress_et), "congress_daily", ["congress_daily"])
    add(runner.run_session_job, _cron_et(sc.earnings_et), "earnings_daily", ["earnings_daily"])
    add(runner.run_job, _cron_et(sc.weekly.time_et, day_of_week=sc.weekly.day), "weekly",
        ["weekly"])  # fmt: skip
    add(runner.backup, _cron_et(settings.config.backup.evening_time_et), "backup")
    add(runner.daily_summary, _cron_et(sc.summary_et), "daily_summary")
    add(plan_stock_daily, _cron_et(sc.replan_et), "stock_daily_replan", [sched, runner, cal])
    plan_stock_daily(sched, runner, cal, now=now, catch_up=True)

    sched.add_listener(_on_event, EVENT_JOB_ERROR | EVENT_JOB_MISSED | EVENT_JOB_MAX_INSTANCES)
    return sched


def _on_event(event: Any) -> None:
    job = getattr(event, "job_id", "?")
    if event.code == EVENT_JOB_ERROR:
        log.error("job_error: %s: %r", job, event.exception, exc_info=event.exception)
    elif event.code == EVENT_JOB_MISSED:
        log.error("job_missed: %s (scheduled %s)", job, event.scheduled_run_time)
    else:
        log.error("job_overlap: %s still running; this run was skipped", job)


def main(argv: list[str] | None = None) -> int:
    from qt.collect.http import HttpClient
    from qt.common import alerts, config
    from qt.common import logging as qtlog
    from qt.db import conn as dbconn

    settings = config.load()
    connect = lambda: dbconn.connect(settings)  # noqa: E731
    qtlog.setup(settings, connect=connect)
    if settings.profile != "paper":
        log.warning("scheduler_not_paper_profile", extra={"profile": settings.profile})
    cal = TradingCalendar(settings.config.calendar)
    http = HttpClient(settings.config.collect.http)
    runner = JobRunner(settings, connect, http, cal)
    sched = build_scheduler(settings, runner, cal)

    def stop(signum: int, _frame: Any) -> None:
        log.info("scheduler_stopping", extra={"signal": signum})
        sched.shutdown(wait=False)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("scheduler_started", extra={"jobs": sorted(j.id for j in sched.get_jobs())})
    try:
        sched.start()
    finally:
        http.close()
        a = alerts.get()
        if a is not None:
            a.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
