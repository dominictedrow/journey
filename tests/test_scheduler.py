"""Scheduler plan: triggers, stock_daily on early closes/holidays, heartbeat gaps."""

import json
from datetime import UTC, date, datetime, timedelta

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from qt.common.calendar import ET, TradingCalendar
from qt.orchestrator import jobs
from qt.orchestrator.jobs import JobRunner, heartbeat, heartbeat_gaps
from qt.orchestrator.scheduler import build_scheduler, plan_stock_daily, stock_daily_time


@pytest.fixture(scope="module")
def cal(settings):
    return TradingCalendar(settings.config.calendar)


def utc(*a):
    return datetime(*a, tzinfo=UTC)


def test_stock_daily_time(cal):
    assert stock_daily_time(cal, date(2026, 11, 24)) == utc(2026, 11, 24, 21, 30)  # 16:30 ET
    assert stock_daily_time(cal, date(2026, 11, 27)) == utc(2026, 11, 27, 18, 30)  # 13:30 ET
    assert stock_daily_time(cal, date(2026, 11, 26)) is None  # Thanksgiving
    assert stock_daily_time(cal, date(2026, 11, 28)) is None  # Saturday
    assert stock_daily_time(cal, date(2026, 11, 29)) is None  # Sunday


def _runner(settings, cal):
    return JobRunner(settings, connect=lambda: None, http=None, cal=cal)


def test_build_scheduler_jobs(settings, cal):
    now = utc(2026, 11, 24, 15, 0)  # 10:00 ET on a session
    sched = build_scheduler(settings, _runner(settings, cal), cal, BackgroundScheduler(), now)
    js = {j.id: j for j in sched.get_jobs()}
    assert set(js) == {
        "news_loop", "heartbeat", "freshness", "crypto_daily", "congress_daily",
        "earnings_daily", "weekly", "backup", "daily_summary", "stock_daily_replan",
        "stock_daily",
    }  # fmt: skip
    sc = settings.config.collect.schedule
    assert isinstance(js["news_loop"].trigger, IntervalTrigger)
    assert js["news_loop"].trigger.interval == timedelta(minutes=sc.news_every_min)
    assert js["news_loop"].next_run_time == now  # runs immediately at startup
    assert isinstance(js["stock_daily"].trigger, DateTrigger)
    assert js["stock_daily"].trigger.run_date == utc(2026, 11, 24, 21, 30)

    def fire(job_id, after):
        return js[job_id].trigger.get_next_fire_time(None, after)

    assert isinstance(js["crypto_daily"].trigger, CronTrigger)
    assert fire("crypto_daily", now) == utc(2026, 11, 25, 0, 15)
    assert fire("congress_daily", now).astimezone(ET).strftime("%H:%M") == "06:00"
    assert fire("backup", now).astimezone(ET).strftime("%H:%M") == "21:00"
    assert fire("daily_summary", now).astimezone(ET).strftime("%H:%M") == "21:30"
    w = fire("weekly", now).astimezone(ET)
    assert (w.strftime("%a %H:%M"), w.date()) == ("Sun 18:00", date(2026, 11, 29))
    # jobs pick these up when the scheduler starts (pending jobs don't carry them yet)
    assert sched._job_defaults == {
        "coalesce": True, "max_instances": 1, "misfire_grace_time": sc.misfire_grace_s,
    }  # fmt: skip


def test_plan_early_close_holiday_and_catch_up(settings, cal):
    r = _runner(settings, cal)
    s = BackgroundScheduler(timezone=ET)
    assert plan_stock_daily(s, r, cal, now=utc(2026, 11, 27, 11, 0)) == utc(2026, 11, 27, 18, 30)
    assert plan_stock_daily(s, r, cal, now=utc(2026, 11, 26, 11, 0)) is None  # holiday
    late = utc(2026, 11, 24, 23, 0)  # 18:00 ET, after the 16:30 decision time
    assert plan_stock_daily(s, r, cal, now=late) is None
    assert plan_stock_daily(s, r, cal, now=late, catch_up=True) == late


def test_session_job_skips_holiday(settings, cal, monkeypatch):
    r = _runner(settings, cal)
    called = []
    monkeypatch.setattr(r, "run_job", lambda job: called.append(job) or ["ran"])
    assert r.run_session_job("earnings_daily", now=utc(2026, 11, 26, 12)) == []
    assert r.run_session_job("earnings_daily", now=utc(2026, 11, 25, 12)) == ["ran"]


def test_known_collectors_fit_stage_limit():
    names = jobs.known_collectors()
    assert names == [
        "news", "fed_rss", "alpaca_assets", "bars_equity", "fred", "hy_archive",
        "crypto_assets", "bars_crypto", "membership", "aliases",
    ]  # fmt: skip
    assert all(len(f"collect_{n}") <= 21 for n in names)


def test_heartbeat_and_gaps(tmp_path):
    t = utc(2026, 11, 25, 15, 0)
    for m in (0, 5, 10, 40, 45):  # 30-minute gap between 10 and 40
        heartbeat(tmp_path, t + timedelta(minutes=m))
    rec = json.loads((tmp_path / "heartbeat.json").read_text())
    assert rec["ts"] == (t + timedelta(minutes=45)).isoformat()
    assert heartbeat_gaps(tmp_path, date(2026, 11, 25), 15) == [
        ("2026-11-25 15:10", "2026-11-25 15:40", 30)
    ]
    assert heartbeat_gaps(tmp_path, date(2026, 11, 26), 15) == []


def test_session_paced_freshness_ignores_weekends_and_holidays(settings, cal):
    from qt.collect.base import stale_after
    from qt.common.config import FreshnessRule

    rule = FreshnessRule(critical=True, max_hours=6, session_paced=True)
    wed = utc(2026, 11, 25, 21, 35)  # ran 16:35 ET Wed; Thu is Thanksgiving
    # next due: Fri 11-27 13:30 ET (early close + 30) -> stale 6 h later
    assert stale_after(wed, rule, cal) == utc(2026, 11, 27, 18, 30) + timedelta(hours=6)
    fri = utc(2026, 11, 27, 18, 31)
    assert stale_after(fri, rule, cal) == utc(2026, 11, 30, 21, 30) + timedelta(hours=6)  # Mon
    morning = utc(2026, 11, 24, 14, 0)  # last ok before today's decision time: due today
    assert stale_after(morning, rule, cal) == utc(2026, 11, 24, 21, 30) + timedelta(hours=6)
    plain = FreshnessRule(critical=True, max_hours=2)
    assert stale_after(wed, plain, None) == wed + timedelta(hours=2)
