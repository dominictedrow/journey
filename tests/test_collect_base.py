"""Collector run wrapper: idempotency, untimed-row rejection, missing keys, failure isolation."""

import dataclasses
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from qt.collect.base import (
    CollectResult,
    alert_stale,
    freshness,
    insert_rows,
    run_all,
    run_collector,
)
from qt.collect.http import HttpClient
from qt.common import alerts
from qt.common import logging as qtlog
from qt.common.calendar import TradingCalendar
from qt.db.conn import connect

pytestmark = pytest.mark.db

T0 = datetime(2026, 11, 24, 21, 0, tzinfo=UTC)
COLS = ["source", "metric", "subject", "ts", "value", "released_at"]


class FakeAlerter(alerts.Alerter):
    def __init__(self, cfg):
        super().__init__(cfg, topic="t", http_post=lambda *a, **k: None, os_notify=lambda *a: True)
        self.sent = []

    def _deliver(self, severity, event, title, body):
        self.sent.append((severity, title, body))
        return True


class MetricCollector:
    """Inserts a fixed set of raw_metrics rows; stands in for a real collector."""

    required_secrets: tuple[str, ...] = ()

    def __init__(self, name="unit_ok", rows=None, fail=False):
        self.name = name
        self.rows = rows if rows is not None else [
            ("unit", "m", "", T0 + timedelta(days=i), float(i), T0 + timedelta(days=i + 1))
            for i in range(3)
        ]
        self.fail = fail
        self.seen_cursor = None

    def collect(self, ctx, *, since, until):
        self.seen_cursor = dict(ctx.cursor)
        if self.fail:
            raise RuntimeError("boom")
        res = CollectResult(cursor={"last_ts": self.rows[-1][3].isoformat()})
        res.add("raw_metrics", insert_rows(ctx.conn, "raw_metrics", COLS, self.rows))
        return res


@pytest.fixture
def env(settings, db, tmp_path):
    fake = FakeAlerter(settings.config.app.alerts)
    qtlog.setup(settings, log_dir=tmp_path, alerter=fake, console=False)
    http = HttpClient(settings.config.collect.http, transport=httpx.MockTransport(lambda r: None))
    kw = dict(
        settings=settings,
        connect=lambda: connect(settings, "primary", db),
        http=http,
        cal=TradingCalendar(settings.config.calendar),
        clock=lambda: T0,
    )
    yield kw, fake
    http.close()


def _state(kw, name):
    with kw["connect"]() as c:
        return c.execute(
            "SELECT last_status, last_ok_at, last_error, cursor, rows_last_run "
            "FROM collector_state WHERE collector = %s",
            (name,),
        ).fetchone()


def test_idempotent_collect(env):
    kw, _ = env
    c = MetricCollector()
    first = run_collector(c, **kw)
    second = run_collector(c, **kw)
    assert first.status == second.status == "ok"
    assert first.result.rows == {"raw_metrics": 3}
    assert second.result.rows == {"raw_metrics": 0}
    assert c.seen_cursor == {"last_ts": (T0 + timedelta(days=2)).isoformat()}  # cursor resumed
    st, ok_at, err, cur, n = _state(kw, "unit_ok")
    assert (st, ok_at, err, n) == ("ok", T0, None, 0)


def test_untimed_rows_rejected(env):
    kw, _ = env
    rows = [
        ("unit", "m", "", T0, 1.0, T0),
        ("unit", "m", "", T0 + timedelta(days=1), 2.0, None),
        ("unit", "m", "", T0 + timedelta(days=2), 3.0, datetime(2026, 11, 25)),  # naive
    ]
    out = run_collector(MetricCollector("unit_rej", rows), **kw)
    assert out.result.rows == {"raw_metrics": 1} and out.result.rejected == 2


def test_missing_key_skips(env):
    kw, _ = env
    c = MetricCollector("unit_key")
    c.required_secrets = ("congress_api_key",)
    settings = dataclasses.replace(
        kw["settings"],
        secrets=kw["settings"].secrets.model_copy(update={"congress_api_key": None}),
    )
    out = run_collector(c, **{**kw, "settings": settings})
    assert out.status == "skipped" and "CONGRESS_API_KEY" in out.error
    st, ok_at, err, _, _ = _state(kw, "unit_key")
    assert st == "skipped" and ok_at is None and "CONGRESS_API_KEY" in err


def test_collector_failure_isolated(env):
    kw, fake = env
    outs = run_all([MetricCollector("unit_fail", fail=True), MetricCollector("unit_after")], **kw)
    assert [o.status for o in outs] == ["failed", "ok"]
    assert "boom" in outs[0].error
    st, ok_at, err, _, _ = _state(kw, "unit_fail")
    assert st == "failed" and ok_at is None and "boom" in err
    crit = [s for s in fake.sent if s[0] == "critical"]
    assert len(crit) == 1 and "collect_unit_fail" in crit[0][1]


def test_failure_keeps_last_ok_and_cursor(env):
    kw, _ = env
    run_collector(MetricCollector("unit_keep"), **kw)
    later = T0 + timedelta(hours=5)
    run_collector(MetricCollector("unit_keep", fail=True), **{**kw, "clock": lambda: later})
    st, ok_at, _, cur, _ = _state(kw, "unit_keep")
    assert st == "failed" and ok_at == T0 and "last_ts" in cur


def test_bad_collector_name(env):
    kw, _ = env
    with pytest.raises(ValueError):
        run_collector(MetricCollector("this_name_is_too_long"), **kw)


def test_freshness_and_stale_alert(env):
    kw, fake = env
    run_collector(MetricCollector("news"), **kw)  # news: critical, max 2h in collect.yaml
    cfg = kw["settings"].config.collect.freshness
    with kw["connect"]() as c:
        fresh = freshness(c, cfg, ["news", "never_ran"], T0 + timedelta(hours=1))
        stale = freshness(c, cfg, ["news", "never_ran"], T0 + timedelta(hours=3))
    assert [f.stale for f in fresh] == [False, True]
    assert stale[0].stale and stale[0].critical and stale[0].age_hours == pytest.approx(3)
    assert alert_stale(stale) == ["news", "never_ran"]
    crit = [s for s in fake.sent if s[0] == "critical"]
    assert len(crit) == 1 and "news" in crit[0][1]  # never_ran is non-critical -> warn only
