from datetime import datetime, timedelta, timezone

import pytest

from qt.common import alerts
from qt.common import logging as qtlog
from qt.common.runs import RUN_ID_RE, new_run_id, run_context
from qt.db.conn import connect


def test_run_id_format():
    now = datetime(2026, 11, 25, 16, 30, tzinfo=timezone(timedelta(hours=-5)))  # 16:30 ET
    rid = new_run_id("collect", now)
    assert RUN_ID_RE.match(rid)
    assert rid.startswith("20261125T2130Z-collect-")  # converted to UTC
    assert len({new_run_id("collect", now) for _ in range(20)}) > 1  # suffix breaks ties


@pytest.mark.parametrize("bad", ["Collect", "c", "has-dash", "x" * 30])
def test_run_id_bad_stage(bad):
    with pytest.raises(ValueError):
        new_run_id(bad)


def test_run_id_naive_rejected():
    with pytest.raises(ValueError):
        new_run_id("collect", datetime(2026, 11, 25, 12, 0))


class FakeAlerter(alerts.Alerter):
    def __init__(self, cfg):
        super().__init__(cfg, topic="t", http_post=lambda *a, **k: None, os_notify=lambda *a: True)
        self.sent = []

    def _deliver(self, severity, event, title, body):
        self.sent.append((severity, title, body))
        return True


@pytest.mark.db
def test_run_context(settings, db, tmp_path):
    fake = FakeAlerter(settings.config.app.alerts)
    qtlog.setup(settings, log_dir=tmp_path, alerter=fake, console=False)
    conn = lambda: connect(settings, "primary", db)  # noqa: E731

    with run_context("unit_ok", settings, conn) as r:
        ok_id = r.run_id
    with pytest.raises(ZeroDivisionError):
        with run_context("unit_fail", settings, conn) as r:
            fail_id = r.run_id
            raise ZeroDivisionError("boom")

    with conn() as c:
        rows = dict(
            c.execute(
                "SELECT run_id, (status, error, finished_at IS NOT NULL, config_hash) FROM runs"
            ).fetchall()
        )
    assert rows[ok_id][0] == "ok" and rows[ok_id][2]
    st, err, finished, h = rows[fail_id]
    assert st == "failed" and "ZeroDivisionError" in err and finished
    assert h == settings.config_hash
    crit = [s for s in fake.sent if s[0] == "critical"]
    assert len(crit) == 1 and "unit_fail" in crit[0][1] and fail_id in crit[0][2]
