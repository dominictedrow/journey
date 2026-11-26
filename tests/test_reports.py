from datetime import date

import pytest

from qt.common.reports import write_daily_summary
from qt.db.conn import connect


def test_summary_scaffold_without_db(settings, tmp_path):
    p = write_daily_summary(date(2026, 11, 25), settings, out_dir=tmp_path)
    text = p.read_text()
    assert p.name == "2026-11-25.md"
    for h in ("Feed freshness", "Runs", "Alerts", "Backups", "Orders, fills and P&L", "Breakers"):
        assert f"## {h}" in text


@pytest.mark.db
def test_summary_reads_alerts(settings, db, tmp_path):
    with connect(settings, "primary", db) as c:
        c.execute(
            "INSERT INTO alerts_log (ts, severity, event, title, delivered) "
            "VALUES ('2026-11-25 15:00+00', 'critical', 'e', 'feed down', true)"
        )
    p = write_daily_summary(
        date(2026, 11, 25),
        settings,
        connect=lambda: connect(settings, "primary", db),
        out_dir=tmp_path,
    )
    assert "feed down" in p.read_text()
