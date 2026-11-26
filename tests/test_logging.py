import json
import logging
from datetime import UTC, datetime, timedelta

from qt.common import logging as qtlog
from qt.common.logging import DailyJsonlHandler, run_id_var, stage_var


def test_jsonl_lines_carry_context(settings, tmp_path):
    qtlog.setup(settings, log_dir=tmp_path, console=False)
    log = logging.getLogger("qt.test")
    t1, t2 = run_id_var.set("20261125T2130Z-collect-a3f9"), stage_var.set("collect")
    try:
        log.info("collected", extra={"feed": "fred", "rows": 412})
    finally:
        run_id_var.reset(t1)
        stage_var.reset(t2)
    log.warning("outside_run")
    files = list(tmp_path.glob("qt_*.jsonl"))
    assert len(files) == 1
    lines = [json.loads(x) for x in files[0].read_text().splitlines()]
    first = next(x for x in lines if x["event"] == "collected")
    assert first["run_id"] == "20261125T2130Z-collect-a3f9" and first["stage"] == "collect"
    assert first["feed"] == "fred" and first["rows"] == 412 and first["ts"].endswith("+00:00")
    outside = next(x for x in lines if x["event"] == "outside_run")
    assert outside["run_id"] is None


def test_retention_prunes_old_files(tmp_path):
    h = DailyJsonlHandler(tmp_path, retention_days=30)
    now = datetime(2026, 11, 25, tzinfo=UTC)
    old = h.path_for((now - timedelta(days=40)).strftime("%Y%m%d"))
    recent = h.path_for((now - timedelta(days=5)).strftime("%Y%m%d"))
    old.write_text("{}\n")
    recent.write_text("{}\n")
    h.prune(today=now)
    assert not old.exists() and recent.exists()
