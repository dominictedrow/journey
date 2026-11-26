"""Logging (D-015): human-readable console + JSON lines in ``logs/qt_YYYYMMDD.jsonl`` (UTC days).

``run_id`` and ``stage`` come from contextvars set by :func:`qt.common.runs.run_context`, so every
line inside a run carries them without the caller passing anything. Use short snake_case messages
as event names and put values in ``extra``::

    log.info("collected", extra={"feed": "fred", "rows": 412})
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from qt.common import alerts
from qt.common.config import Settings
from qt.common.paths import repo_path

run_id_var: ContextVar[str | None] = ContextVar("qt_run_id", default=None)
stage_var: ContextVar[str | None] = ContextVar("qt_stage", default=None)

_STD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}
_QT_HANDLER = "_qt_handler"


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "run_id", None) is None:
            record.run_id = run_id_var.get()
        if getattr(record, "stage", None) is None:
            record.stage = stage_var.get()
        return True


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {
        k: v
        for k, v in vars(record).items()
        if k not in _STD_ATTRS and k not in ("run_id", "stage") and not k.startswith("_")
    }


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, UTC).strftime("%H:%M:%S")
        parts = [ts, f"{record.levelname:<7}"]
        if record.stage:
            parts.append(str(record.stage))
        if record.run_id:
            parts.append(str(record.run_id))
        parts.append(record.getMessage())
        parts += [f"{k}={v}" for k, v in _extras(record).items()]
        line = " ".join(parts)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        obj: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": record.run_id,
            "stage": record.stage,
            "event": msg.split(" ", 1)[0] if msg else None,
            "msg": msg,
            **_extras(record),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, default=str, ensure_ascii=False)


class DailyJsonlHandler(logging.Handler):
    """Writes ``<dir>/qt_YYYYMMDD.jsonl`` for the current UTC day; prunes files past retention."""

    def __init__(self, directory: Path, retention_days: int) -> None:
        super().__init__()
        self.directory = directory
        self.retention_days = retention_days
        self._day: str | None = None
        self._fh: TextIO | None = None
        self._io_lock = threading.Lock()
        directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, day: str) -> Path:
        return self.directory / f"qt_{day}.jsonl"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            day = datetime.fromtimestamp(record.created, UTC).strftime("%Y%m%d")
            with self._io_lock:
                if day != self._day:
                    self._rollover(day)
                assert self._fh is not None
                self._fh.write(line + "\n")
                self._fh.flush()
        except Exception:  # noqa: BLE001
            self.handleError(record)

    def _rollover(self, day: str) -> None:
        if self._fh:
            self._fh.close()
        self._day = day
        self._fh = self.path_for(day).open("a", encoding="utf-8", newline="\n")
        self.prune()

    def prune(self, today: datetime | None = None) -> None:
        cutoff = (today or datetime.now(UTC)) - timedelta(days=self.retention_days)
        for p in self.directory.glob("qt_*.jsonl"):
            try:
                d = datetime.strptime(p.stem[3:], "%Y%m%d").replace(tzinfo=UTC)
            except ValueError:
                continue
            if d < cutoff:
                p.unlink(missing_ok=True)

    def close(self) -> None:
        with self._io_lock:
            if self._fh:
                self._fh.close()
                self._fh = None
                self._day = None
        super().close()


def setup(
    settings: Settings,
    log_dir: Path | None = None,
    alerter: alerts.Alerter | None = None,
    console: bool = True,
    connect: Callable[[], Any] | None = None,
) -> logging.Logger:
    """Configure the root logger once per process. Safe to call again (replaces qt handlers)."""
    lc = settings.config.app.logging
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _QT_HANDLER, False):
            root.removeHandler(h)
            h.close()
    root.setLevel(lc.level)

    ctx = ContextFilter()
    handlers: list[logging.Handler] = []
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(ConsoleFormatter())
        handlers.append(ch)
    fh = DailyJsonlHandler(log_dir or repo_path(lc.dir), lc.retention_days)
    fh.setFormatter(JsonFormatter())
    handlers.append(fh)

    if alerter is None:
        ac = settings.config.app.alerts
        topic = settings.secrets.ntfy_topic
        alerter = alerts.Alerter(
            ac,
            topic.get_secret_value() if topic else None,
            sink=alerts.db_sink(connect) if connect else None,
        )
    alerts.install(alerter)
    handlers.append(alerts.AlertHandler(alerter))

    for h in handlers:
        h.addFilter(ctx)
        setattr(h, _QT_HANDLER, True)
        root.addHandler(h)
    # Quiet noisy libraries; their errors still reach the alert handler.
    for name in ("httpx", "httpcore", "psycopg.pool"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger("qt")
