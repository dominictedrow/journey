"""Push alerts via ntfy (D-016).

Every failure alerts (the root-logger ``AlertHandler`` at ERROR+ routes here) and every trade event
alerts (Phase 6 calls :func:`alert` with ``info``). Identical ``(event, title)`` alerts inside the
coalesce window fold into one "xN" follow-up, so a crash loop can't send hundreds of
pushes but no failure goes unreported. Delivery never raises into the caller; if the push
fails it falls back to an OS notification and is recorded with ``delivered=false``.
"""

from __future__ import annotations

import atexit
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import httpx

from qt.common.config import AlertMuteCfg, AlertsCfg

Severity = Literal["info", "warn", "critical"]
SEVERITIES: tuple[Severity, ...] = ("info", "warn", "critical")
_TAGS = {"info": "information_source", "warn": "warning", "critical": "rotating_light"}

# Logs from inside the alert path go here at WARNING max, so they can never re-trigger an alert.
_internal = logging.getLogger("qt.alerts.internal")

Sink = Callable[[Severity, str, str, str, bool], None]


@dataclass
class _Pending:
    severity: Severity
    title: str
    body: str
    first_sent: float
    count: int = 0  # suppressed repeats since first_sent


@dataclass
class Alerter:
    cfg: AlertsCfg
    topic: str | None
    http_post: Callable[..., httpx.Response] | None = None
    os_notify: Callable[[str, str], bool] | None = None
    sink: Sink | None = None  # e.g. alerts_log writer
    clock: Callable[[], float] = time.monotonic
    _seen: dict[tuple[str, str], _Pending] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ------------------------------------------------------------ public
    def alert(self, severity: Severity, event: str, title: str, body: str = "") -> bool:
        """Send (or coalesce) one alert. Returns True if a push was delivered. Never raises."""
        try:
            return self._alert(severity, event, title, body)
        except Exception as e:  # noqa: BLE001  (alerting must never break the caller)
            _internal.warning("alert path crashed: %r", e)
            return False

    def flush(self) -> None:
        """Send "xN" follow-ups for every coalesced key, regardless of window. Call at exit."""
        with self._lock:
            keys = [k for k, p in self._seen.items() if p.count]
            for k in keys:
                self._send_followup(k)
            self._seen.clear()

    # ------------------------------------------------------------ internals
    def _alert(self, severity: Severity, event: str, title: str, body: str) -> bool:
        if severity not in SEVERITIES:
            severity = "critical"
        now = self.clock()
        key = (event, title)
        with self._lock:
            self._flush_expired(now, skip=key)
            pending = self._seen.get(key)
            if pending and now - pending.first_sent < self.cfg.coalesce_window_s:
                pending.count += 1
                return False
            if pending and pending.count:
                self._send_followup(key)
            self._seen[key] = _Pending(severity, title, body, first_sent=now)
        return self._deliver(severity, event, title, body)

    def _flush_expired(self, now: float, skip: tuple[str, str]) -> None:
        for k, p in list(self._seen.items()):
            if k != skip and now - p.first_sent >= self.cfg.coalesce_window_s:
                if p.count:
                    self._send_followup(k)
                del self._seen[k]

    def _send_followup(self, key: tuple[str, str]) -> None:
        p = self._seen[key]
        mins = max(1, round(self.cfg.coalesce_window_s / 60))
        self._deliver(p.severity, key[0], f"{p.title} (x{p.count} more in last {mins} min)", p.body)
        p.count = 0

    def _deliver(self, severity: Severity, event: str, title: str, body: str) -> bool:
        delivered = False
        if self.cfg.enabled and self.topic:
            delivered = self._push(severity, title, body)
        if not delivered and self.cfg.enabled and self.cfg.os_fallback:
            (self.os_notify or os_notify)(title, body)
        if self.sink:
            try:
                self.sink(severity, event, title, body, delivered)
            except Exception as e:  # noqa: BLE001
                _internal.warning("alerts_log write failed: %r", e)
        return delivered

    def _push(self, severity: Severity, title: str, body: str) -> bool:
        payload = {
            "topic": self.topic,
            "title": title[:250],
            "message": (body or title)[:4000],
            "priority": getattr(self.cfg.priority, severity),
            "tags": [_TAGS[severity]],
        }
        try:
            post = self.http_post or httpx.post
            r = post(self.cfg.server.rstrip("/"), json=payload, timeout=self.cfg.timeout_s)
            if r.status_code >= 300:
                _internal.warning("ntfy returned HTTP %s", r.status_code)
                return False
            return True
        except Exception as e:  # noqa: BLE001
            _internal.warning("ntfy push failed: %r", e)
            return False


def os_notify(title: str, body: str) -> bool:
    """Best-effort local notification. Text is passed as argv/env, never spliced into a script."""
    try:
        if sys.platform == "darwin":
            script = [
                "on run argv",
                "display notification (item 2 of argv) with title (item 1 of argv)",
                "end run",
            ]
            args = ["osascript"] + [a for line in script for a in ("-e", line)]
            subprocess.run([*args, title, body[:200]], timeout=5, check=False, capture_output=True)
            return True
        if sys.platform == "win32":
            ps = (
                "Add-Type -AssemblyName System.Windows.Forms;"
                "$n=New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon=[System.Drawing.SystemIcons]::Warning;$n.Visible=$true;"
                "$n.ShowBalloonTip(10000,$env:QT_N_TITLE,$env:QT_N_BODY,'Warning');"
                "Start-Sleep -s 5;$n.Dispose()"
            )
            env = {**os.environ, "QT_N_TITLE": title, "QT_N_BODY": body[:200]}
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                env=env,
                timeout=15,
                check=False,
                capture_output=True,
            )
            return True
    except Exception as e:  # noqa: BLE001
        _internal.warning("OS notification failed: %r", e)
    return False


def is_muted(record: logging.LogRecord, mute: tuple[AlertMuteCfg, ...]) -> bool:
    """True when ``record`` matches a configured mute rule (its logger or a child, and the text)."""
    if not mute:
        return False
    msg = record.getMessage()
    return any(
        (record.name == m.logger or record.name.startswith(m.logger + ".")) and m.contains in msg
        for m in mute
    )


class AlertHandler(logging.Handler):
    """Root-logger handler: every ERROR/CRITICAL record becomes a critical push (D-016).

    Records matching ``alerts.mute`` (D-046) are skipped here; the file log still keeps them.
    """

    def __init__(self, alerter: Alerter, mute: tuple[AlertMuteCfg, ...] = ()) -> None:
        super().__init__(level=logging.ERROR)
        self.alerter = alerter
        self.mute = mute
        self._guard = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("qt.alerts") or getattr(self._guard, "busy", False):
            return
        if is_muted(record, self.mute):
            return
        self._guard.busy = True
        try:
            msg = record.getMessage()
            stage = getattr(record, "stage", None) or record.name
            first = msg.splitlines()[0] if msg else record.levelname
            title = f"QT failure: {stage}: {first}"[:120]
            body = msg
            if record.exc_info:
                body += "\n\n" + "".join(
                    logging.Formatter().formatException(record.exc_info).splitlines(True)[-6:]
                )
            run_id = getattr(record, "run_id", None)
            if run_id:
                body += f"\n\nrun_id={run_id}"
            self.alerter.alert("critical", f"{record.name}:{first[:120]}", title, body)
        finally:
            self._guard.busy = False


def db_sink(connect: Callable[[], object]) -> Sink:
    """Sink that records every alert (sent or not) in ``alerts_log``."""

    def sink(severity: Severity, event: str, title: str, body: str, delivered: bool) -> None:
        from qt.common.logging import run_id_var  # local import: logging imports this module

        with connect() as conn:  # type: ignore[attr-defined]
            conn.execute(
                "INSERT INTO alerts_log (severity, event, title, body, delivered, run_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (severity, event[:500], title, body, delivered, run_id_var.get()),
            )

    return sink


# ---------------------------------------------------------------- module-level singleton
_alerter: Alerter | None = None


def install(alerter: Alerter | None) -> None:
    global _alerter
    if _alerter is not None:
        _alerter.flush()
    _alerter = alerter
    if alerter is not None:
        atexit.register(alerter.flush)


def get() -> Alerter | None:
    return _alerter


def alert(severity: Severity, event: str, title: str, body: str = "") -> bool:
    """Explicit alert, e.g. trades (info) or a successful backup (info). Never raises."""
    return _alerter.alert(severity, event, title, body) if _alerter else False
