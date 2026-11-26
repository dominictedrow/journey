"""Daily Markdown summary (D-015). Phase 1 ships the scaffold; Phase 2+ fill in the sections."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from qt.common.config import Settings
from qt.common.paths import repo_path

PENDING = "_Not built yet (filled in by a later phase)._"


def _day_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime(d.year, d.month, d.day, tzinfo=UTC)
    return start, start + timedelta(days=1)


def _table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    if not rows:
        return "_None._"
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def write_daily_summary(
    d: date,
    settings: Settings,
    connect: Callable[[], Any] | None = None,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or repo_path(settings.config.app.reports.daily_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = _day_bounds(d)

    runs: list[tuple[Any, ...]] = []
    alerts_rows: list[tuple[Any, ...]] = []
    backups: list[tuple[Any, ...]] = []
    if connect is not None:
        with connect() as conn:
            runs = conn.execute(
                "SELECT stage, status, count(*) FROM runs "
                "WHERE started_at >= %s AND started_at < %s GROUP BY 1, 2 ORDER BY 1, 2",
                (start, end),
            ).fetchall()
            alerts_rows = conn.execute(
                "SELECT to_char(ts AT TIME ZONE 'UTC', 'HH24:MI'), severity, title, delivered "
                "FROM alerts_log WHERE ts >= %s AND ts < %s ORDER BY ts",
                (start, end),
            ).fetchall()
            backups = conn.execute(
                "SELECT to_char(ts AT TIME ZONE 'UTC', 'HH24:MI'), status, dump_file, dump_bytes, "
                "mismatches::text FROM backup_log WHERE ts >= %s AND ts < %s ORDER BY ts",
                (start, end),
            ).fetchall()

    body = f"""# QT daily summary: {d.isoformat()} (UTC day)

Profile `{settings.profile}` · config `{settings.config_hash[:12]}`

## Feed freshness
{PENDING}

## Rows collected
{PENDING}

## Runs
{_table(["stage", "status", "count"], runs)}

## Alerts
{_table(["time (UTC)", "severity", "title", "delivered"], alerts_rows)}

## Backups
{_table(["time (UTC)", "status", "dump", "bytes", "mismatches"], backups)}

## Predictions
{PENDING}

## Orders, fills and P&L
{PENDING}

## Breakers
{PENDING}
"""
    path = out_dir / f"{d.isoformat()}.md"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path
