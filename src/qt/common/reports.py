"""Daily Markdown summary (D-015). Phase 1 ships the scaffold; Phase 2+ fill in the sections."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from qt.common.config import Settings
from qt.common.paths import repo_path

PENDING = "_Not built yet (filled in by a later phase)._"

# Raw tables whose daily inserts are counted under "Rows collected" (all carry ingested_at).
RAW_TABLES = (
    "raw_prices", "raw_macro", "raw_fundamentals", "articles", "index_membership",
    "raw_events", "raw_metrics", "raw_crypto_derivs",
)  # fmt: skip


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
    freshness: list[tuple[Any, ...]] | None = None,
    heartbeat_gaps: list[tuple[Any, ...]] | None = None,
) -> Path:
    """Write ``reports/daily/<d>.md``.

    ``freshness`` rows: (collector, critical, last ok UTC, age h, max h, status, STALE?).
    ``heartbeat_gaps`` rows: (from UTC, to UTC, minutes). Both are computed by the scheduler.
    """
    out_dir = out_dir or repo_path(settings.config.app.reports.daily_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = _day_bounds(d)

    runs: list[tuple[Any, ...]] = []
    alerts_rows: list[tuple[Any, ...]] = []
    backups: list[tuple[Any, ...]] = []
    collected: list[tuple[Any, ...]] = []
    identity: list[tuple[Any, ...]] | None = None
    feats: list[tuple[Any, ...]] | None = None
    feats_asof = None
    if connect is not None:
        from qt.collect import identity_check  # collect builds on common; import late

        with connect() as conn:
            identity = identity_check.suspects(conn, settings.config.collect)
            for t in RAW_TABLES:
                n = conn.execute(
                    f"SELECT count(*) FROM {t} WHERE ingested_at >= %s AND ingested_at < %s",
                    (start, end),
                ).fetchone()[0]
                if n:
                    collected.append((t, f"{n:,}"))
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
            # Features: the newest decision time on or before the end of this day
            feats_asof = conn.execute(
                "SELECT max(asof) FROM features WHERE asof < %s AND asof > %s",
                (end, end - timedelta(days=4)),
            ).fetchone()[0]
            if feats_asof is not None:
                feats = [
                    (g, f"{n:,}", f"{a:,}", f"{nan:.1%}")
                    for g, n, a, nan in conn.execute(
                        "SELECT coalesce(r.feature_group, '?'), count(*), "
                        "count(DISTINCT f.asset_id), avg((f.value IS NULL)::int) "
                        "FROM features f LEFT JOIN feature_registry r ON r.feature_name = "
                        "f.feature_name AND r.version = f.feature_version WHERE f.asof = %s "
                        "GROUP BY 1 ORDER BY 1",
                        (feats_asof,),
                    ).fetchall()
                ]
            backups = conn.execute(
                "SELECT to_char(ts AT TIME ZONE 'UTC', 'HH24:MI'), status, dump_file, dump_bytes, "
                "mismatches::text FROM backup_log WHERE ts >= %s AND ts < %s ORDER BY ts",
                (start, end),
            ).fetchall()

    fresh_md = (
        _table(["collector", "critical", "last ok (UTC)", "age h", "max h", "status", "stale"],
               freshness)
        if freshness is not None
        else PENDING
    )
    collected_md = _table(["table", "rows inserted"], collected) if connect else PENDING
    identity_md = (
        ("None flagged." if not identity else
         "Members whose bars look like another company's (D-035); add an identity override:\n\n"
         + _table(["ticker", "security", "median $M/day", "bars"], identity))
        if identity is not None
        else PENDING
    )
    feats_md = (
        (f"Decision time {feats_asof:%Y-%m-%d %H:%M} UTC.\n\n"
         + _table(["group", "values", "assets", "NaN"], feats)) if feats
        else ("No feature rows yet." if connect else PENDING)
    )
    hb_md = (
        ("No gaps." if not heartbeat_gaps else
         _table(["from (UTC)", "to (UTC)", "minutes"], heartbeat_gaps))
        if heartbeat_gaps is not None
        else PENDING
    )
    body = f"""# QT daily summary: {d.isoformat()} (UTC day)

Profile `{settings.profile}` · config `{settings.config_hash[:12]}`

## Feed freshness
{fresh_md}

## Rows collected
{collected_md}

## Identity check
{identity_md}

## Features
{feats_md}

## Scheduler heartbeat
{hb_md}

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
