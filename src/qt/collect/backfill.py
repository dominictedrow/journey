"""Backfill = the scheduled collectors run by hand (component doc 5, section 13 exit gate item 2).

Every collector resumes from what's stored, so a backfill is idempotent and resumable, and it
goes through the same code (and point-in-time rules) as the daily runs.

    QT_PROFILE=paper uv run python -m qt.collect.backfill --source membership --source bars_equity
    QT_PROFILE=paper uv run python -m qt.collect.backfill --all
    QT_PROFILE=paper uv run python -m qt.collect.backfill --report   # reports/backfill_YYYYMMDD.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from qt.collect.base import Collector, run_collector
from qt.common.calendar import ET

# Order matters: membership creates the member assets that aliases and bars_equity then use.
ORDER = [
    "membership", "alpaca_assets", "crypto_assets", "aliases", "bars_equity", "bars_crypto",
    "corp_actions", "yfinance", "fred", "hy_archive", "edgar_facts", "funding", "crypto_oi",
    "coingecko", "defillama", "earnings", "congress",
]  # fmt: skip


def registry() -> dict[str, Collector]:
    from qt.orchestrator.jobs import JOB_COLLECTORS

    return {c.name: c for make in JOB_COLLECTORS.values() for c in make()}


REPORT_QUERIES = {
    "raw_prices": (
        "SELECT p.source || ' / ' || a.asset_class, min(p.ts)::date, max(p.ts)::date, count(*), "
        "count(DISTINCT p.asset_id) FROM raw_prices p JOIN assets a USING (asset_id) "
        "GROUP BY 1 ORDER BY 1",
        "assets",
    ),
    "raw_macro": (
        "SELECT source, min(period), max(period), count(*), count(DISTINCT series_id) "
        "FROM raw_macro GROUP BY 1 ORDER BY 1",
        "series",
    ),
    "index_membership": (
        "SELECT source, min(snapshot_at)::date, max(snapshot_at)::date, count(*), "
        "count(DISTINCT snapshot_at) FROM index_membership GROUP BY 1 ORDER BY 1",
        "snapshots",
    ),
    "raw_metrics": (
        "SELECT source || ' / ' || metric, min(ts)::date, max(ts)::date, count(*), "
        "count(DISTINCT subject) FROM raw_metrics GROUP BY 1 ORDER BY 1",
        "subjects",
    ),
    "raw_crypto_derivs": (
        "SELECT venue || ' / ' || kind, min(ts)::date, max(ts)::date, count(*), "
        "count(DISTINCT symbol) FROM raw_crypto_derivs GROUP BY 1 ORDER BY 1",
        "symbols",
    ),
    "raw_fundamentals": (
        "SELECT split_part(metric, ':', 1), min(released_at)::date, max(released_at)::date, "
        "count(*), count(DISTINCT asset_id) FROM raw_fundamentals GROUP BY 1 ORDER BY 1",
        "companies",
    ),
    "raw_events": (
        "SELECT source || ' / ' || kind, min(released_at)::date, max(released_at)::date, "
        "count(*), count(DISTINCT external_id) FROM raw_events GROUP BY 1 ORDER BY 1",
        "items",
    ),
    "articles": (
        "SELECT source, min(released_at)::date, max(released_at)::date, count(*), "
        "count(*) FILTER (WHERE extract_status = 'ok') FROM articles GROUP BY 1 ORDER BY 1",
        "extracted ok",
    ),
}


def write_report(conn: psycopg.Connection, out_dir: Path, now: datetime) -> Path:
    lines = [f"# Backfill report {now.astimezone(ET):%Y-%m-%d %H:%M} ET", ""]
    for table, (q, extra) in REPORT_QUERIES.items():
        lines += [f"## {table}", "", f"| source | first | last | rows | {extra} |",
                  "|---|---|---|---:|---:|"]  # fmt: skip
        rows = conn.execute(q).fetchall()
        lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:,} | {r[4]:,} |" for r in rows]
        lines += ["" if rows else "_no rows_", ""]
    a = conn.execute(
        "SELECT asset_class, count(*), count(*) FILTER (WHERE listed_until IS NOT NULL), "
        "count(*) FILTER (WHERE alpaca_tradable) FROM assets GROUP BY 1 ORDER BY 1"
    ).fetchall()
    lines += ["## assets", "", "| class | assets | delisted | tradable |", "|---|---:|---:|---:|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in a]
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"backfill_{now.astimezone(ET):%Y%m%d}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main(argv: list[str] | None = None) -> int:
    from qt.collect.http import HttpClient
    from qt.common import config
    from qt.common import logging as qtlog
    from qt.common.calendar import TradingCalendar
    from qt.common.paths import repo_path
    from qt.db import conn as dbconn

    ap = argparse.ArgumentParser(description="Run collectors by hand (idempotent, resumable).")
    ap.add_argument("--source", action="append", default=[], help="collector name (repeatable)")
    ap.add_argument("--all", action="store_true", help=f"run in order: {', '.join(ORDER)}")
    ap.add_argument("--report", action="store_true", help="write reports/backfill_YYYYMMDD.md")
    args = ap.parse_args(argv)

    reg = registry()
    names = ORDER if args.all else args.source
    unknown = [n for n in names if n not in reg]
    if unknown:
        ap.error(f"unknown collector(s) {unknown}; known: {sorted(reg)}")
    if not names and not args.report:
        ap.error("nothing to do: give --source, --all or --report")

    settings = config.load()
    connect = lambda: dbconn.connect(settings)  # noqa: E731
    qtlog.setup(settings, connect=connect)
    http = HttpClient(settings.config.collect.http)
    cal = TradingCalendar(settings.config.calendar)
    status = 0
    try:
        for n in names:
            out = run_collector(reg[n], settings=settings, connect=connect, http=http, cal=cal)
            rows = out.result.rows if out.result else {}
            print(f"{n:14s} {out.status:8s} {rows} {out.error or ''}")
            status |= out.status == "failed"
        if args.report:
            with connect() as c:
                out_dir = repo_path(Path(settings.config.app.reports.daily_dir).parent)  # reports/
                p = write_report(c, out_dir, datetime.now(UTC))
            print(f"report: {p}")
    finally:
        http.close()
    return status


if __name__ == "__main__":
    sys.exit(main())
