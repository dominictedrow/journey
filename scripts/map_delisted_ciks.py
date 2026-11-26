"""Close the survivorship gaps of former S&P members (component doc 03 section 9, D-043).

    QT_PROFILE=paper uv run python scripts/map_delisted_ciks.py            # map, write, apply
    QT_PROFILE=paper uv run python scripts/map_delisted_ciks.py --dry-run  # reports only

Finds CIKs (name match + filing check), names and sectors (historical Wikipedia GICS, else SIC)
and bars for delisted names; writes configs/cik_overrides.yaml, configs/sector_overrides.yaml,
reports/survivorship_YYYYMMDD.md and reports/cik_review_YYYYMMDD.md; then loads the overrides
into `assets`. Afterwards run the EDGAR backfill for the new CIKs:

    QT_PROFILE=paper uv run python -m qt.collect.backfill --source edgar_facts
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from qt.collect import survivorship
    from qt.collect.aliases import edgar_ua
    from qt.collect.http import HttpClient
    from qt.common import config
    from qt.common import logging as qtlog
    from qt.common.calendar import ET, TradingCalendar
    from qt.common.paths import repo_path
    from qt.db import conn as dbconn

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="write the reports only")
    args = ap.parse_args(argv)

    settings = config.load()
    connect = lambda: dbconn.connect(settings)  # noqa: E731
    qtlog.setup(settings, connect=connect)
    http = HttpClient(settings.config.collect.http)
    cal = TradingCalendar(settings.config.calendar)
    now = datetime.now(UTC)
    try:
        with connect() as conn:
            outs = survivorship.run(conn, settings.config, http, cal, edgar_ua(settings), now,
                                    write_bars=not args.dry_run)  # fmt: skip
            reports = repo_path(Path(settings.config.app.reports.daily_dir).parent)
            rep, review = survivorship.write_reports(
                outs, settings.config.features.survivorship, reports, now.astimezone(ET).date()
            )
            print(f"report: {rep}\nreview: {review}")
            if args.dry_run:
                conn.rollback()
                return 0
            nc, ns = survivorship.save_overrides(outs, settings.config)
            print(f"overrides: {nc} CIK entries, {ns} sector entries")
            fresh = config.load(profile=settings.profile)  # re-read the files just written
            print(f"applied: {survivorship.apply_overrides(conn, fresh.config)}")
    finally:
        http.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
