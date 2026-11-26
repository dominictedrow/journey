"""Scheduler job functions (component doc section 7). Thin: they call collectors under run_context.

``JOB_COLLECTORS`` maps each scheduled job to the collectors it runs. Build steps 3-5 append to
it as each collector lands; a job with no collectors yet logs ``job_no_collectors`` and returns.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from qt.collect.aliases import AliasesCollector
from qt.collect.assets import AlpacaAssetsCollector, CryptoAssetsCollector
from qt.collect.bars import BarsCryptoCollector, BarsEquityCollector
from qt.collect.base import Collector, Connect, Outcome, alert_stale, freshness, run_all
from qt.collect.congress import CongressCollector
from qt.collect.corp_actions import CorpActionsCollector
from qt.collect.crypto_derivs import CryptoOiCollector, FundingCollector
from qt.collect.crypto_structure import CoingeckoCollector, DefillamaCollector
from qt.collect.earnings import EarningsCollector
from qt.collect.edgar import EdgarFactsCollector
from qt.collect.fed import FedRssCollector
from qt.collect.fred import FredCollector, HyArchiveCollector
from qt.collect.http import HttpClient
from qt.collect.membership import MembershipCollector
from qt.collect.news.collector import NewsCollector
from qt.collect.yfin import YfinanceCollector
from qt.common.calendar import ET, TradingCalendar
from qt.common.config import Settings
from qt.common.paths import repo_path
from qt.common.reports import write_daily_summary

log = logging.getLogger("qt.orchestrator")

JOB_COLLECTORS: dict[str, Callable[[], list[Collector]]] = {
    "news_loop": lambda: [NewsCollector(), FedRssCollector()],
    "stock_daily": lambda: [
        AlpacaAssetsCollector(), BarsEquityCollector(), CorpActionsCollector(),
        YfinanceCollector(), FredCollector(), HyArchiveCollector(),
    ],
    "crypto_daily": lambda: [
        CryptoAssetsCollector(), BarsCryptoCollector(), FundingCollector(), CryptoOiCollector(),
        CoingeckoCollector(), DefillamaCollector(),
    ],
    "congress_daily": lambda: [CongressCollector()],
    "earnings_daily": lambda: [EarningsCollector()],
    # aliases fills assets.cik, which edgar_facts reads
    "weekly": lambda: [MembershipCollector(), AliasesCollector(), EdgarFactsCollector()],
}  # fmt: skip


def known_collectors() -> list[str]:
    return [c.name for make in JOB_COLLECTORS.values() for c in make()]


@dataclass
class JobRunner:
    settings: Settings
    connect: Connect
    http: HttpClient
    cal: TradingCalendar
    _stale: set[str] = field(default_factory=set)  # feeds already alerted as stale

    def run_job(self, job: str) -> list[Outcome]:
        collectors = JOB_COLLECTORS[job]()
        if not collectors:
            log.info("job_no_collectors", extra={"job": job})
            return []
        outs = run_all(
            collectors, settings=self.settings, connect=self.connect, http=self.http, cal=self.cal
        )
        log.info("job_done", extra={"job": job, "status": {o.collector: o.status for o in outs}})
        return outs

    def run_session_job(self, job: str, now: datetime | None = None) -> list[Outcome]:
        """Run ``job`` only on XNYS sessions (earnings_daily)."""
        today = (now or datetime.now(UTC)).astimezone(ET).date()
        if not self.cal.is_session(today):
            log.info("job_skipped_non_session", extra={"job": job, "date": today.isoformat()})
            return []
        return self.run_job(job)

    # ------------------------------------------------------------ ops jobs
    def check_freshness(self, now: datetime | None = None) -> list[str]:
        """Alert each feed once when it turns stale; it re-arms after the feed recovers.

        A collector that has never run (no ``collector_state`` row) isn't alerted here: until
        the scheduler has run it once, "never" is expected, and the daily summary shows it.
        """
        now = now or datetime.now(UTC)
        with self.connect() as conn:
            report = freshness(
                conn, self.settings.config.collect.freshness, known_collectors(), now, self.cal
            )
        stale_now = {f.collector for f in report if f.stale and f.last_status is not None}
        newly = [f for f in report if f.collector in stale_now - self._stale]
        for name in self._stale - stale_now:
            log.info("feed_fresh_again", extra={"collector": name})
        self._stale = stale_now
        return alert_stale(newly)

    def daily_summary(self, now: datetime | None = None) -> Path:
        now = now or datetime.now(UTC)
        d = now.astimezone(ET).date()
        sched = self.settings.config.collect.schedule
        with self.connect() as conn:
            report = freshness(
                conn, self.settings.config.collect.freshness, known_collectors(), now, self.cal
            )
        rows = [
            (
                f.collector,
                "yes" if f.critical else "no",
                f"{f.last_ok_at:%Y-%m-%d %H:%M}" if f.last_ok_at else "never",
                f"{f.age_hours:.1f}" if f.age_hours is not None else "",
                f"{f.max_hours:g}",
                f.last_status or "",
                "**STALE**" if f.stale else "",
            )
            for f in report
        ]
        gaps = heartbeat_gaps(repo_path(sched.run_dir), d, sched.heartbeat_gap_min)
        return write_daily_summary(
            d, self.settings, connect=self.connect, freshness=rows, heartbeat_gaps=gaps
        )

    def backup(self) -> bool:
        from qt.db.backup import backup_job

        return backup_job(self.settings, self.connect)


# ---------------------------------------------------------------- heartbeat
def _hb_log(run_dir: Path, d: date) -> Path:
    return run_dir / f"heartbeat_{d:%Y%m%d}.jsonl"


def heartbeat(run_dir: Path, now: datetime | None = None) -> Path:
    """Write ``heartbeat.json`` (atomic replace) and append to the ET-day heartbeat log."""
    now = now or datetime.now(UTC)
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"ts": now.astimezone(UTC).isoformat(), "pid": os.getpid()}
    p = run_dir / "heartbeat.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec), encoding="utf-8")
    tmp.replace(p)
    with _hb_log(run_dir, now.astimezone(ET).date()).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return p


def heartbeat_gaps(run_dir: Path, d: date, max_gap_min: int) -> list[tuple[str, str, int]]:
    """Gaps longer than ``max_gap_min`` between heartbeats on ET day ``d`` (incl. the day before's
    last beat, so a gap across midnight is seen)."""
    beats: list[datetime] = []
    for day in (d - timedelta(days=1), d):
        p = _hb_log(run_dir, day)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    beats.append(datetime.fromisoformat(json.loads(line)["ts"]))
    beats.sort()
    day_start = datetime(d.year, d.month, d.day, tzinfo=ET).astimezone(UTC)
    out = []
    for a, b in zip(beats, beats[1:], strict=False):
        if b < day_start:
            continue
        mins = (b - a).total_seconds() / 60
        if mins > max_gap_min:
            out.append((f"{a:%Y-%m-%d %H:%M}", f"{b:%Y-%m-%d %H:%M}", round(mins)))
    return out
