"""Earnings calendar -> raw_events, source 'nasdaq', kind 'earnings' (component doc 6.9, D-031).

``api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD`` (unofficial, needs a browser-like UA)
lists one day's reports. Only tickers ever in the universe are kept.

- **live slice**: today through ``lookahead_days`` ahead, every run. ``released_at`` = fetch
  time. Each observation is a row keyed by ``(ticker:fiscal quarter, report date + time)``, so a
  moved date is a new row and the old one is kept (that's the point-in-time history).
- **backfill slice**: past weekdays from ``backfill_start``, ``backfill_budget`` days per run.
  ``released_at`` = report date - 7 days at 00:00 ET, ``payload.approx = true`` (§3 default; this
  assumes companies announce dates at least a week ahead, which is optimistic for a few). A
  date seen live first keeps its live row (same key).
- Weekends are skipped (Nasdaq lists none). Finnhub fallback: not built (no key; D-031).
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from qt.collect import pit
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.collect.fed import EVENT_COLS
from qt.common.calendar import ET

log = logging.getLogger("qt.collect.earnings")


def _now() -> datetime:
    return datetime.now(UTC)


def weekdays(a: date, b: date) -> list[date]:
    return [a + timedelta(days=i) for i in range((b - a).days + 1)
            if (a + timedelta(days=i)).weekday() < 5]  # fmt: skip


def earnings_rows(
    day: date,
    items: list[dict[str, Any]],
    universe: set[str],
    fetched: datetime,
    backfill: bool,
    pit_cfg: Any,
) -> list[tuple]:
    out = []
    for r in items:
        sym = (r.get("symbol") or "").strip().upper()
        if sym not in universe:
            continue
        when = r.get("time") or ""
        fq = r.get("fiscalQuarterEnding") or day.isoformat()
        if backfill:
            rel, approx = pit.earnings_backfill(pit_cfg, day)
        else:
            rel, approx = pit.first_seen(fetched), False
        out.append((
            "nasdaq", f"{sym}:{fq}", hashlib.sha256(f"{day}|{when}".encode()).hexdigest(),
            "earnings", datetime.combine(day, datetime.min.time(), ET).astimezone(UTC),
            r.get("name"), None,
            {**r, "symbol": sym, "report_date": day.isoformat(), "approx": approx,
             "backfill": backfill},
            rel,
        ))  # fmt: skip
    return out


class EarningsCollector:
    name = "earnings"
    required_secrets: tuple[str, ...] = ()

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def _day(self, ctx: RunCtx, d: date) -> list[dict[str, Any]]:
        cfg = ctx.settings.config.collect.earnings
        j = ctx.http.get(cfg.url, params={"date": d.isoformat()},
                         headers={"User-Agent": cfg.user_agent, "Accept": "application/json"},
                         ).json()  # fmt: skip
        return ((j or {}).get("data") or {}).get("rows") or []

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.earnings
        pit_cfg = ctx.settings.config.collect.pit
        now = self.clock()
        today = now.astimezone(ET).date()
        cursor = dict(ctx.cursor)
        res = CollectResult(cursor=cursor)
        universe = {r[0] for r in ctx.conn.execute(
            "SELECT ticker FROM index_membership UNION "
            "SELECT ticker FROM assets WHERE asset_class = 'equity'").fetchall()}  # fmt: skip
        if not universe:
            raise RuntimeError("no universe tickers: run membership first")

        def run(days: list[date], backfill: bool) -> None:
            for d in days:
                rows = earnings_rows(d, self._day(ctx, d), universe, now, backfill, pit_cfg)
                res.add("raw_events", insert_rows(ctx.conn, "raw_events", EVENT_COLS, rows))
                ctx.conn.commit()
                if backfill:
                    cursor["backfill_next"] = (d + timedelta(days=1)).isoformat()

        run(weekdays(today, today + timedelta(days=cfg.lookahead_days)), backfill=False)
        nxt = date.fromisoformat(cursor.get("backfill_next", cfg.backfill_start))
        todo = weekdays(nxt, today - timedelta(days=1))[: cfg.backfill_budget]
        try:
            run(todo, backfill=True)
        except Exception as e:  # noqa: BLE001  keep the progress made; the next run resumes
            ctx.conn.rollback()
            cursor["backfill_error"] = f"{type(e).__name__}: {e}"[:300]
            log.warning("earnings_backfill_failed", extra={"error": cursor["backfill_error"]})
        else:
            cursor.pop("backfill_error", None)
        if not todo:  # caught up (or only a weekend left): resume from today next time
            cursor["backfill_next"] = max(nxt, today).isoformat()
        return res
