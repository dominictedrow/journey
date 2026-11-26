"""Alpaca corporate actions -> raw_events, source 'alpaca_ca' (D-030, D-031).

Raw bars are stored unadjusted (D-030); Phase 3 builds adjusted series from these. Every action
type Alpaca returns is kept (``kind`` = ``ca_<type>``, e.g. ``ca_forward_splits``,
``ca_cash_dividends``, ``ca_name_changes``), with the vendor record as ``payload``.

- Symbols: every ticker ever in the index plus every equity/ETF asset's ticker. A reused symbol
  brings the other company's actions too; Phase 3 maps by ticker era, as for news tags.
- ``event_ts`` = the ex date (else effective, else process date) at 00:00 ET.
- ``released_at`` = ``min(fetch time, event date 00:00 ET)`` (``pit.corp_action``).
- First run: from ``corp_actions.start``; later runs: the last ``refetch_days`` (late records)
  through ``lookahead_days`` ahead (announced actions). A revised record has a new content hash,
  so it's a new row.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

from qt.collect import pit
from qt.collect.alpaca import Alpaca
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.collect.fed import EVENT_COLS
from qt.common.calendar import ET

log = logging.getLogger("qt.collect.corp_actions")

DATE_FIELDS = ("ex_date", "effective_date", "process_date", "payable_date")


def _now() -> datetime:
    return datetime.now(UTC)


def event_date(rec: dict[str, Any]) -> date | None:
    for f in DATE_FIELDS:
        if rec.get(f):
            return date.fromisoformat(rec[f])
    return None


def ca_row(kind: str, rec: dict[str, Any], fetched: datetime) -> tuple | None:
    d = event_date(rec)
    if d is None or not rec.get("id"):
        return None
    body = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    sym = rec.get("symbol") or rec.get("new_symbol") or rec.get("acquirer_symbol") or ""
    return (
        "alpaca_ca", str(rec["id"]), hashlib.sha256(body.encode()).hexdigest(), f"ca_{kind}",
        datetime.combine(d, datetime.min.time(), ET).astimezone(UTC), f"{kind} {sym}".strip(),
        None, {**rec, "type": kind}, pit.corp_action(d, fetched),
    )  # fmt: skip


class CorpActionsCollector:
    name = "corp_actions"
    required_secrets: tuple[str, ...] = ("alpaca_api_key_id", "alpaca_api_secret_key")

    def __init__(
        self,
        clock: Callable[[], datetime] = _now,
        alpaca: Callable[[RunCtx], Any] | None = None,
    ) -> None:
        self.clock = clock
        self.make_alpaca = alpaca or (lambda ctx: Alpaca.from_settings(ctx.settings, ctx.http))

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.corp_actions
        now = self.clock()
        today = now.astimezone(ET).date()
        cursor = dict(ctx.cursor)
        res = CollectResult(cursor=cursor)
        end = today + timedelta(days=cfg.lookahead_days)
        symbols = [r[0] for r in ctx.conn.execute(
            "SELECT ticker FROM index_membership UNION "
            "SELECT ticker FROM assets WHERE asset_class IN ('equity', 'etf') ORDER BY 1"
        ).fetchall()]  # fmt: skip
        if not symbols:
            raise RuntimeError("no symbols: run membership and alpaca_assets first")
        # Symbols already backfilled get a short refetch; any symbol added since (a new ETF in
        # universe.extra_etfs, a new index member) gets the full history from cfg.start. Cursors
        # from before the per-symbol list existed are treated as not backfilled (idempotent).
        done = set(cursor.get("symbols_done") or [])
        recent = today - timedelta(days=cfg.refetch_days)
        full = date.fromisoformat(cfg.start)
        batches = [
            (full, [s for s in symbols if s not in done]),
            (recent, [s for s in symbols if s in done]),
        ]
        api = self.make_alpaca(ctx)
        n = cfg.symbols_per_request
        for start, syms in batches:
            for i in range(0, len(syms), n):
                recs = api.corporate_actions(syms[i : i + n], start, end, cfg.page_limit)
                rows = [r for r in (ca_row(k, rec, now) for k, rec in recs) if r is not None]
                res.rejected += len(recs) - len(rows)
                res.add("raw_events", insert_rows(ctx.conn, "raw_events", EVENT_COLS, rows))
                ctx.conn.commit()
        cursor["backfilled"] = True
        cursor["symbols_done"] = sorted(done | set(symbols))
        cursor["symbols"] = len(symbols)
        return res
