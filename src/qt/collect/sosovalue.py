"""SoSoValue spot crypto ETF flows -> raw_metrics, source 'sosovalue' (6.10, Q10, D-032).

``GET {base_url}/etfs/summary-history?symbol=BTC&country_code=US&limit=N`` (header
``x-soso-api-key``): daily totals across all spot ETFs of one coin, newest first, trading days
only. Stored per field as a metric (``total_net_inflow``, ``total_value_traded``,
``total_net_assets``, ``cum_net_inflow``, USD), ``subject`` = ``US-BTC``, ``ts`` = trade date
00:00 ET.

- ``released_at`` = ``min(trade date + 1 day 09:00 ET, fetch time)``: flows publish the next
  morning (§3 history rule), and a row we already hold earlier than that was known when fetched.
- Settlement is T+1: the newest row can have nulls (``total_net_inflow`` etc.) until it
  settles. Null fields are not stored; the settled value is stored by a later run (the key
  includes the metric, so the other fields of that day aren't blocked).
- Non-critical (Q10). Skipped, loudly, until ``SOSOVALUE_API_KEY`` is set. The docs limit date
  ranges to the last month; ``history_limit`` asks for up to 300 rows and takes what comes back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

from qt.collect import pit
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.collect.crypto_structure import METRIC_COLS
from qt.common.calendar import ET

log = logging.getLogger("qt.collect.sosovalue")

FIELDS = ("total_net_inflow", "total_value_traded", "total_net_assets", "cum_net_inflow")


def _now() -> datetime:
    return datetime.now(UTC)


def flow_rows(subject: str, items: list[dict[str, Any]], fetched: datetime,
              pit_cfg: Any) -> list[tuple]:  # fmt: skip
    out = []
    for r in items:
        if not r.get("date"):
            continue
        d = date.fromisoformat(r["date"])
        ts = datetime.combine(d, datetime.min.time(), ET).astimezone(UTC)
        rel = min(pit.sosovalue_history(pit_cfg, d), pit.first_seen(fetched))
        for f in FIELDS:
            v = r.get(f)
            if v is None:
                continue  # not settled yet (T+1)
            out.append(("sosovalue", f, subject, ts, float(v), "usd", rel))
    return out


class SosovalueCollector:
    name = "sosovalue"
    required_secrets: tuple[str, ...] = ("sosovalue_api_key",)

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.sosovalue
        key = ctx.settings.secrets.sosovalue_api_key
        assert key is not None
        hdr = {"x-soso-api-key": key.get_secret_value()}
        now = self.clock()
        cursor = dict(ctx.cursor)
        res = CollectResult(cursor=cursor)
        limit = cfg.daily_limit if cursor.get("backfilled") else cfg.history_limit
        failed: dict[str, str] = {}
        for sym in cfg.symbols:
            subject = f"{cfg.country_code}-{sym}"
            try:
                items = ctx.http.get(
                    f"{cfg.base_url}/etfs/summary-history",
                    params={"symbol": sym, "country_code": cfg.country_code, "limit": limit},
                    headers=hdr,
                ).json()
                if isinstance(items, dict):  # error envelope instead of the documented array
                    raise RuntimeError(f"unexpected response: {str(items)[:200]}")
                rows = flow_rows(subject, items, now, ctx.settings.config.collect.pit)
                res.add("raw_metrics", insert_rows(ctx.conn, "raw_metrics", METRIC_COLS, rows))
                ctx.conn.commit()
                cursor.setdefault("first_day", {})[subject] = min(
                    (r["date"] for r in items if r.get("date")), default=None
                )
            except Exception as e:  # noqa: BLE001  (one coin failing never stops the rest)
                ctx.conn.rollback()
                failed[subject] = f"{type(e).__name__}: {e}"
                log.warning(
                    "sosovalue_failed", extra={"subject": subject, "error": failed[subject]}
                )
        if failed:
            raise RuntimeError(f"{len(failed)} SoSoValue series failed: {failed}")
        cursor["backfilled"] = True
        return res
