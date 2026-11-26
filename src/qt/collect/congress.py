"""Congress.gov bill actions -> raw_events, source 'congress' (component doc 6.8, Q8, D-031).

Which bills: listed per congress (``/bill/{congress}``) or by update time (``/bill?fromDateTime``);
each list item has the title, introduction date and latest action. Actions are fetched only
for *candidates*: a tagged title, or a latest action after the introduction date (a bill that
was only introduced and referred has no event to record). A candidate is kept if any of its
actions touches a configured committee, or it has a tag; then **every** action is stored, with
``payload.status`` from ``congress.status_rules`` and ``payload.is_event`` for the event statuses.
``payload.tags``: ``debt_ceiling`` / ``cr_shutdown`` (title or action text).

Two slices per run, one code path:

- **live**: bills updated since the last run; ``released_at`` = first-seen (fetch time).
- **backfill**: congresses from ``first_congress`` on, ``backfill_budget`` actions requests per
  run, resumed from the cursor until done. ``released_at`` = 23:59:59 ET on the action date,
  ``payload.backfill = true``. Actions dated on/after the live start are left to the live slice,
  so the backfill rule never stamps something we could only have seen live.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, time
from typing import Any

from qt.collect import pit
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.collect.fed import EVENT_COLS
from qt.collect.http import HttpError
from qt.common.calendar import ET
from qt.common.config import CongressCfg

log = logging.getLogger("qt.collect.congress")


def _now() -> datetime:
    return datetime.now(UTC)


def current_congress(d: date) -> int:
    return (d.year - 1789) // 2 + 1


def bill_id(b: dict[str, Any]) -> str:
    return f"{b['congress']}-{b['type'].upper()}-{b['number']}"


def tags_for(cfg: CongressCfg, *texts: str) -> list[str]:
    blob = "\n".join(t for t in texts if t)
    return sorted(k for k, p in cfg.tags.items() if re.search(p, blob, re.IGNORECASE))


def is_candidate(cfg: CongressCfg, b: dict[str, Any]) -> bool:
    if tags_for(cfg, b.get("title", "")):
        return True
    intro, latest = b.get("introducedDate"), (b.get("latestAction") or {}).get("actionDate")
    return intro is None or latest is None or latest > intro


def status_of(cfg: CongressCfg, a: dict[str, Any]) -> str | None:
    for r in cfg.status_rules:
        if (r.type is None or r.type == a.get("type")) and re.search(r.pattern, a.get("text", "")):
            return r.status
    return None


def action_rows(
    cfg: CongressCfg,
    pit_cfg: Any,
    bill: dict[str, Any],
    actions: list[dict[str, Any]],
    fetched: datetime,
    backfill: bool,
    live_from: date | None,
) -> list[tuple]:
    """Rows for a bill's actions; [] if it touches none of the committees and has no tag."""
    committees = set(cfg.committees)
    touched = {c.get("systemCode") for a in actions for c in a.get("committees") or []}
    btags = tags_for(cfg, bill.get("title", ""), *(a.get("text", "") for a in actions))
    if not (touched & committees) and not btags:
        return []
    bid = bill_id(bill)
    rows = []
    for a in actions:
        d = date.fromisoformat(a["actionDate"])
        if backfill and live_from is not None and d >= live_from:
            continue
        t = time.fromisoformat(a["actionTime"]) if a.get("actionTime") else time(0)
        st = status_of(cfg, a)
        body = json.dumps(a, sort_keys=True, separators=(",", ":"))
        src = (a.get("sourceSystem") or {}).get("code", "")
        ext = f"{bid}:{a['actionDate']}:{a.get('actionCode') or a.get('type', '')}:{src}:" \
              f"{hashlib.sha1(a.get('text', '').encode()).hexdigest()[:10]}"  # fmt: skip
        rel = pit.congress_backfill(pit_cfg, d) if backfill else pit.first_seen(fetched)
        rows.append((
            "congress", ext, hashlib.sha256(body.encode()).hexdigest(), "congress_action",
            datetime.combine(d, t, ET).astimezone(UTC), bill.get("title"), bill.get("url"),
            {"bill": bid, "action_type": a.get("type"), "text": a.get("text"),
             "code": a.get("actionCode"), "committees": sorted(
                 c.get("systemCode") for c in a.get("committees") or [] if c.get("systemCode")),
             "status": st, "is_event": st in cfg.event_statuses, "tags": btags,
             "backfill": backfill},
            rel,
        ))  # fmt: skip
    return rows


class CongressCollector:
    name = "congress"
    required_secrets: tuple[str, ...] = ("congress_api_key",)

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def _get(self, ctx: RunCtx, path: str, **params: Any) -> dict[str, Any]:
        cfg = ctx.settings.config.collect.congress
        key = ctx.settings.secrets.congress_api_key
        assert key is not None
        q = {"api_key": key.get_secret_value(), "format": "json", **params}
        return ctx.http.get(f"{cfg.base_url}/{path}", params=q).json()

    def _pages(self, ctx: RunCtx, path: str, field: str, offset: int = 0,
               **params: Any) -> Iterator[tuple[int, list[dict[str, Any]]]]:  # fmt: skip
        lim = ctx.settings.config.collect.congress.page_limit
        while True:
            j = self._get(ctx, path, limit=lim, offset=offset, **params)
            items = j.get(field) or []
            yield offset, items
            offset += len(items)
            if not items or offset >= int((j.get("pagination") or {}).get("count", 0)):
                return

    def _actions(self, ctx: RunCtx, b: dict[str, Any]) -> list[dict[str, Any]]:
        path = f"bill/{b['congress']}/{b['type'].lower()}/{b['number']}/actions"
        return [a for _, page in self._pages(ctx, path, "actions") for a in page]

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.congress
        pit_cfg = ctx.settings.config.collect.pit
        now = self.clock()
        cursor = dict(ctx.cursor)
        res = CollectResult(cursor=cursor)
        first_run = "live_since" not in cursor
        live_since = now if first_run else datetime.fromisoformat(cursor["live_since"])
        live_from = live_since.astimezone(ET).date()

        def store(b: dict[str, Any], backfill: bool) -> None:
            try:
                actions = self._actions(ctx, b)
            except HttpError as e:
                if e.status != 404:
                    raise
                log.warning("congress_bill_missing", extra={"bill": bill_id(b)})
                return  # a withdrawn/renumbered bill must not wedge the backfill on its page
            rows = action_rows(cfg, pit_cfg, b, actions, now, backfill, live_from)
            res.add("raw_events", insert_rows(ctx.conn, "raw_events", EVENT_COLS, rows))

        # live slice: bills updated since the last run
        seen = 0
        if not first_run:
            for _, page in self._pages(
                ctx, "bill", "bills", sort="updateDate+asc",
                fromDateTime=live_since.strftime("%Y-%m-%dT%H:%M:%SZ"),
                toDateTime=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ):  # fmt: skip
                for b in page:
                    if is_candidate(cfg, b):
                        store(b, backfill=False)
                        seen += 1
                ctx.conn.commit()
        cursor["live_since"] = now.isoformat()
        cursor["live_bills"] = seen

        # backfill slice
        bf = cursor.setdefault("backfill", {"congress": cfg.first_congress, "offset": 0})
        budget = cfg.backfill_budget
        last = current_congress(now.astimezone(ET).date())
        try:
            while not bf.get("done") and budget > 0:
                if bf["congress"] > last:
                    bf["done"] = True
                    break
                finished = True
                for off, page in self._pages(ctx, f"bill/{bf['congress']}", "bills", bf["offset"]):
                    cands = [b for b in page if is_candidate(cfg, b)]
                    # stop before a page that would overrun the budget, but always take at
                    # least one page per run, or a page bigger than the budget would wedge it
                    if len(cands) > budget and budget < cfg.backfill_budget:
                        bf["offset"] = off  # resume at this page next run
                        finished = False
                        budget = 0
                        break
                    for b in cands:
                        store(b, backfill=True)
                    budget -= len(cands)
                    bf["offset"] = off + len(page)
                    ctx.conn.commit()
                if finished:
                    bf["congress"] += 1
                    bf["offset"] = 0
        except Exception as e:  # noqa: BLE001  keep the progress made; the next run resumes
            ctx.conn.rollback()
            bf["last_error"] = f"{type(e).__name__}: {e}"[:300]
            log.warning("congress_backfill_failed", extra={"error": bf["last_error"]})
        else:
            bf.pop("last_error", None)
        log.info("congress_progress", extra={"backfill": dict(bf), "live_bills": seen})
        return res
