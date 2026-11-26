"""Fed RSS: FOMC statements, minutes, projections, press releases, Board speeches -> raw_events.

Polled in the 15-min news loop so first-seen times are accurate (component doc 6.7).

- **Live** rows: ``released_at`` = first-seen time.
- **First run** (empty cursor): the items already in the feeds are a backfill, so they use the
  item's official published time (section 3, "Fed item (backfill)"). Otherwise months-old
  statements would look like today's news.

An item appearing in several feeds (press_monetary is a subset of press_all) is stored once,
with the kind from the first feed listed in config.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime

from qt.collect import pit
from qt.collect.base import CollectResult, RunCtx, insert_rows
from qt.collect.news.feeds import FeedEntry, fetch_feed, track_feed
from qt.common.config import FedCfg

log = logging.getLogger("qt.collect.fed")

EVENT_COLS = [
    "source", "external_id", "content_hash", "kind", "event_ts", "title", "url", "payload",
    "released_at",
]  # fmt: skip


def _now() -> datetime:
    return datetime.now(UTC)


def classify(cfg: FedCfg, feed: str, title: str) -> str:
    for rule in cfg.kind_rules:
        if re.search(rule.pattern, title, re.IGNORECASE):
            return rule.kind
    return cfg.feed_default_kind.get(feed, "press")


def speaker(cfg: FedCfg, title: str) -> tuple[str | None, str]:
    """Speeches feed titles are ``"<Surname>, <speech title>"``."""
    name = title.split(",", 1)[0].strip() if "," in title else None
    if name is None:
        return None, cfg.default_role
    role = cfg.speaker_roles.get(name)
    if role is None:
        log.warning("fed_speaker_unmapped", extra={"speaker": name})
        return name, cfg.default_role
    return name, role


def content_hash(e: FeedEntry) -> str:
    pub = e.published_at.isoformat() if e.published_at else ""
    return hashlib.sha256(f"{e.title}\x1f{e.canonical_url}\x1f{pub}".encode()).hexdigest()


class FedRssCollector:
    name = "fed_rss"
    required_secrets: tuple[str, ...] = ()

    def __init__(self, clock: Callable[[], datetime] = _now) -> None:
        self.clock = clock

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.fed
        cursor = dict(ctx.cursor)
        backfill = not cursor.get("initialized", False)
        res = CollectResult(cursor=cursor)
        rows, done, failed = [], set(), {}

        for feed, url in cfg.feeds.items():
            try:
                entries = fetch_feed(ctx.http, feed, url)
            except Exception as e:  # noqa: BLE001
                failed[feed] = f"{type(e).__name__}: {e}"
                track_feed(cursor, feed, False, cfg.feed_fail_alert_after, failed[feed])
                continue
            track_feed(cursor, feed, True, cfg.feed_fail_alert_after)
            for e in entries:
                if e.canonical_url in done:
                    continue
                done.add(e.canonical_url)
                row = self._row(cfg, feed, e, backfill)
                if row is None:
                    res.rejected += 1
                else:
                    rows.append(row)

        if len(failed) == len(cfg.feeds):
            raise RuntimeError(f"all {len(failed)} Fed feeds failed: {failed}")
        res.add("raw_events", insert_rows(ctx.conn, "raw_events", EVENT_COLS, rows))
        cursor["initialized"] = True
        return res

    def _row(self, cfg: FedCfg, feed: str, e: FeedEntry, backfill: bool) -> tuple | None:
        kind = classify(cfg, feed, e.title)
        payload: dict = {
            "feed": feed,
            "guid": e.guid,
            "published_at": e.published_at.isoformat() if e.published_at else None,
            "summary": e.summary,
            "tags": list(e.tags),
            "backfill": backfill,
        }
        if kind == "speech":
            payload["speaker"], payload["role"] = speaker(cfg, e.title)
        try:
            released = (
                pit.fed_item_backfill(e.published_at) if backfill else pit.first_seen(self.clock())
            )
        except pit.UntimeableRow:
            log.warning("fed_item_untimed", extra={"feed": feed, "url": e.canonical_url})
            return None
        return (
            "fed", e.canonical_url, content_hash(e), kind, e.published_at, e.title,
            e.canonical_url, payload, released,
        )  # fmt: skip
