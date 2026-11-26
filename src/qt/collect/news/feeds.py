"""RSS fetch + parse (feedparser), URL canonicalization, per-feed failure tracking."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import feedparser

from qt.collect.http import HttpClient

log = logging.getLogger("qt.collect.news")


@dataclass(frozen=True)
class FeedEntry:
    source: str
    guid: str | None
    url: str  # link as published (may carry tracking params); used to fetch the page
    canonical_url: str
    url_hash: str
    title: str
    published_at: datetime | None  # the publisher's claim; never used for point-in-time
    summary: str
    tags: tuple[str, ...] = ()


def canonical_url(url: str) -> str:
    """Lowercase scheme/host, drop query, fragment, userinfo, default port and trailing slash."""
    p = urlsplit(url.strip())
    scheme = (p.scheme or "https").lower()
    host = (p.hostname or "").lower()
    default_port = {"http": 80, "https": 443}.get(scheme)
    if p.port and p.port != default_port:
        host = f"{host}:{p.port}"
    path = p.path or "/"
    if len(path) > 1:
        path = path.rstrip("/")
    return urlunsplit((scheme, host, path, "", ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()


def _published(e: Any) -> datetime | None:
    t = e.get("published_parsed") or e.get("updated_parsed")
    return datetime(*t[:6], tzinfo=UTC) if t else None  # feedparser normalizes to UTC


def parse_feed(source: str, content: bytes) -> list[FeedEntry]:
    f = feedparser.parse(content)
    if f.get("bozo") and not f.entries:
        raise ValueError(f"{source}: unparseable feed ({type(f.get('bozo_exception')).__name__})")
    out = []
    for e in f.entries:
        link = e.get("link")
        if not link:
            continue
        canon = canonical_url(link)
        out.append(
            FeedEntry(
                source=source,
                guid=e.get("id"),
                url=link,
                canonical_url=canon,
                url_hash=hashlib.sha256(canon.encode("utf-8")).hexdigest(),
                title=(e.get("title") or "").strip(),
                published_at=_published(e),
                summary=(e.get("summary") or "").strip(),
                tags=tuple(t.get("term", "") for t in e.get("tags", []) if t.get("term")),
            )
        )
    return out


def fetch_feed(http: HttpClient, source: str, url: str) -> list[FeedEntry]:
    return parse_feed(source, http.get(url).content)


def track_feed(
    cursor: dict[str, Any], source: str, ok: bool, alert_after: int, error: str = ""
) -> None:
    """Count consecutive failures per feed in the cursor and escalate at ``alert_after``.

    One failed fetch is a warning (feeds blip); ``alert_after`` in a row is an ERROR, which pushes
    a critical alert (D-016). The count keeps going, and every further ``alert_after`` failures
    alert again, so a feed that stays down keeps being reported without paging every loop.
    """
    fails: dict[str, int] = cursor.setdefault("feed_failures", {})
    if ok:
        if fails.get(source, 0) >= alert_after:
            log.info("feed_recovered", extra={"feed": source, "after_failures": fails[source]})
        fails.pop(source, None)
        return
    n = fails.get(source, 0) + 1
    fails[source] = n
    extra = {"feed": source, "consecutive": n, "error": error}
    if n % alert_after == 0:
        log.error("feed_down: %s failed %d times in a row", source, n, extra=extra)
    else:
        log.warning("feed_failed", extra=extra)
