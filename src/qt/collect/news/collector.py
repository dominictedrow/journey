"""News collector: the 5 RSS feeds every 15 min (component doc 6.7, D-008, D-022).

Per new entry (canonical URL not in ``article_seen``):
fetch page -> archive raw HTML -> trafilatura -> word count -> dedup -> tag -> insert
``articles`` + ``article_seen`` together. ``released_at`` is first-seen (our insert time).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from qt.collect import pit
from qt.collect.base import CollectResult, InsertStats, RunCtx, insert_rows
from qt.collect.http import HttpError
from qt.collect.news.dedup import (
    find_near_dup,
    find_title_dup,
    simhash64,
    title_hash,
    to_signed,
)
from qt.collect.news.extract import Extracted, HtmlArchive, extract
from qt.collect.news.feeds import FeedEntry, fetch_feed, track_feed
from qt.collect.news.tag import Tagger, load_aliases
from qt.common.paths import repo_path

log = logging.getLogger("qt.collect.news")

ARTICLE_COLS = [
    "url", "url_hash", "simhash", "source", "published_at", "released_at", "title", "body",
    "tickers", "word_count", "extract_status", "guid", "title_hash", "near_dup_of",
]  # fmt: skip


def _now() -> datetime:
    return datetime.now(UTC)


class NewsCollector:
    name = "news"
    required_secrets: tuple[str, ...] = ()

    def __init__(self, clock: Callable[[], datetime] = _now, archive: HtmlArchive | None = None):
        self.clock = clock
        self.archive = archive

    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult:
        cfg = ctx.settings.config.collect.news
        archive = self.archive or HtmlArchive(repo_path(cfg.archive_dir))
        tagger = Tagger(load_aliases(ctx.conn), ctx.settings.config.aliases.finance_cues)
        cursor = dict(ctx.cursor)
        res = CollectResult(cursor=cursor)
        failed: dict[str, str] = {}

        for source, url in cfg.feeds.items():
            try:
                entries = fetch_feed(ctx.http, source, url)
            except Exception as e:  # noqa: BLE001  (one feed down never stops the others)
                failed[source] = f"{type(e).__name__}: {e}"
                track_feed(cursor, source, False, cfg.feed_fail_alert_after, failed[source])
                continue
            track_feed(cursor, source, True, cfg.feed_fail_alert_after)
            new = self._unseen(ctx, entries)[: cfg.max_new_per_feed]
            for e in new:
                res.add("articles", self._ingest(ctx, e, archive, tagger))
            ctx.conn.commit()  # each feed's articles are durable before the next feed

        if len(failed) == len(cfg.feeds):
            raise RuntimeError(f"all {len(failed)} news feeds failed: {failed}")
        return res

    # ------------------------------------------------------------ internals
    @staticmethod
    def _unseen(ctx: RunCtx, entries: list[FeedEntry]) -> list[FeedEntry]:
        uniq: dict[str, FeedEntry] = {}
        for e in entries:
            uniq.setdefault(e.url_hash, e)
        if not uniq:
            return []
        seen = {
            r[0]
            for r in ctx.conn.execute(
                "SELECT url_hash FROM article_seen WHERE url_hash = ANY(%s)", (list(uniq),)
            ).fetchall()
        }
        return [e for h, e in uniq.items() if h not in seen]

    def _fetch_page(self, ctx: RunCtx, e: FeedEntry) -> tuple[str | None, int | None]:
        try:
            r = ctx.http.get(e.url)
            return r.text, r.status_code
        except HttpError as err:
            log.warning("article_fetch_failed", extra={"feed": e.source, "status": err.status})
            return None, err.status
        except Exception as err:  # noqa: BLE001
            log.warning("article_fetch_failed", extra={"feed": e.source, "error": repr(err)})
            return None, None

    def _ingest(
        self, ctx: RunCtx, e: FeedEntry, archive: HtmlArchive, tagger: Tagger
    ) -> InsertStats:
        cfg = ctx.settings.config.collect.news
        html, status = self._fetch_page(ctx, e)
        fetched_at = self.clock()
        archive.write(e.source, fetched_at, e.url, status, html)
        ex: Extracted = extract(html, e.url, cfg.min_words)

        text = ex.body or f"{e.title}\n{e.summary}"
        sh = simhash64(text, cfg.simhash.shingle)
        sh_signed = to_signed(sh) if sh is not None else None
        th = title_hash(e.title)
        released = pit.first_seen(self.clock())
        dup = find_title_dup(ctx.conn, th) or find_near_dup(
            ctx.conn, sh_signed, released, cfg.simhash.window_days, cfg.simhash.max_hamming
        )
        tickers = tagger.tag(f"{e.title}\n{ex.body or e.summary}", released.date())
        row = (
            e.canonical_url, e.url_hash, sh_signed, e.source, e.published_at, released, e.title,
            ex.body, tickers, ex.word_count, ex.status, e.guid, th, dup,
        )  # fmt: skip
        stats = insert_rows(ctx.conn, "articles", ARTICLE_COLS, [row])
        if stats.inserted:
            insert_rows(
                ctx.conn,
                "article_seen",
                ["url_hash", "simhash", "first_seen"],
                [(e.url_hash, sh_signed, released)],
            )
        return stats
