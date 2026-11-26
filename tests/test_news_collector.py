"""News + Fed RSS collectors end to end against mocked feeds (no network)."""

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from conftest import FIXTURES, FakeAlerter, Router, with_collect
from qt.collect.base import run_collector
from qt.collect.fed import FedRssCollector, classify, speaker
from qt.collect.http import HttpClient
from qt.collect.news.collector import NewsCollector
from qt.collect.news.extract import HtmlArchive
from qt.common import logging as qtlog
from qt.common.calendar import TradingCalendar
from qt.db.conn import connect

N = FIXTURES / "news"
T0 = datetime(2026, 11, 25, 15, 0, tzinfo=UTC)

NEWS_FEEDS = {"a": "https://news-a.example/rss", "b": "https://news-b.example/feed"}
FED_FEEDS = {
    "press_monetary": "https://fed.example/feeds/press_monetary.xml",
    "speeches": "https://fed.example/feeds/speeches.xml",
    "press_all": "https://fed.example/feeds/press_all.xml",
}


def routes():
    return {
        "https://news-a.example/rss": (N / "feed_a.xml").read_bytes(),
        "https://news-b.example/feed": (N / "feed_b.xml").read_bytes(),
        "https://news-a.example/markets/rates-hold": (N / "article_rates.html").read_text(),
        "https://news-a.example/tech/chipmaker-record": (N / "article_chip.html").read_text(),
        "https://news-b.example/2026/11/20/rates-hold-steady/": (
            N / "article_rates.html"
        ).read_text(),
        "https://news-b.example/2026/11/24/wrap": (N / "article_short.html").read_text(),
        "https://fed.example/feeds/press_monetary.xml": (N / "fed_monetary.xml").read_bytes(),
        "https://fed.example/feeds/speeches.xml": (N / "fed_speeches.xml").read_bytes(),
        "https://fed.example/feeds/press_all.xml": (N / "fed_all.xml").read_bytes(),
    }


@pytest.fixture
def env(settings, db, tmp_path):
    s = with_collect(settings, news={"feeds": NEWS_FEEDS}, fed={"feeds": FED_FEEDS})
    fake = FakeAlerter(s.config.app.alerts)
    qtlog.setup(s, log_dir=tmp_path / "logs", alerter=fake, console=False)
    router = Router(routes())
    http = HttpClient(
        s.config.collect.http, transport=httpx.MockTransport(router), sleep=lambda _: None
    )
    kw = dict(
        settings=s,
        connect=lambda: connect(s, "primary", db),
        http=http,
        cal=TradingCalendar(s.config.calendar),
        clock=lambda: T0,
    )
    yield kw, router, fake, tmp_path
    http.close()


def _articles(kw):
    with kw["connect"]() as c:
        return c.execute(
            "SELECT article_id, url, source, published_at, released_at, extract_status, "
            "word_count, near_dup_of, simhash IS NOT NULL FROM articles ORDER BY article_id"
        ).fetchall()


# ---------------------------------------------------------------- news
@pytest.mark.db
def test_news_collect_end_to_end(env):
    kw, router, _, tmp = env
    clock = iter(T0 + timedelta(seconds=i) for i in range(100))
    c = NewsCollector(clock=lambda: next(clock), archive=HtmlArchive(tmp / "arc"))
    out = run_collector(c, **kw)
    assert out.status == "ok" and out.result.rows == {"articles": 4}

    rows = _articles(kw)
    by_url = {r[1]: r for r in rows}
    rates_a = by_url["https://news-a.example/markets/rates-hold"]  # canonical: no utm params
    rates_b = by_url["https://news-b.example/2026/11/20/rates-hold-steady"]
    wrap = by_url["https://news-b.example/2026/11/24/wrap"]
    assert rates_a[5] == "ok" and rates_a[6] >= 150 and rates_a[8]
    assert rates_b[7] == rates_a[0]  # stored, flagged as dup of the earlier article
    assert wrap[5] == "short"  # stored, not dropped
    for r in rows:  # first-seen, never the publisher's (backdated) time
        assert r[4] >= T0 and r[3] < r[4]
    with kw["connect"]() as cx:
        assert cx.execute("SELECT count(*) FROM article_seen").fetchone()[0] == 4
    assert len(list((tmp / "arc").rglob("*.jsonl.gz"))) == 2  # one per source per day


@pytest.mark.db
def test_news_idempotent(env):
    kw, router, _, tmp = env
    c = NewsCollector(archive=HtmlArchive(tmp / "arc"))
    run_collector(c, **kw)
    pages_before = len([x for x in router.calls if "rss" not in x and "feed" not in x])
    out = run_collector(c, **kw)
    assert out.status == "ok" and out.result.total == 0
    pages_after = len([x for x in router.calls if "rss" not in x and "feed" not in x])
    assert pages_after == pages_before  # seen URLs are not fetched again


@pytest.mark.db
def test_news_one_feed_down(env):
    kw, router, fake, tmp = env
    router.routes["https://news-b.example/feed"] = httpx.Response(500)
    c = NewsCollector(archive=HtmlArchive(tmp / "arc"))
    out = run_collector(c, **kw)
    assert out.status == "ok" and out.result.rows == {"articles": 2}
    assert out.result.cursor["feed_failures"] == {"b": 1}
    assert not [s for s in fake.sent if s[0] == "critical"]  # one blip only warns
    for _ in range(3):  # 4 consecutive failures -> critical alert
        out = run_collector(c, **kw)
    assert out.result.cursor["feed_failures"] == {"b": 4}
    assert any("feed_down" in s[1] for s in fake.sent if s[0] == "critical")


@pytest.mark.db
def test_news_all_feeds_down_fails(env):
    kw, router, fake, tmp = env
    for u in NEWS_FEEDS.values():
        router.routes[u] = httpx.ConnectError("down")
    out = run_collector(NewsCollector(archive=HtmlArchive(tmp / "arc")), **kw)
    assert out.status == "failed" and "all 2 news feeds failed" in out.error


@pytest.mark.db
def test_article_page_failure_stored(env):
    kw, router, _, tmp = env
    router.routes["https://news-a.example/tech/chipmaker-record"] = httpx.Response(403)
    run_collector(NewsCollector(archive=HtmlArchive(tmp / "arc")), **kw)
    rows = {r[1]: r for r in _articles(kw)}
    chip = rows["https://news-a.example/tech/chipmaker-record"]
    assert chip[5] == "failed" and chip[6] == 0 and chip[8]  # simhash from title+summary


# ---------------------------------------------------------------- fed
def test_fed_classify_and_speaker(settings):
    cfg = settings.config.collect.fed
    assert classify(cfg, "press_monetary", "Federal Reserve issues FOMC statement") == (
        "fomc_statement"
    )
    assert classify(cfg, "press_all", "Minutes of the Federal Open Market Committee, X") == (
        "minutes"
    )
    assert classify(cfg, "speeches", "Waller, Opening Remarks") == "speech"
    assert classify(cfg, "press_all", "Agencies announce thresholds") == "press"
    assert speaker(cfg, "Waller, Opening Remarks") == ("Waller", "Governor")
    assert speaker(cfg, "Newname, Something") == ("Newname", cfg.default_role)


def _events(kw):
    with kw["connect"]() as c:
        return c.execute(
            "SELECT kind, url, released_at, payload->>'role', payload->>'feed', "
            "(payload->>'backfill')::bool FROM raw_events ORDER BY url"
        ).fetchall()


@pytest.mark.db
def test_fed_first_run_backfill_then_live(env):
    kw, router, _, _ = env
    c = FedRssCollector(clock=lambda: T0)
    out = run_collector(c, **kw)
    assert out.status == "ok" and out.result.rows == {"raw_events": 5}  # statement deduped
    ev = {r[1]: r for r in _events(kw)}
    stmt = ev["https://fed.example/newsevents/pressreleases/monetary20261028a.htm"]
    assert stmt[0] == "fomc_statement" and stmt[4] == "press_monetary"  # first feed wins
    assert stmt[2] == datetime(2026, 10, 28, 18, 0, tzinfo=UTC) and stmt[5]  # published time
    assert ev["https://fed.example/newsevents/speech/waller20261120a.htm"][3] == "Governor"

    # live: a new item gets first-seen time
    later = T0 + timedelta(hours=2)
    router.routes["https://fed.example/feeds/press_monetary.xml"] = (
        (N / "fed_monetary.xml")
        .read_text()
        .replace(
            "</channel>",
            "<item><title>Federal Reserve issues FOMC statement</title>"
            "<link>https://fed.example/newsevents/pressreleases/monetary20261209a.htm</link>"
            "<pubDate>Wed, 09 Dec 2026 19:00:00 GMT</pubDate></item></channel>",
        )
        .encode()
    )
    out = run_collector(FedRssCollector(clock=lambda: later), **kw)
    assert out.result.rows == {"raw_events": 1}
    new = {r[1]: r for r in _events(kw)}[
        "https://fed.example/newsevents/pressreleases/monetary20261209a.htm"
    ]
    assert new[2] == later and new[5] is False
