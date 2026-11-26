"""News helpers: canonical URLs, title hashes, SimHash, extraction, archive, alias tagging."""

from datetime import UTC, date, datetime

from conftest import FIXTURES
from qt.collect.news.dedup import hamming, simhash64, title_hash, to_signed
from qt.collect.news.extract import HtmlArchive, extract
from qt.collect.news.feeds import canonical_url, parse_feed, url_hash
from qt.collect.news.tag import Alias, Tagger

NEWS = FIXTURES / "news"


def test_canonical_url_strips_query_fragment():
    a = "https://News-A.example/markets/rates-hold?utm_source=rss&utm_medium=feed#0"
    assert canonical_url(a) == "https://news-a.example/markets/rates-hold"
    assert canonical_url("https://x.example/a/b/") == "https://x.example/a/b"
    assert canonical_url("https://x.example:443/a") == "https://x.example/a"
    assert url_hash(a) == url_hash("https://news-a.example/markets/rates-hold")


def test_identical_titles_collide():
    assert title_hash("Rates hold steady as inflation cools") == title_hash(
        "RATES HOLD STEADY as inflation cools!"
    )
    assert title_hash("Rates hold steady") != title_hash("Rates rise")
    assert title_hash("") is None and title_hash("  !! ") is None


def test_simhash_near_dup():
    base = (NEWS / "article_rates.html").read_text()
    body = extract(base, "https://a/x", 150).body
    edited = body.replace("prudent for now", "prudent for the moment", 1)
    other = extract((NEWS / "article_chip.html").read_text(), "https://a/y", 150).body
    h1, h2, h3 = simhash64(body), simhash64(edited), simhash64(other)
    assert hamming(h1, h2) <= 3  # flagged as near-dup
    assert hamming(h1, h3) > 3
    assert simhash64("") is None


def test_to_signed_fits_bigint():
    assert to_signed((1 << 64) - 1) == -1
    assert to_signed(5) == 5
    assert -(1 << 63) <= to_signed(1 << 63) < (1 << 63)


def test_short_body_flagged_not_dropped():
    ex = extract((NEWS / "article_short.html").read_text(), "https://b/wrap", 150)
    assert ex.status == "short" and ex.body and 0 < ex.word_count < 150
    assert extract(None, "u", 150).status == "failed"
    assert extract("<html></html>", "u", 150).status == "failed"
    ok = extract((NEWS / "article_rates.html").read_text(), "https://a/r", 150)
    assert ok.status == "ok" and ok.word_count >= 150
    assert "Copyright" not in ok.body and "Home | Markets" not in ok.body  # boilerplate dropped


def test_parse_feed_fields():
    es = parse_feed("a", (NEWS / "feed_a.xml").read_bytes())
    assert len(es) == 2
    e = es[0]
    assert e.canonical_url == "https://news-a.example/markets/rates-hold"
    assert e.published_at == datetime(2026, 11, 20, 14, 0, tzinfo=UTC)
    assert e.guid == "https://news-a.example/markets/rates-hold#0"


def test_archive_roundtrip(tmp_path):
    arc = HtmlArchive(tmp_path)
    t = datetime(2026, 11, 25, 23, 59, tzinfo=UTC)
    p = arc.write("bbc", t, "https://x/1", 200, "<p>1</p>")
    arc.write("bbc", t, "https://x/2", None, None)
    assert p.relative_to(tmp_path).as_posix() == "2026/11/25/bbc.jsonl.gz"
    rows = HtmlArchive.read(p)
    assert [r["url"] for r in rows] == ["https://x/1", "https://x/2"] and rows[1]["html"] is None


# ---------------------------------------------------------------- alias tagging (Q7)
CUES = ["shares", "stock", "earnings"]
ALIASES = [
    Alias("GOOGL", "Google"),
    Alias("GOOG", "Google"),
    Alias("GOOGL", "Alphabet"),
    Alias("TGT", "Target", case_sensitive=True, needs_cue=True),
    Alias("FB", "Facebook", valid_to=date(2022, 6, 9)),
    Alias("META", "Facebook", valid_from=date(2022, 6, 9)),
    Alias("META", "Meta Platforms"),
]


def test_alias_tagging():
    t = Tagger(ALIASES, CUES)
    d = date(2026, 11, 25)
    assert t.tag("google unveils a new model", d) == ["GOOG", "GOOGL"]
    assert t.tag("The target for inflation is 2%", d) == []
    assert t.tag("Target cut its forecast.", d) == []  # no finance cue
    assert t.tag("Target shares fell 5% after the update.", d) == ["TGT"]
    assert t.tag("target shares fell", d) == []  # needs_cue => case-sensitive
    assert t.tag("Facebook launched a feature", date(2019, 3, 1)) == ["FB"]
    assert t.tag("Facebook launched a feature", d) == ["META"]
    assert t.tag("Googleplex tour", d) == []  # word boundary


def test_tagger_without_aliases():
    assert Tagger([], CUES).tag("Google shares", date(2026, 1, 1)) == []
