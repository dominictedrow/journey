"""EDGAR ticker map + alias rebuild, and tagging with historical tickers (no network)."""

import dataclasses
import json
from datetime import UTC, date, datetime

import httpx
import pytest
from pydantic import SecretStr

from conftest import Router
from qt.collect.aliases import AliasesCollector, short_name
from qt.collect.base import run_collector
from qt.collect.http import HttpClient
from qt.collect.news.tag import Tagger, load_aliases
from qt.common.calendar import TradingCalendar
from qt.db.conn import connect

FILLER = 1000  # the collector refuses a map with < 1000 tickers
EDGAR = {str(i): v for i, v in enumerate([
    {"cik_str": 1326801, "ticker": "META", "title": "Meta Platforms, Inc."},
    {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
    {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."},
    {"cik_str": 27419, "ticker": "TGT", "title": "TARGET CORP"},
    {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
] + [{"cik_str": 10_000 + i, "ticker": f"Z{i}", "title": "F"} for i in range(FILLER)])}


def test_short_name(settings):
    cfg = settings.config.aliases
    assert short_name("Alphabet Inc. (Class A)", cfg) == "Alphabet"
    assert short_name("Target Corporation", cfg) == "Target"
    assert short_name("Meta Platforms, Inc.", cfg) == "Meta Platforms"
    assert short_name("Berkshire Hathaway Class B", cfg) == "Berkshire Hathaway"
    assert short_name("JPMorgan Chase & Co.", cfg) == "JPMorgan Chase"


@pytest.fixture
def env(settings, db):
    s = dataclasses.replace(
        settings,
        secrets=settings.secrets.model_copy(
            update={"sec_edgar_user_agent": SecretStr("qt test test@example.com")}
        ),
    )
    router = Router({s.config.collect.edgar.tickers_url: json.dumps(EDGAR)})
    http = HttpClient(s.config.collect.http, transport=httpx.MockTransport(router),
                      sleep=lambda _: None)  # fmt: skip
    kw = dict(settings=s, connect=lambda: connect(s, "primary", db), http=http,
              cal=TradingCalendar(s.config.calendar))  # fmt: skip
    with kw["connect"]() as c:
        ids = {}
        for key, ticker, name in [
            ("SPX:FB:2015-12-31", "META", "Meta Platforms"),
            ("SPX:GOOGL:2015-12-31", "GOOGL", "Alphabet Inc. (Class A)"),
            ("SPX:GOOG:2015-12-31", "GOOG", "Alphabet Inc. (Class C)"),
            ("SPX:TGT:2015-12-31", "TGT", "Target Corporation"),
        ]:
            ids[ticker] = c.execute(
                "INSERT INTO assets (ticker, asset_class, name, security_key) "
                "VALUES (%s, 'equity', %s, %s) RETURNING asset_id", (ticker, name, key),
            ).fetchone()[0]  # fmt: skip
        snaps = [
            (datetime(2015, 12, 31, 5, tzinfo=UTC), "FB"),
            (datetime(2022, 6, 9, 4, tzinfo=UTC), "META"),
        ]
        for at, t in snaps:
            c.execute(
                "INSERT INTO index_membership (index_code, snapshot_at, ticker, source, asset_id, "
                "released_at) VALUES ('SPX', %s, %s, 'fja05680', %s, %s)", (at, t, ids["META"], at),
            )  # fmt: skip
    yield kw, router, ids
    http.close()


@pytest.mark.db
def test_aliases_rebuild_and_historical_tags(env):
    kw, router, ids = env
    out = run_collector(AliasesCollector(), **kw)
    assert out.status == "ok", out.error
    with kw["connect"]() as c:
        cik = dict(c.execute("SELECT ticker, cik FROM assets").fetchall())
        tagger = Tagger(load_aliases(c), kw["settings"].config.aliases.finance_cues)
        n1 = c.execute("SELECT count(*) FROM aliases").fetchone()[0]
    assert cik["META"] == "0001326801" and cik["TGT"] == "0000027419"

    d19, d23 = date(2019, 3, 1), date(2023, 3, 1)
    assert tagger.tag("Facebook launched a feature", d19) == ["FB"]
    assert tagger.tag("Facebook launched a feature", d23) == ["META"]
    assert tagger.tag("google unveils a new model", d23) == ["GOOG", "GOOGL"]
    assert tagger.tag("Target shares fell 5% after the update.", d23) == ["TGT"]
    assert tagger.tag("Target cut its forecast.", d23) == []  # one-word name needs a cue
    assert tagger.tag("The target for inflation is 2%", d23) == []
    assert tagger.tag("FB stock rose", d19) == ["FB"] and tagger.tag("FB stock rose", d23) == []

    run_collector(AliasesCollector(), **kw)  # rebuild replaces, never duplicates
    with kw["connect"]() as c:
        assert c.execute("SELECT count(*) FROM aliases").fetchone()[0] == n1


@pytest.mark.db
def test_edgar_ua_guard(env):
    kw, router, _ = env
    s = kw["settings"]
    no_at = dataclasses.replace(s, secrets=s.secrets.model_copy(
        update={"sec_edgar_user_agent": SecretStr("qt test")}))  # fmt: skip
    out = run_collector(AliasesCollector(), **dict(kw, settings=no_at))
    assert out.status == "failed" and "contact email" in out.error
    assert router.calls == []  # refused before any request
    missing = dataclasses.replace(s, secrets=s.secrets.model_copy(
        update={"sec_edgar_user_agent": None}))  # fmt: skip
    assert run_collector(AliasesCollector(), **dict(kw, settings=missing)).status == "skipped"
