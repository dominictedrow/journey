"""Corporate actions, Congress, earnings, EDGAR facts and yfinance against fakes (no network)."""

import dataclasses
from datetime import UTC, date, datetime, timedelta

import httpx
import pandas as pd
import pytest
from pydantic import SecretStr

from conftest import with_collect
from qt.collect import pit
from qt.collect.base import run_collector
from qt.collect.congress import CongressCollector, current_congress, is_candidate, status_of
from qt.collect.corp_actions import CorpActionsCollector
from qt.collect.earnings import EarningsCollector
from qt.collect.edgar import EdgarFactsCollector, acceptance_map
from qt.collect.http import HttpClient
from qt.collect.yfin import YfinanceCollector
from qt.common.calendar import ET, TradingCalendar
from qt.common.config import YfTicker
from qt.db.conn import connect

D = date.fromisoformat
NOW = datetime(2026, 11, 25, 16, 0, tzinfo=ET).astimezone(UTC)  # Wed, before the close


def et(d, hh=0, mm=0, ss=0):
    return (
        datetime.combine(D(d), datetime.min.time(), ET)
        .replace(hour=hh, minute=mm, second=ss)
        .astimezone(UTC)
    )


class Handler:
    """MockTransport handler: (host, path) -> callable(request) -> response."""

    def __init__(self):
        self.routes, self.calls = {}, []

    def __call__(self, req):
        self.calls.append((req.url.host, req.url.path, dict(req.url.params)))
        fn = self.routes.get((req.url.host, req.url.path))
        return fn(req) if fn else httpx.Response(404)


@pytest.fixture
def env(settings, db):
    s = dataclasses.replace(
        settings,
        secrets=settings.secrets.model_copy(
            update={
                "congress_api_key": SecretStr("cg"),
                "sec_edgar_user_agent": SecretStr("qt t@example.com"),
                "alpaca_api_key_id": SecretStr("PKTEST"),
                "alpaca_api_secret_key": SecretStr("x"),
            }
        ),
    )
    h = Handler()
    http = HttpClient(s.config.collect.http, transport=httpx.MockTransport(h), sleep=lambda _: None)
    kw = dict(
        settings=s,
        connect=lambda: connect(s, "primary", db),
        http=http,
        cal=TradingCalendar(s.config.calendar),
        clock=lambda: NOW,
    )
    with kw["connect"]() as c:
        aid = c.execute(
            "INSERT INTO assets (ticker, asset_class, cik, security_key) "
            "VALUES ('AAA', 'equity', '0000000001', 'SPX:AAA:2016-01-01') RETURNING asset_id"
        ).fetchone()[0]
        c.execute(
            "INSERT INTO index_membership (index_code, snapshot_at, ticker, source, "
            "asset_id, released_at) VALUES ('SPX', %s, 'AAA', 'wikipedia', %s, %s)",
            (NOW, aid, NOW),
        )
        c.execute("INSERT INTO assets (ticker, asset_class) VALUES ('SPY', 'etf')")
    yield kw, h
    http.close()


def _q(kw, sql, *p):
    with kw["connect"]() as c:
        return c.execute(sql, p).fetchall()


def _with(kw, **sections):
    return dict(kw, settings=with_collect(kw["settings"], **sections))


# ---------------------------------------------------------------- PIT rule
def test_corp_action_release_rule():
    fetched = et("2026-11-25", 16)
    assert pit.corp_action(D("2020-08-31"), fetched) == et("2020-08-31")  # history: ex date
    assert pit.corp_action(D("2026-12-15"), fetched) == fetched  # announced: when we saw it
    with pytest.raises(ValueError):
        pit.corp_action(D("2020-08-31"), datetime(2026, 11, 25))


# ---------------------------------------------------------------- corporate actions
class FakeCA:
    def __init__(self):
        self.calls = []
        self.recs = [
            (
                "forward_splits",
                {
                    "id": "s1",
                    "symbol": "AAA",
                    "ex_date": "2020-08-31",
                    "new_rate": 4,
                    "old_rate": 1,
                },
            ),
            (
                "cash_dividends",
                {"id": "d1", "symbol": "AAA", "ex_date": "2026-12-15", "rate": 0.25},
            ),
            ("cash_dividends", {"id": "bad", "symbol": "AAA"}),  # no date: untimeable
        ]

    def corporate_actions(self, symbols, start, end, limit):
        self.calls.append((tuple(symbols), start, end))
        return [r for r in self.recs]


@pytest.mark.db
def test_corp_actions_backfill_then_window(env):
    kw, _ = env
    fake = FakeCA()
    col = CorpActionsCollector(clock=lambda: NOW, alpaca=lambda ctx: fake)
    out = run_collector(col, **kw)
    assert out.status == "ok", out.error
    rows = dict(_q(kw, "SELECT external_id, released_at FROM raw_events WHERE source='alpaca_ca'"))
    assert rows == {"s1": et("2020-08-31"), "d1": NOW}
    assert out.result.rejected == 1
    assert fake.calls[0][0] == ("AAA", "SPY") and fake.calls[0][1] == D("2016-01-01")
    assert fake.calls[0][2] == D("2026-11-25") + timedelta(days=90)

    fake.recs[1][1]["rate"] = 0.30  # a revised record: new content hash -> new row
    out = run_collector(col, **kw)
    assert out.result.rows["raw_events"] == 1
    assert fake.calls[1][1] == D("2026-11-25") - timedelta(days=30)

    # A symbol added after the backfill (a new extra ETF) gets its full history, the rest a window.
    with kw["connect"]() as c:
        c.execute("INSERT INTO assets (ticker, asset_class) VALUES ('TLT', 'etf')")
        c.commit()
    fake.calls.clear()
    assert run_collector(col, **kw).status == "ok"
    assert [(syms, start) for syms, start, _ in fake.calls] == [
        (("TLT",), D("2016-01-01")),
        (("AAA", "SPY"), D("2026-11-25") - timedelta(days=30)),
    ]


# ---------------------------------------------------------------- Congress
def bill(num, title, intro, latest, ctype="HR"):
    return {
        "congress": 119,
        "type": ctype,
        "number": str(num),
        "title": title,
        "introducedDate": intro,
        "latestAction": {"actionDate": latest, "text": "x"},
        "url": f"https://api.congress.gov/v3/bill/119/{ctype.lower()}/{num}",
    }


def act(d, typ, text, committee=None):
    a = {"actionDate": d, "type": typ, "text": text, "actionCode": typ[:3]}
    if committee:
        a["committees"] = [{"systemCode": committee}]
    return a


BILLS = [
    bill(1, "A referral-only bill", "2025-03-01", "2025-03-01"),
    bill(2, "Tax Relief Act", "2025-03-01", "2025-07-04"),
    bill(3, "Parks Act", "2025-03-01", "2025-05-01"),
    bill(4, "To suspend the public debt limit", "2025-04-01", "2025-04-01"),
]
ACTIONS = {
    "2": [
        act(
            "2025-03-01", "IntroReferral", "Referred to the Committee on Ways and Means.", "hswm00"
        ),
        act(
            "2025-05-02",
            "Committee",
            "Reported (Amended) by the Committee on Ways and Means.",
            "hswm00",
        ),
        act(
            "2025-06-01", "Floor", "Passed/agreed to in House: On passage Passed by recorded vote."
        ),
        act("2025-07-04", "BecameLaw", "Became Public Law No: 119-21."),
    ],
    "3": [
        act("2025-03-01", "IntroReferral", "Referred to Natural Resources.", "hsii00"),
        act("2025-05-01", "Committee", "Reported by Natural Resources.", "hsii00"),
    ],
    "4": [act("2025-04-01", "IntroReferral", "Referred to the Committee on Rules.", "hsru00")],
}


def congress_routes(h, live_bills=()):
    def bills(req):
        q = dict(req.url.params)
        items = list(live_bills) if "fromDateTime" in q else BILLS
        off, lim = int(q["offset"]), int(q["limit"])
        return httpx.Response(
            200, json={"bills": items[off : off + lim], "pagination": {"count": len(items)}}
        )

    h.routes[("api.congress.gov", "/v3/bill/119")] = bills
    h.routes[("api.congress.gov", "/v3/bill")] = bills
    for num, acts in ACTIONS.items():
        h.routes[("api.congress.gov", f"/v3/bill/119/hr/{num}/actions")] = lambda req, a=acts: (
            httpx.Response(200, json={"actions": a, "pagination": {"count": len(a)}})
        )


def test_congress_rules(settings):
    cfg = settings.config.collect.congress
    assert current_congress(D("2026-11-25")) == 119 and current_congress(D("2015-01-06")) == 114
    assert not is_candidate(cfg, BILLS[0]) and is_candidate(cfg, BILLS[1])
    assert is_candidate(cfg, BILLS[3])  # tagged title, even though only referred
    st = [status_of(cfg, a) for a in ACTIONS["2"]]
    assert st == [None, "reported", "passed_house", "became_law"]


@pytest.mark.db
def test_congress_backfill_then_live(env):
    kw, h = env
    kw = _with(kw, congress={"first_congress": 119, "page_limit": 2})
    congress_routes(h)
    out = run_collector(CongressCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    rows = _q(
        kw,
        "SELECT payload->>'bill', payload->>'status', (payload->>'is_event')::bool, "
        "payload->'tags', released_at, event_ts FROM raw_events "
        "WHERE source = 'congress' ORDER BY event_ts, 1",
    )
    bills = {r[0] for r in rows}
    assert bills == {"119-HR-2", "119-HR-4"}  # 1: not a candidate; 3: other committee, no tag
    law = [r for r in rows if r[1] == "became_law"][0]
    assert law[2] is True and law[4] == et("2025-07-04", 23, 59, 59)  # backfill rule
    assert [r for r in rows if r[0] == "119-HR-4"][0][3] == ["debt_ceiling"]
    assert not any("/v3/bill/119/hr/1/actions" in p for _, p, _ in h.calls)
    state = _q(kw, "SELECT cursor FROM collector_state WHERE collector = 'congress'")[0][0]
    assert state["backfill"]["congress"] == 120 and state["live_since"]

    # live: a new action on bill 2 is stamped first-seen
    ACTIONS["2"].append(act("2026-11-26", "Floor", "Passed/agreed to in Senate: Passed."))
    congress_routes(h, live_bills=[bill(2, "Tax Relief Act", "2025-03-01", "2026-11-26")])
    later = NOW + timedelta(days=1)
    out = run_collector(CongressCollector(clock=lambda: later), **kw)
    assert out.status == "ok", out.error
    new = _q(
        kw,
        "SELECT payload->>'status', released_at FROM raw_events WHERE source='congress' "
        "AND payload->>'status' = 'passed_senate'",
    )
    assert new == [("passed_senate", later)]
    ACTIONS["2"].pop()


@pytest.mark.db
def test_congress_backfill_budget_resumes(env):
    kw, h = env
    kw = _with(kw, congress={"first_congress": 119, "page_limit": 2, "backfill_budget": 1})
    congress_routes(h)
    out = run_collector(CongressCollector(clock=lambda: NOW), **kw)
    bf = out.result.cursor["backfill"]
    assert (bf["congress"], bf["offset"]) == (119, 2)  # first page always taken, then stop
    out = run_collector(CongressCollector(clock=lambda: NOW), **kw)
    assert out.result.cursor["backfill"]["congress"] == 120


# ---------------------------------------------------------------- earnings
def nasdaq(days):
    def fn(req):
        d = req.url.params["date"]
        rows = days.get(d)
        return httpx.Response(200, json={"data": {"rows": rows} if rows is not None else None})

    return fn


@pytest.mark.db
def test_earnings_live_backfill_and_moved_date(env):
    kw, h = env
    kw = _with(
        kw, earnings={"backfill_start": "2026-11-18", "backfill_budget": 3, "lookahead_days": 7}
    )
    days = {
        "2026-11-19": [
            {"symbol": "AAA", "time": "time-after-hours", "fiscalQuarterEnding": "Sep/2026"},
            {"symbol": "ZZZ", "time": "time-pre-market", "fiscalQuarterEnding": "Sep/2026"},
        ],
        "2026-12-01": [
            {"symbol": "AAA", "time": "time-pre-market", "fiscalQuarterEnding": "Dec/2026"}
        ],
    }
    h.routes[("api.nasdaq.com", "/api/calendar/earnings")] = nasdaq(days)
    out = run_collector(EarningsCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    rows = _q(
        kw,
        "SELECT external_id, payload->>'report_date', (payload->>'approx')::bool, "
        "released_at FROM raw_events WHERE source = 'nasdaq' ORDER BY 2",
    )
    assert rows == [
        ("AAA:Sep/2026", "2026-11-19", True, et("2026-11-12")),  # backfill: report - 7 days
        ("AAA:Dec/2026", "2026-12-01", False, NOW),  # live: first seen; ZZZ not in universe
    ]
    asked = [q["date"] for host, _, q in h.calls if host == "api.nasdaq.com"]
    assert "2026-11-28" not in asked and "2026-11-29" not in asked  # weekend skipped
    assert out.result.cursor["backfill_next"] == "2026-11-21"  # 18, 19, 20 done (budget 3)

    days["2026-12-02"] = days.pop("2026-12-01")  # the company moves its date
    out = run_collector(EarningsCollector(clock=lambda: NOW), **kw)
    moved = _q(
        kw,
        "SELECT payload->>'report_date' FROM raw_events "
        "WHERE external_id = 'AAA:Dec/2026' ORDER BY released_at, 1",
    )
    assert [r[0] for r in moved] == ["2026-12-01", "2026-12-02"]  # both observations kept


# ---------------------------------------------------------------- EDGAR
def test_acceptance_map_parses_utc():
    m = acceptance_map(
        [{"accessionNumber": ["a-1"], "acceptanceDateTime": ["2026-11-03T13:05:15.000Z"]}]
    )
    assert m["a-1"] == datetime(2026, 11, 3, 13, 5, 15, tzinfo=UTC)


def edgar_routes(h, newest="0001-26-000002", cik="0000000001"):
    sub = {
        "filings": {
            "recent": {
                "accessionNumber": [newest, "0001-26-000001"],
                "acceptanceDateTime": ["2026-11-03T21:05:00.000Z", ""],
            },
            "files": [{"name": "CIK0000000001-submissions-001.json"}],
        }
    }
    old = {
        "accessionNumber": ["0001-15-000009"],
        "acceptanceDateTime": ["2015-02-01T21:00:00.000Z"],
    }
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2026-07-01",
                                "end": "2026-09-30",
                                "val": 100,
                                "accn": newest,
                                "form": "10-Q",
                                "filed": "2026-11-03",
                            },
                            {
                                "start": "2025-07-01",
                                "end": "2025-09-30",
                                "val": 90,
                                "accn": newest,
                                "form": "10-Q",
                                "filed": "2026-11-03",
                            },
                            {
                                "start": "2025-07-01",
                                "end": "2025-09-30",
                                "val": 90,
                                "accn": "0001-26-000001",
                                "form": "10-Q",
                                "filed": "2026-08-01",
                            },
                            {
                                "start": "2014-10-01",
                                "end": "2014-12-31",
                                "val": 50,
                                "accn": "0001-15-000009",
                                "form": "10-Q",
                                "filed": "2015-02-01",
                            },
                        ]
                    }
                }
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "end": "2026-10-20",
                                "val": 1e9,
                                "accn": newest,
                                "form": "10-Q",
                                "filed": "2026-11-03",
                            }
                        ]
                    }
                }
            },
        }
    }
    sub["filings"]["files"] = [{"name": f"CIK{cik}-submissions-001.json"}]
    h.routes[("data.sec.gov", f"/submissions/CIK{cik}.json")] = lambda r: httpx.Response(
        200, json=sub
    )
    h.routes[("data.sec.gov", f"/submissions/CIK{cik}-submissions-001.json")] = lambda r: (
        httpx.Response(200, json=old)
    )
    h.routes[("data.sec.gov", f"/api/xbrl/companyfacts/CIK{cik}.json")] = lambda r: (
        httpx.Response(200, json=facts)
    )


@pytest.mark.db
def test_edgar_facts_acceptance_join_fallback_and_skip(env, tmp_path):
    kw, h = env
    kw = _with(kw, edgar={"cache_dir": str(tmp_path / "edgar")})
    edgar_routes(h)
    with kw["connect"]() as c:  # a second share class of the same filer (GOOGL/GOOG)
        c.execute(
            "INSERT INTO assets (ticker, asset_class, cik, security_key) "
            "VALUES ('AAB', 'equity', '0000000001', 'SPX:AAB:2016-01-01')"
        )
    out = run_collector(EdgarFactsCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    per_asset = _q(kw, "SELECT a.ticker, count(*) FROM raw_fundamentals f JOIN assets a "
                       "USING (asset_id) GROUP BY 1 ORDER BY 1")
    assert per_asset == [("AAA", 5), ("AAB", 5)]
    assert out.result.cursor["filers"] == 1 and out.result.cursor["unchanged"] == 0
    rows = _q(
        kw,
        "SELECT fiscal_period, metric, accession, released_at FROM raw_fundamentals f "
        "JOIN assets a USING (asset_id) WHERE a.ticker = 'AAA' "
        "ORDER BY released_at, fiscal_period",
    )
    assert rows == [
        (
            "2014-10-01/2014-12-31",
            "us-gaap:Revenues",
            "0001-15-000009",
            datetime(2015, 2, 1, 21, tzinfo=UTC),
        ),  # acceptance from the older (cached) page
        ("2025-07-01/2025-09-30", "us-gaap:Revenues", "0001-26-000001", et("2026-08-02")),
        (
            "2025-07-01/2025-09-30",
            "us-gaap:Revenues",
            "0001-26-000002",
            datetime(2026, 11, 3, 21, 5, tzinfo=UTC),
        ),  # prior-year comparative, own row
        (
            "2026-07-01/2026-09-30",
            "us-gaap:Revenues",
            "0001-26-000002",
            datetime(2026, 11, 3, 21, 5, tzinfo=UTC),
        ),
        (
            "2026-10-20",
            "dei:EntityCommonStockSharesOutstanding",
            "0001-26-000002",
            datetime(2026, 11, 3, 21, 5, tzinfo=UTC),
        ),
    ]
    assert out.result.cursor["approx_released"] == 2  # filed + 1 day fallback, once per asset
    assert (tmp_path / "edgar" / "submissions" / "CIK0000000001-submissions-001.json").exists()

    h.calls.clear()
    out = run_collector(EdgarFactsCollector(clock=lambda: NOW), **kw)
    assert out.result.cursor["unchanged"] == 1
    assert not any("companyfacts" in p for _, p, _ in h.calls)  # nothing new filed: skipped

    # a new concept in config refetches every filer once, even with nothing new filed
    concepts = kw["settings"].config.collect.edgar.concepts + ["us-gaap:CostOfRevenue"]
    kw = _with(kw, edgar={"cache_dir": str(tmp_path / "edgar"), "concepts": concepts})
    h.calls.clear()
    out = run_collector(EdgarFactsCollector(clock=lambda: NOW), **kw)
    assert out.result.cursor["unchanged"] == 0
    assert any("companyfacts" in p for _, p, _ in h.calls)
    h.calls.clear()
    out = run_collector(EdgarFactsCollector(clock=lambda: NOW), **kw)
    assert out.result.cursor["unchanged"] == 1

    # an asset that gains this filer's CIK later (renamed ticker, former member) is fetched once
    with kw["connect"]() as c:
        c.execute(
            "INSERT INTO assets (ticker, asset_class, cik, security_key) "
            "VALUES ('AAC', 'equity', '0000000001', 'SPX:AAC:2016-01-01')"
        )
    h.calls.clear()
    out = run_collector(EdgarFactsCollector(clock=lambda: NOW), **kw)
    assert out.result.cursor["unchanged"] == 0
    assert _q(kw, "SELECT count(*) FROM raw_fundamentals f JOIN assets a USING (asset_id) "
                  "WHERE a.ticker = 'AAC'") == [(5,)]
    out = run_collector(EdgarFactsCollector(clock=lambda: NOW), **kw)
    assert out.result.cursor["unchanged"] == 1


@pytest.mark.db
def test_edgar_predecessor_cik_only_before_until(env, tmp_path):
    kw, h = env
    edgar_routes(h)
    edgar_routes(h, newest="0009-26-000002", cik="0000000009")  # the old filer
    from qt.common.config import PredecessorCikCfg

    pred = {"SPX:AAA:2016-01-01": [PredecessorCikCfg(cik="9", until="2020-01-01", note="t")]}
    kw = _with(kw, edgar={"cache_dir": str(tmp_path / "edgar"), "predecessor_ciks": pred})
    out = run_collector(EdgarFactsCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    assert out.result.cursor["filers"] == 2
    assert any("CIK0000000009.json" in p for _, p, _ in h.calls)
    # the old filer's 2025/2026 periods are cut (>= until); only the successor's remain
    rows = _q(kw, "SELECT fiscal_period, accession FROM raw_fundamentals")
    assert not [r for r in rows if r[1].startswith("0009-")]
    assert ("2014-10-01/2014-12-31", "0001-15-000009") in rows


@pytest.mark.db
def test_edgar_ua_guard(env):
    kw, _ = env
    s = kw["settings"]
    bad = dict(
        kw,
        settings=dataclasses.replace(
            s,
            secrets=s.secrets.model_copy(
                update={"sec_edgar_user_agent": SecretStr("no-email-here")}
            ),
        ),
    )
    out = run_collector(EdgarFactsCollector(clock=lambda: NOW), **bad)
    assert out.status == "failed" and "EdgarUaError" in out.error
    none = dict(
        kw,
        settings=dataclasses.replace(
            s, secrets=s.secrets.model_copy(update={"sec_edgar_user_agent": None})
        ),
    )
    assert run_collector(EdgarFactsCollector(clock=lambda: NOW), **none).status == "skipped"


# ---------------------------------------------------------------- yfinance
def frame(dates, closes):
    idx = pd.DatetimeIndex([pd.Timestamp(d).tz_localize("America/New_York") for d in dates])
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Adj Close": closes,
            "Volume": [0.0] * len(closes),
        },
        index=idx,
    )


@pytest.mark.db
def test_yfinance_release_rules_and_assets(env):
    kw, _ = env
    kw = _with(
        kw,
        yfinance={
            "tickers": {
                "SPY": YfTicker(rule="xnys", asset_class="etf"),
                "^VIX": YfTicker(rule="xnys", asset_class="index"),
                "NQ=F": YfTicker(rule="futures_fx", asset_class="future"),
            }
        },
    )
    calls = []

    def fetch(t, start):
        calls.append((t, start))
        if t == "NQ=F":  # futures list the current trade date (in progress at 16:00 ET)
            return frame(["2026-11-23", "2026-11-24", "2026-11-25"], [1.0, 2.0, 3.0])
        return frame(["2026-11-22", "2026-11-24", "2026-11-25"], [1.0, float("nan"), 3.0])

    col = YfinanceCollector(clock=lambda: NOW, fetch=fetch, sleep=lambda _: None, cache=False)
    out = run_collector(col, **kw)
    assert out.status == "ok", out.error
    rows = _q(
        kw,
        "SELECT a.ticker, a.asset_class, p.ts, p.released_at FROM raw_prices p "
        "JOIN assets a USING (asset_id) WHERE p.source = 'yfinance' ORDER BY 1, 3",
    )
    assert rows == [
        ("NQ=F", "future", et("2026-11-23"), et("2026-11-23", 17)),
        ("NQ=F", "future", et("2026-11-24"), et("2026-11-24", 17)),
    ]  # SPY/^VIX: Sunday rejected, NaN rejected, today not closed yet
    assert out.result.rejected == 4
    spy = _q(kw, "SELECT count(*) FROM assets WHERE ticker = 'SPY'")[0][0]
    assert spy == 1  # reused the existing ETF row
    assert calls[0] == ("SPY", D("2000-01-01"))

    after = et("2026-11-25", 17, 30)
    out = run_collector(
        YfinanceCollector(clock=lambda: after, fetch=fetch, sleep=lambda _: None, cache=False), **kw
    )
    got = {
        (t, r)
        for t, r in _q(
            kw,
            "SELECT a.ticker, p.released_at FROM raw_prices p JOIN "
            "assets a USING (asset_id) WHERE p.ts = %s",
            et("2026-11-25"),
        )
    }
    assert got == {
        ("SPY", et("2026-11-25", 16, 15)),
        ("^VIX", et("2026-11-25", 16, 15)),
        ("NQ=F", et("2026-11-25", 17)),
    }


# ---------------------------------------------------------------- identity overrides (D-035)
def _override_settings(kw):
    from qt.common.config import IdentityOverrideCfg

    s = kw["settings"]
    ov = IdentityOverrideCfg(
        security_key="SPX:AAA:2016-01-01",
        listed_until=D("2026-11-20"),
        exclude_sources=["alpaca"],
        successor={"symbol": "NEWAAA", "start": "2026-11-18", "end": "2026-11-20"},
        note="test",
    )
    collect = s.config.collect.model_copy(update={"identity_overrides": [ov]})
    return dict(
        kw,
        settings=dataclasses.replace(
            s, config=s.config.model_copy(update={"collect": collect})
        ),
    )


class FakeBars:
    def __init__(self):
        self.calls = []

    def stock_bars(self, symbol, start, end, adjustment="raw"):
        from qt.collect.alpaca import Bar

        self.calls.append((symbol, start, end))
        days = [D("2026-11-18"), D("2026-11-19"), D("2026-11-20"), D("2026-11-23")]
        return [Bar(et(str(d)).astimezone(UTC), 1, 1, 1, 40.0 + i, 1e6, None, None)
                for i, d in enumerate(days)]  # fmt: skip


@pytest.mark.db
def test_succ_bars_refetch_once_under_successor_symbol(env):
    from qt.collect.bars import SuccessorBarsCollector

    kw = _override_settings(env[0])
    fake = FakeBars()
    col = SuccessorBarsCollector(clock=lambda: NOW, alpaca=lambda ctx: fake)
    out = run_collector(col, **kw)
    assert out.status == "ok", out.error
    rows = _q(
        kw,
        "SELECT a.security_key, p.ts, p.close, p.source, p.released_at FROM raw_prices p "
        "JOIN assets a USING (asset_id) ORDER BY p.ts",
    )
    # stored under the overridden security, capped at successor.end, released at the close
    assert [(r[0], r[2], r[3]) for r in rows] == [
        ("SPX:AAA:2016-01-01", 40.0, "alpaca_succ"),
        ("SPX:AAA:2016-01-01", 41.0, "alpaca_succ"),
        ("SPX:AAA:2016-01-01", 42.0, "alpaca_succ"),
    ]
    assert rows[0][4] == et("2026-11-18", 16)
    assert fake.calls == [("NEWAAA", D("2026-11-18"), D("2026-11-20"))]
    assert run_collector(col, **kw).status == "ok"
    assert len(fake.calls) == 1  # done: not requested again


@pytest.mark.db
def test_bar_exclusions_and_identity_check(env):
    from qt.collect import identity_check
    from qt.features.inputs import bar_exclusions, load_bars

    kw = _override_settings(env[0])
    s = kw["settings"]
    with kw["connect"]() as c:
        aid = c.execute(
            "SELECT asset_id FROM assets WHERE security_key = 'SPX:AAA:2016-01-01'"
        ).fetchone()[0]
        for i in range(25):  # a "member" whose stored bars trade $10k a day: contaminated
            t = et("2026-10-01").astimezone(UTC) + timedelta(days=i)
            for src, close in (("alpaca", 1.0), ("alpaca_succ", 50.0)):
                vol = 1e4 if src == "alpaca" else 1e6
                c.execute(
                    "INSERT INTO raw_prices (asset_id, ts, close, volume, source, released_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (aid, t, close, vol, src, t + timedelta(hours=16)),
                )
        c.execute(
            "INSERT INTO index_membership (index_code, snapshot_at, ticker, source, asset_id, "
            "released_at) VALUES ('SPX', %s, 'AAA', 'fja05680', %s, %s)",
            (et("2026-09-30"), aid, et("2026-09-30")),
        )
        c.commit()
        excl = bar_exclusions(c, s.config.collect.identity_overrides)
        assert excl == [(aid, "alpaca")]
        bars = load_bars(c, NOW, s.config.features.bars, exclude=excl)
        assert set(bars["source"]) == {"alpaca_succ"}
        # with the override the hidden bars don't count; without it the asset is flagged
        assert identity_check.suspects(c, s.config.collect) == []
        bare = s.config.collect.model_copy(
            update={"identity_overrides": [], "identity_check": s.config.collect.identity_check
                    .model_copy(update={"sources": ["alpaca"]})}
        )  # fmt: skip
        assert [r[1] for r in identity_check.suspects(c, bare)] == ["SPX:AAA:2016-01-01"]
