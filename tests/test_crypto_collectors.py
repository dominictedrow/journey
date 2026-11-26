"""Crypto structure (CoinGecko, DefiLlama) and derivatives (funding, OI) against fake APIs."""

import dataclasses
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr

from qt.collect.base import run_collector
from qt.collect.crypto_derivs import (
    CryptoOiCollector,
    FundingCollector,
    annualize,
    funding_rows,
    to_usd,
)
from qt.collect.crypto_structure import CoingeckoCollector, DefillamaCollector, daily_rows
from qt.collect.http import HttpClient
from qt.common.calendar import TradingCalendar
from qt.db.conn import connect

NOW = datetime(2026, 11, 26, 6, 0, tzinfo=UTC)
DAY = timedelta(days=1)
MID = datetime(2026, 11, 26, tzinfo=UTC)  # today's 00:00 UTC point: released tomorrow


def ms(dt):
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------- pure functions
def test_annualize_okx_8h_and_hyperliquid_1h():
    assert annualize(0.0001, 8) == pytest.approx(0.0001 * 3 * 365)
    assert annualize(0.0000125, 1) == pytest.approx(0.0000125 * 24 * 365)
    assert annualize(0.0001, 8) == pytest.approx(annualize(0.0000125, 1))
    with pytest.raises(ValueError):
        annualize(0.1, 0)


def test_to_usd_at_mark():
    assert to_usd(2.5, "base", 60_000) == 150_000
    assert to_usd(879_487_230, "usd", 60_000) == 879_487_230
    with pytest.raises(ValueError):
        to_usd(1, "contracts", 1)


def test_funding_rows_interval_and_guards():
    t0 = datetime(2026, 11, 25, tzinfo=UTC)
    s = [
        (t0, 1e-4),
        (t0 + timedelta(hours=8), 2e-4),
        (t0 + timedelta(hours=12), 3e-4),
        (NOW + timedelta(hours=2), 9e-4),
    ]
    rows = funding_rows("okx", "BTC", s, None, 8, NOW)
    assert [(r[2], r[5]) for r in rows] == [
        (t0, 8),
        (t0 + timedelta(hours=8), 8),
        (t0 + timedelta(hours=12), 4),
    ]  # first: default interval; the future settlement is not stored
    assert all(r[2] == r[9] for r in rows)  # released at settlement
    again = funding_rows("okx", "BTC", s, t0 + timedelta(hours=8), 8, NOW)
    assert [r[2] for r in again] == [t0 + timedelta(hours=12)]
    gap = funding_rows(
        "okx", "BTC", [(t0 + timedelta(days=3), 1e-4)], t0, 8, t0 + timedelta(days=4)
    )
    assert gap[0][5] == 8  # a data gap doesn't become a 72 h interval


def test_daily_rows_drop_now_points_and_unreleased():
    pts = [(MID - 2 * DAY, 1.0), (MID - DAY, 2.0), (MID, 3.0), (NOW, 4.0), (MID - 3 * DAY, None)]
    rows, bad = daily_rows("coingecko", "volume_usd", "BTC", pts, "usd", NOW)
    assert [(r[3], r[4], r[6]) for r in rows] == [
        (MID - 2 * DAY, 1.0, MID - DAY),
        (MID - DAY, 2.0, MID),
    ]
    assert bad == 1


# ---------------------------------------------------------------- fake APIs
class FakeApis:
    def __init__(self):
        self.calls = []
        self.fail = set()
        self.okx_funding = {
            "BTC-USDT-SWAP": [MID - timedelta(hours=8 * i) for i in range(6)],
        }
        self.hl_start = datetime(2026, 11, 25, tzinfo=UTC)

    def __call__(self, req: httpx.Request) -> httpx.Response:
        host, path, q = req.url.host, req.url.path, dict(req.url.params)
        self.calls.append((host, path, q))
        if host in self.fail:
            return httpx.Response(500)
        if host == "api.coingecko.com":
            if path.endswith("/global"):
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "updated_at": int(NOW.timestamp()) - 60,
                            "market_cap_percentage": {"btc": 57.1, "eth": 11.2},
                            "total_market_cap": {"usd": 2.1e12},
                            "total_volume": {"usd": 9e10},
                        }
                    },
                )
            days = int(q["days"])
            pts = [MID - i * DAY for i in range(days, -1, -1)] + [NOW]
            series = [[ms(p), 100.0 + i] for i, p in enumerate(pts)]
            return httpx.Response(
                200, json={"total_volumes": series, "market_caps": series, "prices": series}
            )
        if host == "stablecoins.llama.fi":
            pts = [MID - i * DAY for i in range(5, -1, -1)]
            return httpx.Response(
                200,
                json=[
                    {
                        "date": str(int(p.timestamp())),
                        "totalCirculatingUSD": {"peggedUSD": 1e11 + i},
                    }
                    for i, p in enumerate(pts)
                ],
            )
        if host == "www.okx.com":
            inst = q["instId"]
            if path.endswith("funding-rate-history"):
                times = self.okx_funding.get(inst, [MID])
                if "after" in q:
                    times = [t for t in times if ms(t) < int(q["after"])]
                times = sorted(times, reverse=True)[: int(q["limit"])]
                return httpx.Response(
                    200,
                    json={
                        "code": "0",
                        "data": [
                            {
                                "fundingTime": str(ms(t)),
                                "fundingRate": "0.0001",
                                "realizedRate": "0.00009",
                            }
                            for t in times
                        ],
                    },
                )
            if path.endswith("open-interest"):
                return httpx.Response(
                    200,
                    json={"code": "0", "data": [{"oi": "1000", "oiCcy": "10", "oiUsd": "600000"}]},
                )
            if path.endswith("mark-price"):
                return httpx.Response(200, json={"code": "0", "data": [{"markPx": "60000"}]})
        if host == "api.hyperliquid.xyz":
            body = json.loads(req.content)
            if body["type"] == "fundingHistory":
                start = datetime.fromtimestamp(body["startTime"] / 1000, UTC)
                hours = [self.hl_start + timedelta(hours=h, milliseconds=11) for h in range(6)]
                return httpx.Response(
                    200,
                    json=[
                        {"coin": body["coin"], "fundingRate": "0.0000125", "time": ms(t)}
                        for t in hours
                        if t >= start and t <= NOW
                    ],
                )
            return httpx.Response(
                200,
                json=[
                    {"universe": [{"name": s} for s in ("BTC", "ETH", "SOL")]},
                    [{"openInterest": "2", "markPx": "100"}] * 3,
                ],
            )
        if host == "www.deribit.com":
            usd = q["instrument_name"].endswith("-PERPETUAL") and "USDC" not in q["instrument_name"]
            return httpx.Response(
                200,
                json={"result": [{"open_interest": 5000.0 if usd else 3.0, "mark_price": 200.0}]},
            )
        if host == "futures.kraken.com":
            return httpx.Response(200, json={"ticker": {"openInterest": 4.0, "markPrice": 50.0}})
        return httpx.Response(404)


@pytest.fixture
def env(settings, db):
    s = dataclasses.replace(
        settings,
        secrets=settings.secrets.model_copy(
            update={"coingecko_demo_api_key": SecretStr("cg-test")}
        ),
    )
    fake = FakeApis()
    http = HttpClient(
        s.config.collect.http, transport=httpx.MockTransport(fake), sleep=lambda _: None
    )
    kw = dict(
        settings=s,
        connect=lambda: connect(s, "primary", db),
        http=http,
        cal=TradingCalendar(s.config.calendar),
        clock=lambda: NOW,
    )
    with kw["connect"]() as c:
        for sym in ("BTC/USD", "ETH/USD", "NOID/USD"):
            c.execute(
                "INSERT INTO assets (ticker, asset_class, alpaca_tradable) "
                "VALUES (%s, 'crypto', true)",
                (sym,),
            )
    yield kw, fake
    http.close()


def _q(kw, sql, *p):
    with kw["connect"]() as c:
        return c.execute(sql, p).fetchall()


@pytest.mark.db
def test_coingecko_backfill_then_increment(env):
    kw, fake = env
    out = run_collector(CoingeckoCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    rows = _q(
        kw,
        "SELECT subject, metric, count(*), max(ts), min(released_at - ts) "
        "FROM raw_metrics WHERE source = 'coingecko' AND subject <> '' "
        "GROUP BY 1, 2 ORDER BY 1, 2",
    )
    assert {r[0] for r in rows} == {"BTC", "ETH"}  # NOID has no CoinGecko id: skipped
    for _, _, n, last, lag in rows:
        assert n == 365 and last == MID - DAY and lag == DAY  # today's point not yet released
    glob = dict(_q(kw, "SELECT metric, released_at FROM raw_metrics WHERE subject = ''"))
    assert glob["btc_dominance_pct"] == NOW
    days_asked = [int(q["days"]) for h, p, q in fake.calls if "market_chart" in p]
    assert days_asked == [365, 365]

    fake.calls.clear()
    out = run_collector(CoingeckoCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok" and out.result.rows["raw_metrics"] == 0  # idempotent
    assert [int(q["days"]) for h, p, q in fake.calls if "market_chart" in p] == [30, 30]


@pytest.mark.db
def test_coingecko_missing_key_skips(env):
    kw, _ = env
    s = kw["settings"]
    kw = dict(
        kw,
        settings=dataclasses.replace(
            s, secrets=s.secrets.model_copy(update={"coingecko_demo_api_key": None})
        ),
    )
    out = run_collector(CoingeckoCollector(clock=lambda: NOW), **kw)
    assert out.status == "skipped" and "COINGECKO_DEMO_API_KEY" in out.error


@pytest.mark.db
def test_defillama_full_history_idempotent(env):
    kw, _ = env
    out = run_collector(DefillamaCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok" and out.result.rows["raw_metrics"] == 5  # today's point excluded
    assert (
        run_collector(DefillamaCollector(clock=lambda: NOW), **kw).result.rows["raw_metrics"] == 0
    )


@pytest.mark.db
def test_funding_okx_pages_back_and_hyperliquid_forward(env, settings):
    kw, fake = env
    s = kw["settings"]
    d = s.config.collect.derivs
    s2 = dataclasses.replace(
        s,
        config=s.config.model_copy(
            update={
                "collect": s.config.collect.model_copy(
                    update={
                        "derivs": d.model_copy(
                            update={
                                "symbols": ["BTC"],
                                "funding_start": "2026-11-25T00:00:00",
                                "okx": d.okx.model_copy(update={"page_limit": 2}),
                            }
                        )
                    }
                )
            }
        ),
    )
    kw = dict(kw, settings=s2)
    out = run_collector(FundingCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    okx = _q(
        kw,
        "SELECT ts, funding_rate, funding_interval_h FROM raw_crypto_derivs "
        "WHERE venue = 'okx' ORDER BY ts",
    )
    assert len(okx) == 6 and okx[-1][0] == MID and okx[0][1] == pytest.approx(0.00009)
    assert {r[2] for r in okx} == {8}
    hl = _q(kw, "SELECT ts, funding_interval_h FROM raw_crypto_derivs WHERE venue = 'hyperliquid'")
    assert len(hl) == 6 and all(t.microsecond == 0 for t, _ in hl) and {h for _, h in hl} == {1}

    fake.okx_funding["BTC-USDT-SWAP"].append(MID + timedelta(hours=4))  # settled at 04:00
    out = run_collector(FundingCollector(clock=lambda: NOW), **kw)
    assert out.result.rows["raw_crypto_derivs"] == 1
    last = _q(
        kw,
        "SELECT funding_interval_h FROM raw_crypto_derivs WHERE venue = 'okx' "
        "ORDER BY ts DESC LIMIT 1",
    )
    assert last[0][0] == 4


@pytest.mark.db
def test_oi_snapshot_all_venues_and_partial_failure(env):
    kw, fake = env
    out = run_collector(CryptoOiCollector(clock=lambda: NOW), **kw)
    assert out.status == "ok", out.error
    rows = {
        (v, s): (n, u, m)
        for v, s, n, u, m in _q(
            kw, "SELECT venue, symbol, oi_contracts, oi_usd, mark_price FROM raw_crypto_derivs"
        )
    }
    assert len(rows) == 12
    assert rows[("okx", "BTC")] == (1000, 600000, 60000)
    assert rows[("deribit", "BTC")] == (5000, 5000, 200)  # inverse: quoted in USD
    assert rows[("deribit", "SOL")] == (3, 600, 200)  # linear USDC: coins x mark
    assert rows[("kraken", "ETH")] == (4, 200, 50)
    assert rows[("hyperliquid", "SOL")] == (2, 200, 100)

    fake.fail.add("futures.kraken.com")
    later = NOW + timedelta(hours=1)
    out = run_collector(CryptoOiCollector(clock=lambda: later), **kw)
    assert out.status == "failed" and "kraken" in out.error
    n = _q(kw, "SELECT count(*) FROM raw_crypto_derivs WHERE ts = %s", later)[0][0]
    assert n == 9  # the other venues were still stored


# ---------------------------------------------------------------- SoSoValue
def sosovalue_env(kw, rows_by_symbol):
    from qt.collect.sosovalue import SosovalueCollector

    s = kw["settings"]
    s = dataclasses.replace(
        s, secrets=s.secrets.model_copy(update={"sosovalue_api_key": SecretStr("soso-test")})
    )
    seen = []

    def handler(req):
        seen.append((req.headers.get("x-soso-api-key"), dict(req.url.params)))
        return httpx.Response(200, json=rows_by_symbol[req.url.params["symbol"]])

    http = HttpClient(
        s.config.collect.http, transport=httpx.MockTransport(handler), sleep=lambda _: None
    )
    return dict(kw, settings=s, http=http), SosovalueCollector, seen


@pytest.mark.db
def test_sosovalue_release_rule_and_t1_nulls(env):
    from qt.common.calendar import ET

    kw, _ = env
    unsettled = {
        "date": "2026-11-25",
        "total_net_inflow": None,
        "total_value_traded": 4.7e9,
        "total_net_assets": None,
        "cum_net_inflow": None,
    }
    settled = {
        "date": "2026-11-24",
        "total_net_inflow": -5.5e7,
        "total_value_traded": 4.1e9,
        "total_net_assets": 5.6e10,
        "cum_net_inflow": 1.35e10,
    }
    rows = {"BTC": [unsettled, settled], "ETH": [settled], "SOL": []}
    kw2, Col, seen = sosovalue_env(kw, rows)
    out = run_collector(Col(clock=lambda: NOW), **kw2)
    assert out.status == "ok", out.error
    assert seen[0][0] == "soso-test" and seen[0][1]["limit"] == "300"
    n = _q(kw2, "SELECT count(*) FROM raw_metrics WHERE source = 'sosovalue'")[0][0]
    assert n == 4 + 1 + 4  # BTC: 4 settled fields + 1 unsettled day's traded value; ETH: 4
    nov24_next_9 = datetime(2026, 11, 25, 9, 0, tzinfo=ET).astimezone(UTC)
    rel24 = _q(
        kw2,
        "SELECT DISTINCT released_at FROM raw_metrics WHERE ts = %s",
        datetime(2026, 11, 24, tzinfo=ET).astimezone(UTC),
    )
    assert rel24 == [(nov24_next_9,)]  # history rule: next morning 09:00 ET
    # 2026-11-25's rule time (11-26 09:00 ET) is after the fetch: known from when we fetched it
    v25 = _q(
        kw2,
        "SELECT released_at FROM raw_metrics WHERE ts = %s",
        datetime(2026, 11, 25, tzinfo=ET).astimezone(UTC),
    )
    assert v25 == [(NOW,)]

    rows["BTC"][0] = dict(settled, date="2026-11-25")  # it settles the next day
    later = NOW + timedelta(days=1)
    out = run_collector(Col(clock=lambda: later), **kw2)
    assert out.result.rows["raw_metrics"] == 3  # the three fields that were null
    assert seen[-1][1]["limit"] == "10"
    kw2["http"].close()


@pytest.mark.db
def test_sosovalue_skipped_without_key(env):
    from qt.collect.sosovalue import SosovalueCollector

    kw, _ = env
    s = kw["settings"]  # independent of whatever key the real .env holds
    s = dataclasses.replace(s, secrets=s.secrets.model_copy(update={"sosovalue_api_key": None}))
    out = run_collector(SosovalueCollector(clock=lambda: NOW), **dict(kw, settings=s))
    assert out.status == "skipped" and "SOSOVALUE_API_KEY" in out.error


def test_sosovalue_unwraps_the_live_envelope():
    from qt.collect.sosovalue import unwrap

    rows = [{"date": "2026-11-25", "total_net_inflow": -4.5e8}]
    assert unwrap({"code": 0, "message": "success", "data": rows}) == rows  # live shape
    assert unwrap(rows) == rows  # documented shape
    for bad in ({"code": 40001, "message": "invalid key", "data": None},
                {"code": 0, "data": {"x": 1}}, "nope"):  # fmt: skip
        with pytest.raises(RuntimeError, match="unexpected response"):
            unwrap(bad)


@pytest.mark.db
def test_sosovalue_envelope_end_to_end(env):
    kw, _ = env
    day = {"date": "2026-11-24", "total_net_inflow": -5.5e7, "total_value_traded": 4.1e9,
           "total_net_assets": 5.6e10, "cum_net_inflow": 1.35e10}  # fmt: skip
    wrap = {"code": 0, "message": "success", "data": [day]}
    kw2, Col, _ = sosovalue_env(kw, {"BTC": wrap, "ETH": wrap, "SOL": wrap})
    out = run_collector(Col(clock=lambda: NOW), **kw2)
    assert out.status == "ok", out.error
    assert _q(kw2, "SELECT count(*) FROM raw_metrics WHERE source = 'sosovalue'")[0][0] == 12
    kw2["http"].close()
