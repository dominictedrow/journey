"""Alpaca assets + daily bars collectors against a fake Alpaca (no network)."""

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

from conftest import with_collect
from qt.collect.alpaca import Bar
from qt.collect.assets import AlpacaAssetsCollector, CryptoAssetsCollector, crypto_pool
from qt.collect.bars import BarsCryptoCollector, BarsEquityCollector
from qt.collect.base import run_collector
from qt.common.calendar import ET, TradingCalendar
from qt.db.conn import connect
from qt.db.pit import latest_rows

D = date.fromisoformat


def et(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=ET).astimezone(UTC)


class FakeAlpaca:
    def __init__(self, cal):
        self.cal = cal
        self.calls = []
        self.ids = {s: str(uuid.uuid4()) for s in ["AAA", "CA", "SPY", "BTC/USD", "ETH/USD",
                                                   "USDC/USD", "BTC/USDT", "XYZ/USD"]}  # fmt: skip
        self.crypto_active = {"BTC/USD", "ETH/USD", "USDC/USD", "BTC/USDT"}

    def _eq(self, sym, a, b):
        out = []
        for d in self.cal.sessions(a, min(b, D("2026-11-27"))):
            vol = 0.0 if sym == "CA" and d > D("2026-11-10") else 1000.0  # filler after delisting
            out.append(Bar(et(d.year, d.month, d.day), 10, 11, 9, 10.5, vol, 10.2, 5))
        return out

    def stock_bars(self, sym, a, b=None, adjustment="raw"):
        self.calls.append(("one", sym, a))
        return self._eq(sym, a, b or D("2026-11-27"))

    def stock_bars_multi(self, syms, a, b=None, adjustment="raw"):
        self.calls.append(("multi", tuple(syms), a))
        return {s: self._eq(s, a, b or D("2026-11-27")) for s in syms}

    def crypto_bars(self, syms, a, b=None):
        self.calls.append(("crypto", tuple(syms), a))
        days = [a + timedelta(days=i) for i in range((D("2026-11-25") - a).days + 1)]
        return {
            s: [Bar(datetime(d.year, d.month, d.day, tzinfo=UTC), 1, 2, 0.5, 1.5, 7, 1.2, 3)
                for d in days]
            for s in syms
        }  # fmt: skip

    def assets(self, asset_class, status="active"):
        if asset_class == "crypto":
            return [
                {"symbol": s, "id": self.ids[s], "name": s, "status": "active",
                 "tradable": s != "XYZ/USD"}
                for s in sorted(self.crypto_active | {"XYZ/USD"})
            ]  # fmt: skip
        return [
            {"symbol": s, "id": self.ids[s], "name": f"{s} Inc", "tradable": True}
            for s in ("AAA", "SPY")
        ]


@pytest.fixture
def env(settings, db):
    s = with_collect(
        settings,
        alpaca={"prices_start": "2026-11-02", "crypto_start": "2026-11-20"},
        universe={"extra_etfs": ["SPY", "NOPE"]},
    )
    cal = TradingCalendar(s.config.calendar)
    fake = FakeAlpaca(cal)
    kw = dict(settings=s, connect=lambda: connect(s, "primary", db), http=None, cal=cal)
    with kw["connect"]() as c:  # members as the membership collector leaves them
        c.execute(
            "INSERT INTO assets (ticker, asset_class, alpaca_id, alpaca_tradable, security_key) "
            "VALUES ('AAA', 'equity', %s, false, 'SPX:AAA:2015-12-31')",
            (fake.ids["AAA"],),
        )
        c.execute(
            "INSERT INTO assets (ticker, asset_class, security_key, listed_until) "
            "VALUES ('CA', 'equity', 'SPX:CA:2015-12-31', '2026-11-10')"
        )
    return kw, fake


def _q(kw, sql, *a):
    with kw["connect"]() as c:
        return c.execute(sql, a).fetchall()


# ---------------------------------------------------------------- assets
def test_crypto_pool_rules(settings):
    fake = FakeAlpaca(TradingCalendar())
    pool = crypto_pool(fake.assets("crypto"), settings.config.collect.crypto)
    assert [a["symbol"] for a in pool] == ["BTC/USD", "ETH/USD"]  # no stablecoin/USDT/untradable


@pytest.mark.db
def test_alpaca_assets_etfs_and_tradable(env):
    kw, fake = env
    out = run_collector(AlpacaAssetsCollector(alpaca=lambda ctx: fake), **kw)
    assert out.status == "ok", out.error
    rows = dict(_q(kw, "SELECT ticker, alpaca_tradable FROM assets"))
    assert rows == {"AAA": True, "CA": False, "SPY": True}  # NOPE isn't on Alpaca: warned


@pytest.mark.db
def test_crypto_assets_pool_and_drop(env):
    kw, fake = env
    run_collector(CryptoAssetsCollector(alpaca=lambda ctx: fake), **kw)
    fake.crypto_active.discard("ETH/USD")
    out = run_collector(CryptoAssetsCollector(alpaca=lambda ctx: fake), **kw)
    assert out.result.cursor["pool"] == ["BTC/USD"]
    rows = dict(_q(kw, "SELECT ticker, alpaca_tradable FROM assets WHERE asset_class='crypto'"))
    assert rows == {"BTC/USD": True, "ETH/USD": False}


# ---------------------------------------------------------------- bars
@pytest.mark.db
def test_bars_equity_backfill_then_daily(env):
    kw, fake = env
    run_collector(AlpacaAssetsCollector(alpaca=lambda ctx: fake), **kw)
    t1 = et(2026, 11, 24, 15, 0)  # Tue 15:00 ET: today's bar is still in progress
    out = run_collector(BarsEquityCollector(clock=lambda: t1, alpaca=lambda ctx: fake), **kw)
    assert out.status == "ok", out.error
    n = dict(_q(kw, "SELECT a.ticker, count(*) FROM raw_prices p JOIN assets a USING (asset_id) "
                    "GROUP BY 1"))  # fmt: skip
    sessions_to_23 = len(kw["cal"].sessions(D("2026-11-02"), D("2026-11-23")))
    assert n["AAA"] == n["SPY"] == sessions_to_23  # 11-24 skipped: not closed yet
    assert n["CA"] == len(kw["cal"].sessions(D("2026-11-02"), D("2026-11-10")))  # to listed_until
    assert {c[0] for c in fake.calls} == {"one"}  # history: one symbol per request

    fake.calls.clear()
    t2 = et(2026, 11, 27, 13, 30)  # Fri after the 13:00 early close
    out = run_collector(BarsEquityCollector(clock=lambda: t2, alpaca=lambda ctx: fake), **kw)
    assert out.result.rows == {"raw_prices": 2 * 3}  # 11-24, 11-25, 11-27 for AAA and SPY
    assert fake.calls == [("multi", ("AAA", "SPY"), D("2026-11-24"))]  # batched; CA not asked
    rel = _q(kw, "SELECT released_at FROM raw_prices p JOIN assets a USING (asset_id) "
                 "WHERE a.ticker = 'AAA' AND ts = %s", et(2026, 11, 27))  # fmt: skip
    assert rel == [(et(2026, 11, 27, 13, 0),)]  # early close

    out = run_collector(BarsEquityCollector(clock=lambda: t2, alpaca=lambda ctx: fake), **kw)
    assert out.result.total == 0  # idempotent


@pytest.mark.db
def test_bars_no_lookahead(env):
    kw, fake = env
    run_collector(AlpacaAssetsCollector(alpaca=lambda ctx: fake), **kw)
    run_collector(BarsEquityCollector(clock=lambda: et(2026, 11, 25, 20), alpaca=lambda ctx: fake),
                  **kw)  # fmt: skip
    with kw["connect"]() as c:
        before = latest_rows(c, "raw_prices", et(2026, 11, 24, 15, 59), columns=["ts"])
        after = latest_rows(c, "raw_prices", et(2026, 11, 24, 16, 0), columns=["ts"])
    assert et(2026, 11, 24) not in {r["ts"] for r in before}
    assert et(2026, 11, 24) in {r["ts"] for r in after}


@pytest.mark.db
def test_bars_crypto(env):
    kw, fake = env
    run_collector(CryptoAssetsCollector(alpaca=lambda ctx: fake), **kw)
    t = datetime(2026, 11, 25, 0, 15, tzinfo=UTC)
    out = run_collector(BarsCryptoCollector(clock=lambda: t, alpaca=lambda ctx: fake), **kw)
    assert out.status == "ok", out.error
    assert out.result.rows == {"raw_prices": 2 * 5}  # 11-20..11-24; the 11-25 bar is in progress
    rel = _q(kw, "SELECT min(released_at), max(released_at) FROM raw_prices")
    assert rel == [(datetime(2026, 11, 21, tzinfo=UTC), datetime(2026, 11, 25, tzinfo=UTC))]
    out = run_collector(BarsCryptoCollector(clock=lambda: t, alpaca=lambda ctx: fake), **kw)
    assert out.result.total == 0
