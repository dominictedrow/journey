"""S&P 500 membership: security identity across renames, delistings and ticker reuse (no network).

The fake Alpaca reproduces the probed behaviours (2026-11-25): an old symbol serves the old
company's bars for old dates (FB = Facebook to 2022-06-08), a renamed security has identical
bars under both symbols, and a delisted symbol keeps returning zero-volume filler bars until it
is reused (CA).
"""

import uuid
from datetime import UTC, date, datetime, timedelta

import pytest

from conftest import Router, with_collect
from qt.collect.alpaca import Bar
from qt.collect.base import run_collector
from qt.collect.http import HttpClient
from qt.collect.membership import (
    Identity,
    MembershipCollector,
    Sessions,
    Snapshot,
    continuous_end,
    fja_snapshots,
    listings,
    parse_fja,
    resolve,
)
from qt.common.calendar import ET, TradingCalendar
from qt.db.conn import connect

D = date.fromisoformat
TODAY = D("2026-11-25")
FJA = """date,tickers
2015-06-01,"AAA,FB,OLD,CA,GONE"
2015-12-31,"AAA,FB,OLD,CA"
2016-03-01,"AAA,FB,OLD,NEW1"
2022-06-09,"AAA,META,OLD,NEW1"
2023-01-03,"AAA,META,NEW1"
2026-12-01,"AAA,META,NEW1,LATE"
"""


def wiki_html(symbols):
    rows = "".join(
        f"<tr><td>{s}</td><td>{s} Corp</td><td>Tech</td><td>x</td><td>x</td><td>2000-01-01</td>"
        f"<td>{1000 + i}</td><td>1900</td></tr>"
        for i, s in enumerate(symbols)
    )
    head = "".join(
        f"<th>{h}</th>"
        for h in ["Symbol", "Security", "GICS Sector", "GICS Sub-Industry",
                  "Headquarters Location", "Date added", "CIK", "Founded"]
    )  # fmt: skip
    return f'<table id="constituents"><tr>{head}</tr>{rows}</table>'


class FakeAlpaca:
    """Bars by (symbol, session); prices encode the security so identity is checkable."""

    ETF_FROM, CA_DEAD, CA_REUSE = D("2025-01-02"), D("2016-03-15"), D("2026-01-02")

    def __init__(self, cal):
        self.sessions = cal.sessions(D("2015-01-02"), TODAY)
        self.calls = []
        self.ids = {s: str(uuid.uuid4()) for s in ["AAA", "META", "OLD", "NEW1", "CA", "FB"]}

    @staticmethod
    def _bar(d, price, vol=1000.0):
        t = datetime(d.year, d.month, d.day, tzinfo=ET).astimezone(UTC)
        return Bar(t, price, price + 1, price - 1, price, vol, price, 10)

    def _series(self, sym, d):
        fb = 100 + d.toordinal() % 97  # "Facebook"
        if sym in ("AAA", "OLD"):
            return self._bar(d, (7 if sym == "AAA" else 9) + d.toordinal() % 13)
        if sym == "META":
            return self._bar(d, fb)
        if sym == "FB":
            if d <= D("2022-06-08"):
                return self._bar(d, fb)
            return self._bar(d, 25.0) if d >= self.ETF_FROM else None  # reused by an ETF
        if sym == "NEW1":
            return self._bar(d, 50 + d.toordinal() % 7) if d >= D("2016-02-01") else None
        if sym == "CA":
            if d <= self.CA_DEAD:
                return self._bar(d, 44 + d.toordinal() % 3)
            if d < self.CA_REUSE:
                return self._bar(d, 44.44, vol=0.0)  # filler after delisting
            return self._bar(d, 25.0 + d.toordinal() % 5)
        if sym == "GONE":
            return self._bar(d, 3.0) if d <= D("2015-08-01") else None
        return None

    def stock_bars(self, sym, a, b, adjustment="raw"):
        self.calls.append((sym, a, b))
        return [x for d in self.sessions if a <= d <= b if (x := self._series(sym, d))]

    def assets(self, asset_class, status="active"):
        return [
            {"symbol": s, "id": i, "name": f"{s} Inc", "tradable": True}
            for s, i in self.ids.items()
        ]


@pytest.fixture(scope="module")
def cal(settings):
    return TradingCalendar(settings.config.calendar)


def _snaps(cal):
    fja = fja_snapshots(parse_fja(FJA), D("2016-01-01"), TODAY)
    now = datetime(2026, 11, 25, 15, tzinfo=UTC)
    return fja + [Snapshot(TODAY, frozenset({"AAA", "META", "NEW1", "CA"}), "wikipedia", now)]


def _ident(settings, cal, fake):
    return Identity(
        bars=fake.stock_bars,
        ss=Sessions(cal, D("2015-01-02"), TODAY),
        cfg=settings.config.collect.membership,
        today=TODAY,
    )


# ---------------------------------------------------------------- pure
def test_fja_window_and_wikipedia_cutoff():
    snaps = fja_snapshots(parse_fja(FJA), D("2016-01-01"), TODAY)
    # starts with the list in force on 2016-01-01; rows from our first Wikipedia day on are dropped
    assert [s.on for s in snaps] == [D("2015-12-31"), D("2016-03-01"), D("2022-06-09"),
                                     D("2023-01-03")]  # fmt: skip
    assert snaps[0].at == datetime(2015, 12, 31, 5, tzinfo=UTC)  # 00:00 ET


def test_listings_spells():
    ls = {(x.ticker, x.first): x for x in listings(_snaps(TradingCalendar()))}
    assert ls[("FB", D("2015-12-31"))].removed_on == D("2022-06-09")
    assert ls[("CA", D("2015-12-31"))].removed_on == D("2016-03-01")
    assert ls[("CA", TODAY)].removed_on is None  # second spell: still a member


def test_continuous_end_ignores_zero_volume_filler(settings, cal):
    fake = FakeAlpaca(cal)
    ss = Sessions(cal, D("2015-01-02"), TODAY)
    bars = fake.stock_bars("CA", D("2015-12-31"), D("2026-03-01"))
    assert continuous_end(bars, D("2015-12-31"), ss, 10) == FakeAlpaca.CA_DEAD


def test_resolve_renames_reuse_and_delistings(settings, cal):
    fake = FakeAlpaca(cal)
    secs = {s.key: s for s in resolve(_snaps(cal), _ident(settings, cal, fake))}
    assert set(secs) == {
        "SPX:AAA:2015-12-31", "SPX:FB:2015-12-31", "SPX:OLD:2015-12-31", "SPX:CA:2015-12-31",
        "SPX:NEW1:2016-03-01", "SPX:CA:2026-11-25",
    }  # fmt: skip
    meta = secs["SPX:FB:2015-12-31"]
    assert [x.ticker for x in meta.listings] == ["FB", "META"] and meta.ticker == "META"
    assert meta.current and meta.listed_until is None
    old_ca, new_ca = secs["SPX:CA:2015-12-31"], secs["SPX:CA:2026-11-25"]
    assert old_ca.listed_until == FakeAlpaca.CA_DEAD  # reuse split + filler ignored
    assert new_ca.current
    old = secs["SPX:OLD:2015-12-31"]
    assert not old.current and old.listed_until is None  # left the index, still trades


def test_identity_cache_skips_requests(settings, cal):
    fake = FakeAlpaca(cal)
    ident = _ident(settings, cal, fake)
    resolve(_snaps(cal), ident)
    n = len(fake.calls)
    again = Identity(bars=fake.stock_bars, ss=ident.ss, cfg=ident.cfg, today=TODAY,
                     cache=dict(ident.cache))  # fmt: skip
    resolve(_snaps(cal), again)
    # renames, re-entries and final ends are cached; only "still trading" (OLD) is re-checked
    assert [c[0] for c in fake.calls[n:]] == ["OLD"]


# ---------------------------------------------------------------- collector
@pytest.fixture
def env(settings, db, cal, tmp_path):
    s = with_collect(
        settings,
        membership={"wikipedia_min_rows": 1, "crosscheck_report": str(tmp_path / "cc.md")},
    )
    m = s.config.collect.membership
    fja_key = m.fja_url.replace("%26", "&").replace("%20", " ")  # Router keys are decoded paths
    wiki_key = m.wikipedia_url.replace("%26", "&")
    router = Router({fja_key: FJA.encode(), wiki_key: wiki_html(["AAA", "META", "NEW1", "CA"])})
    import httpx

    http = HttpClient(s.config.collect.http, transport=httpx.MockTransport(router),
                      sleep=lambda _: None)  # fmt: skip
    fake = FakeAlpaca(cal)
    kw = dict(settings=s, connect=lambda: connect(s, "primary", db), http=http, cal=cal)
    yield kw, fake, router, tmp_path
    http.close()


@pytest.mark.db
def test_membership_collect_end_to_end(env):
    kw, fake, router, tmp = env
    t0 = datetime(2026, 11, 25, 15, tzinfo=UTC)
    c = MembershipCollector(clock=lambda: t0, alpaca=lambda ctx: fake)
    out = run_collector(c, **kw)
    assert out.status == "ok", out.error
    assert out.result.rows == {"index_membership": 15 + 4}
    with kw["connect"]() as cx:
        assets = {
            r[0]: r[1:]
            for r in cx.execute(
                "SELECT security_key, ticker, alpaca_id::text, listed_until, cik, alpaca_tradable "
                "FROM assets"
            ).fetchall()
        }
        fb_2016 = cx.execute(
            "SELECT a.security_key FROM index_membership m JOIN assets a USING (asset_id) "
            "WHERE m.ticker = 'FB' AND m.snapshot_at < '2017-01-01' LIMIT 1"
        ).fetchone()[0]
    assert len(assets) == 6
    assert assets["SPX:FB:2015-12-31"][:2] == ("META", fake.ids["META"])  # not the FB ETF
    assert fb_2016 == "SPX:FB:2015-12-31"
    assert assets["SPX:CA:2015-12-31"][1:3] == (None, FakeAlpaca.CA_DEAD)
    assert assets["SPX:CA:2026-11-25"][1] == fake.ids["CA"]
    assert assets["SPX:CA:2026-11-25"][3] == f"{1003:010d}"  # CIK from Wikipedia, zero-padded
    assert assets["SPX:OLD:2015-12-31"][1] == fake.ids["OLD"]
    cc = (tmp / "cc.md").read_text()
    assert "Only in fja05680 (0)" in cc and "Only in Wikipedia (1): CA" in cc

    # next week: only the new Wikipedia snapshot is added; asset rows are stable
    out = run_collector(MembershipCollector(clock=lambda: t0 + timedelta(days=7),
                                            alpaca=lambda ctx: fake), **kw)  # fmt: skip
    assert out.status == "ok" and out.result.rows == {"index_membership": 4}
    with kw["connect"]() as cx:
        assert cx.execute("SELECT count(*) FROM assets").fetchone()[0] == 6
