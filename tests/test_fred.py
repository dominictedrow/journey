"""FRED/ALFRED + HY archive collectors against a fake FRED API (no network).

The fake reproduces the two API behaviours the collector relies on (live probes 2026-11-25):
realtime_start is clipped to the requested start, and a realtime range with more vintage dates
than the limit is an HTTP 400.
"""

import dataclasses
from datetime import UTC, date, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr

from conftest import with_collect
from qt.collect.base import run_collector
from qt.collect.fred import FredCollector, HyArchiveCollector, Obs, new_vintages, windows
from qt.collect.http import HttpClient
from qt.common.calendar import TradingCalendar
from qt.db.conn import connect
from qt.db.pit import macro_asof

D = date.fromisoformat
END = D("9999-12-31")
TODAY = D("2026-11-25")


class FakeFred:
    """``series[sid]`` = ALFRED truth: [(period, realtime_start, realtime_end, value), ...]."""

    def __init__(self, series, max_vintages, today=TODAY):
        self.series, self.max, self.today = series, max_vintages, today
        self.calls = []
        self.fail = set()
        self.extra_count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        q = dict(request.url.params)
        sid, ep = q["series_id"], request.url.path.rsplit("/", 1)[-1]
        self.calls.append((ep, sid, q.get("realtime_start"), q.get("realtime_end")))
        if sid in self.fail:
            return httpx.Response(500)
        segs = self.series[sid]
        a = D(q.get("realtime_start", self.today.isoformat()))
        b = D(q.get("realtime_end", self.today.isoformat()))
        vint = sorted({s[1] for s in segs})
        if ep == "vintagedates":
            v = [x.isoformat() for x in vint if x >= a]
            return httpx.Response(200, json={"count": len(v), "vintage_dates": v})
        # like FRED, count one more vintage than vintagedates lists (the one in force at start)
        if len([x for x in vint if a <= x <= b]) + self.extra_count > self.max:
            return httpx.Response(400, json={"error_message": "exceeds the maximum"})
        rows = [
            {"date": p.isoformat(), "realtime_start": max(rs, a).isoformat(),
             "realtime_end": min(re, b).isoformat(), "value": "." if v is None else str(v)}
            for p, rs, re, v in sorted(segs) if rs <= b and re >= a
        ]  # fmt: skip
        off, lim = int(q.get("offset", 0)), int(q.get("limit", 100000))
        return httpx.Response(200, json={"count": len(rows), "observations": rows[off : off + lim]})


def cpi_truth():
    return [
        (D("2026-01-01"), D("2026-02-11"), D("2026-03-10"), 100.0),
        (D("2026-01-01"), D("2026-03-11"), END, 101.0),  # revised
        (D("2026-02-01"), D("2026-03-11"), END, 102.0),
        (D("2026-03-01"), D("2026-04-10"), END, 103.0),
    ]


@pytest.fixture
def env(settings, db):
    s = with_collect(settings, fred={"max_vintages_per_request": 2, "page_limit": 2})
    s = dataclasses.replace(
        s, secrets=s.secrets.model_copy(update={"fred_api_key": SecretStr("test-key")})
    )
    fake = FakeFred(
        {
            "CPIAUCSL": cpi_truth(),
            "VIXCLS": [(D("2026-11-20"), D("2026-11-20"), END, 20.5)],
            "BAMLH0A0HYM2": [
                (D("2026-11-23"), D("2026-11-23"), END, 2.90),
                (D("2026-11-24"), D("2026-11-24"), END, 2.95),
            ],
            "BADSER": [],
        },
        max_vintages=2,
    )
    http = HttpClient(
        s.config.collect.http, transport=httpx.MockTransport(fake), sleep=lambda _: None
    )
    kw = dict(
        settings=s,
        connect=lambda: connect(s, "primary", db),
        http=http,
        cal=TradingCalendar(s.config.calendar),
    )
    yield kw, fake
    http.close()


def _rows(kw, sid, source="alfred"):
    with kw["connect"]() as c:
        return c.execute(
            "SELECT period, vintage, value, released_at FROM raw_macro "
            "WHERE series_id = %s AND source = %s ORDER BY period, vintage",
            (sid, source),
        ).fetchall()


# ---------------------------------------------------------------- pure helpers
def test_windows_start_on_vintage_dates():
    v = [D("2026-01-05"), D("2026-02-05"), D("2026-03-05"), D("2026-04-05"), D("2026-05-05")]
    assert windows(v, 2) == [
        ("2026-01-05", "2026-03-04"), ("2026-03-05", "2026-05-04"), ("2026-05-05", "9999-12-31"),
    ]  # fmt: skip
    assert windows(v[:1], 2000) == [("2026-01-05", "9999-12-31")]


def test_new_vintages_drops_clipped_continuations():
    p = D("2026-01-01")
    held = {p: (D("2026-03-11"), 101.0)}
    obs = [
        Obs(p, D("2026-04-10"), 101.0),  # clipped: same value, stamped with the request start
        Obs(p, D("2026-05-12"), 99.0),  # a real revision
        Obs(D("2026-02-01"), D("2026-04-10"), 102.0),  # new period
        Obs(p, D("2026-03-11"), 101.0),  # already held
    ]
    out = new_vintages(obs, held)
    assert [(o.period, o.realtime_start, o.value) for o in out] == [
        (D("2026-02-01"), D("2026-04-10"), 102.0),
        (p, D("2026-05-12"), 99.0),
    ]
    assert held[p] == (D("2026-05-12"), 99.0)


# ---------------------------------------------------------------- ALFRED
@pytest.mark.db
def test_fred_backfill_windows_and_released_at(env):
    kw, fake = env
    out = run_collector(FredCollector(series=["CPIAUCSL", "VIXCLS"]), **kw)
    assert out.status == "ok", out.error
    cpi = _rows(kw, "CPIAUCSL")
    assert [(r[0], r[1], r[2]) for r in cpi] == [
        (D("2026-01-01"), D("2026-02-11"), 100.0),
        (D("2026-01-01"), D("2026-03-11"), 101.0),
        (D("2026-02-01"), D("2026-03-11"), 102.0),
        (D("2026-03-01"), D("2026-04-10"), 103.0),
    ]  # the window starting 2026-04-10 re-sent P1/P2 clipped: not stored again
    assert cpi[1][3] == datetime(2026, 3, 11, 12, 30, tzinfo=UTC)  # 08:30 EDT
    windows_used = [c for c in fake.calls if c[0] == "observations" and c[1] == "CPIAUCSL"]
    assert {(c[2], c[3]) for c in windows_used} == {
        ("2026-02-11", "2026-04-09"), ("2026-04-10", "9999-12-31"),
    }  # fmt: skip
    # VIXCLS for Fri 11-20 is stamped 11-20 but released Mon 11-23 12:00 ET (D-029)
    assert _rows(kw, "VIXCLS")[0][3] == datetime(2026, 11, 23, 17, 0, tzinfo=UTC)


@pytest.mark.db
def test_fred_incremental_and_idempotent(env):
    kw, fake = env
    c = FredCollector(series=["CPIAUCSL"])
    run_collector(c, **kw)
    fake.series["CPIAUCSL"] = cpi_truth()[:3] + [
        (D("2026-03-01"), D("2026-04-10"), D("2026-05-11"), 103.0),
        (D("2026-03-01"), D("2026-05-12"), END, 104.0),
        (D("2026-04-01"), D("2026-05-12"), END, 105.0),
    ]
    out = run_collector(c, **kw)
    assert out.result.rows == {"raw_macro": 2}
    assert out.result.cursor["series"]["CPIAUCSL"]["last_vintage"] == "2026-05-12"
    out = run_collector(c, **kw)
    assert out.status == "ok" and out.result.total == 0

    # point in time: the 104 revision is invisible until 2026-05-12 08:30 ET
    with kw["connect"]() as cx:
        before = dict(
            (p, v)
            for p, v, _ in macro_asof(cx, "CPIAUCSL", datetime(2026, 5, 12, 12, 29, tzinfo=UTC))
        )
        after = dict(
            (p, v)
            for p, v, _ in macro_asof(cx, "CPIAUCSL", datetime(2026, 5, 12, 12, 30, tzinfo=UTC))
        )
    assert before[D("2026-03-01")] == 103.0 and D("2026-04-01") not in before
    assert after[D("2026-03-01")] == 104.0 and after[D("2026-04-01")] == 105.0


@pytest.mark.db
def test_fred_one_series_failing_is_isolated(env):
    kw, fake = env
    fake.fail.add("BADSER")
    out = run_collector(FredCollector(series=["BADSER", "CPIAUCSL"]), **kw)
    assert out.status == "failed" and "BADSER" in out.error
    assert len(_rows(kw, "CPIAUCSL")) == 4  # the healthy series still landed


@pytest.mark.db
def test_fred_missing_key_skips(env):
    kw, _ = env
    s = kw["settings"]
    kw = dict(
        kw,
        settings=dataclasses.replace(
            s, secrets=s.secrets.model_copy(update={"fred_api_key": None})
        ),
    )
    assert run_collector(FredCollector(series=["CPIAUCSL"]), **kw).status == "skipped"


# ---------------------------------------------------------------- HY archive
@pytest.mark.db
def test_hy_archive(env):
    kw, fake = env
    t0 = datetime(2026, 11, 24, 21, 30, tzinfo=UTC)  # 16:30 ET decision run
    fake.today = D("2026-11-24")
    # an ALFRED row with the same (period, vintage) must not block the archive row (migration 006)
    with kw["connect"]() as c:
        c.execute(
            "INSERT INTO raw_macro (series_id, period, value, vintage, released_at, source) "
            "VALUES ('BAMLH0A0HYM2', '2026-11-23', 2.90, '2026-11-24', %s, 'alfred')",
            (t0,),
        )
    out = run_collector(HyArchiveCollector(clock=lambda: t0), **kw)
    assert out.status == "ok" and out.result.rows == {"raw_macro": 2}
    rows = _rows(kw, "BAMLH0A0HYM2", "fred_archive")
    assert {(r[1], r[3]) for r in rows} == {(D("2026-11-24"), t0)}  # vintage = fetch date ET

    t1 = t0 + timedelta(days=1)
    out = run_collector(HyArchiveCollector(clock=lambda: t1), **kw)
    assert out.result.total == 0  # unchanged values are not re-stored

    fake.series["BAMLH0A0HYM2"][1] = (D("2026-11-24"), D("2026-11-24"), END, 3.01)  # revised
    out = run_collector(HyArchiveCollector(clock=lambda: t1), **kw)
    assert out.result.rows == {"raw_macro": 1}
    assert [r[2] for r in _rows(kw, "BAMLH0A0HYM2", "fred_archive")] == [2.90, 2.95, 3.01]


@pytest.mark.db
def test_fred_window_400_splits(env):
    kw, fake = env
    fake.extra_count = 1  # a 2-vintage window now reads as 3 > 2: HTTP 400 -> split in half
    out = run_collector(FredCollector(series=["CPIAUCSL"]), **kw)
    assert out.status == "ok", out.error
    assert len(_rows(kw, "CPIAUCSL")) == 4
    obs = [c for c in fake.calls if c[0] == "observations"]
    assert ("2026-02-11", "2026-03-10") in {(c[2], c[3]) for c in obs}  # the split half
