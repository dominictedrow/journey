"""Survivorship gaps (doc 03 section 9, D-043): name matching, filing check, history, bars."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from qt.collect import survivorship as sv
from qt.common.calendar import TradingCalendar

SFX = [
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "holdings",
    "holding",
    "group",
    "plc",
    "nv",
    "sa",
    "ag",
    "llc",
    "lp",
    "the",
]


@pytest.fixture
def cfg(settings):
    return settings.config.features.survivorship


def _target(first=date(2015, 12, 31), last=date(2019, 12, 5), tickers=None, **kw):
    tickers = tickers or [("STI", first, last)]
    base = dict(
        asset_id=1,
        security_key="SPX:STI:2015-12-31",
        ticker=tickers[-1][0],
        first=first,
        last=last,
        listed_until=last,
        cik=None,
        sector=None,
        sector_source=None,
        name=None,
    )
    base.update(kw)
    return sv.Target(**base, tickers=list(tickers))


def _filer(cik, first, last, every_days=91):
    dates = list(pd.date_range(first, last, freq=f"{every_days}D").date)
    return sv.Filer(cik, f"filer {cik}", [], "6021", dates)


# ---------------------------------------------------------------- names (test_cik_matching)
@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("SunTrust Banks, Inc.", "suntrust banks"),
        ("SUNTRUST BANKS INC", "suntrust banks"),
        ("Mylan N.V.", "mylan"),
        ("MYLAN INC /PA/", "mylan"),
        ("The Hershey Company", "hershey"),
        ("Johnson & Johnson", "johnson and johnson"),
        ("A. O. Smith Corp", "ao smith"),
        ("TECO Energy, Inc. (holding)", "teco energy"),
        ("Holdings Inc", "holdings"),  # never strips the whole name
        ("SVB Financial Group", "svb financial"),
    ],
)
def test_normalize_name(raw, want):
    assert sv.normalize_name(raw, SFX) == want


def test_parse_cik_lookup():
    text = "SUNTRUST BANKS INC:0000750556:\nSUNTRUST BANKS, INC.:750556:\nBAD LINE\nX:Y:\n"
    lk = sv.parse_cik_lookup(text, SFX)
    assert lk == {"suntrust banks": {"0000750556"}}


def test_filer_covers_span(cfg):
    f = _filer("1", date(2010, 1, 1), date(2019, 11, 1))
    assert f.covers(date(2015, 12, 31), date(2019, 12, 5), cfg)
    old_shell = _filer("2", date(1995, 1, 1), date(2005, 1, 1))
    assert not old_shell.covers(date(2015, 12, 31), date(2019, 12, 5), cfg)
    late_start = _filer("3", date(2017, 1, 1), date(2019, 11, 1))
    assert not late_start.covers(date(2015, 12, 31), date(2019, 12, 5), cfg)


def _lookup(*pairs):
    out: dict[str, set[str]] = {}
    for name, cik in pairs:
        out.setdefault(sv.normalize_name(name, SFX), set()).add(cik)
    return out


def test_exact_match_accepted_only_with_covering_filings(cfg):
    t = _target()
    lk = _lookup(("SUNTRUST BANKS INC", "0000750556"), ("SunTrust Banks Inc /GA/", "0000999999"))
    filers = {
        "0000750556": _filer("0000750556", date(2010, 1, 1), date(2019, 11, 1)),
        "0000999999": _filer("0000999999", date(1996, 1, 1), date(2001, 1, 1)),
    }
    m = sv.match_cik(t, ["SunTrust Banks, Inc."], lk, list(lk), filers.get, cfg)
    assert (m.status, m.cik, m.match) == ("accepted", "0000750556", "exact")


def test_exact_match_two_covering_filers_goes_to_review(cfg):
    t = _target()
    lk = _lookup(("SunTrust Banks Inc", "0000000001"), ("SunTrust Banks Corp", "0000000002"))
    filers = {
        c: _filer(c, date(2010, 1, 1), date(2019, 11, 1)) for c in ("0000000001", "0000000002")
    }
    m = sv.match_cik(t, ["SunTrust Banks"], lk, list(lk), filers.get, cfg)
    assert m.status == "review" and "2 filers" in m.reason


def test_fuzzy_match_outside_filing_span_is_rejected(cfg):
    t = _target()
    lk = _lookup(("SUNTRUST BANK INC", "0000000003"))  # one letter off: fuzzy, not exact
    filers = {"0000000003": _filer("0000000003", date(1990, 1, 1), date(2002, 1, 1))}
    m = sv.match_cik(t, ["SunTrust Banks, Inc."], lk, list(lk), filers.get, cfg)
    assert m.status == "review" and m.cik is None
    assert m.candidates and m.candidates[0][3] is False


def test_fuzzy_match_with_covering_filings_is_accepted(cfg):
    t = _target()
    lk = _lookup(("SUNTRUST BANK INC", "0000000003"), ("Totally Different Co", "0000000004"))
    filers = {"0000000003": _filer("0000000003", date(2010, 1, 1), date(2019, 11, 1))}
    m = sv.match_cik(t, ["SunTrust Banks, Inc."], lk, list(lk), filers.get, cfg)
    assert (m.status, m.cik, m.match) == ("accepted", "0000000003", "fuzzy")
    assert m.score >= cfg.fuzzy_threshold


def test_no_name_goes_to_review(cfg):
    m = sv.match_cik(_target(), [], {}, [], lambda c: None, cfg)
    assert m.status == "review"


# ---------------------------------------------------------------- wikipedia history
def _table_html(rows, symbol_col="Symbol"):
    df = pd.DataFrame(rows, columns=[symbol_col, "Security", "GICS Sector", "CIK"])
    return df.to_html(index=False)


def test_parse_constituents_old_and_new_headers():
    rows = [(f"T{i}", f"Co {i}", "Energy", 1000 + i) for i in range(5)]
    for col in ("Symbol", "Ticker symbol"):
        t = sv.parse_constituents(_table_html(rows, col), min_rows=5)
        assert t["T3"].name == "Co 3" and t["T3"].cik == "0000001003"
    assert sv.parse_constituents(_table_html(rows), min_rows=6) is None  # broken revision


class FakeWiki:
    def __init__(self, revs, tables):
        self.revs, self.tables = revs, tables  # [(ts, revid)], revid -> table

    def before(self, d):
        cut = datetime(d.year, d.month, d.day, 5, tzinfo=UTC)
        older = [r for r in self.revs if r[0] < cut]
        return older[-1] if older else None

    def table(self, revid):
        return self.tables.get(revid)


def test_find_in_wiki_steps_back_and_maps_old_sector_names(cfg):
    t = _target(tickers=[("TE", date(2015, 12, 31), date(2016, 6, 27))], last=date(2016, 6, 27))
    row = sv.WikiRow("TE", "TECO Energy", "Telecommunications Services", "0000350563")
    revs = [(datetime(2016, 5, 1, tzinfo=UTC), 10), (datetime(2016, 6, 26, tzinfo=UTC), 11)]
    # the newest revision before the last day already dropped TE (early edit); 30 days back has it
    w = FakeWiki(revs, {10: {"TE": row}, 11: {"XOM": row}})
    hit = sv.find_in_wiki(t, w, cfg)
    assert hit.revid == 10 and hit.name == "TECO Energy"
    assert hit.sector == "Communication Services"  # pre-2018 name -> current sector name


def test_find_in_wiki_none_when_never_listed(cfg):
    t = _target()
    w = FakeWiki([(datetime(2019, 1, 1, tzinfo=UTC), 1)], {1: {"XOM": None}})
    assert sv.find_in_wiki(t, w, cfg) is None


def test_changes_name_window(cfg):
    ch = pd.DataFrame(
        {
            "date": [date(2019, 12, 9), date(2010, 1, 1)],
            "ticker": ["STI", "STI"],
            "name": ["SunTrust Banks", "Old STI"],
        }
    )
    ch["key"] = ch["ticker"].map(sv.norm_ticker)
    assert sv.changes_name(_target(), ch, cfg) == "SunTrust Banks"


# ---------------------------------------------------------------- sectors
def test_sic_map_narrowest_range_wins(settings):
    m = settings.config.sic_to_gics
    assert m.sector(2834) == "Health Care"  # inside 2800-2899 Materials
    assert m.sector(3674) == "Information Technology"
    assert m.sector(3621) == "Industrials"
    assert m.sector(6798) == "Real Estate"
    assert m.sector(6021) == "Financials"
    assert m.sector(9995) is None and m.sector(None) is None and m.sector("x") is None


# ---------------------------------------------------------------- bars
@pytest.fixture(scope="module")
def cal():
    from qt.common import config

    return TradingCalendar(config.load().config.calendar)


def _bars(c, start, end):
    d = c.sessions(start, end)
    return pd.DataFrame(
        {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1e6}, index=d
    )


def test_bars_exact_span_accepted_successor_rejected(cal):
    first, end = date(2015, 12, 31), date(2019, 12, 5)
    ok, _ = sv.judge_bars(_bars(cal, date(2015, 12, 1), date(2019, 12, 4)), first, end, cal, 5)
    assert ok
    reused, _ = sv.judge_bars(_bars(cal, date(2022, 5, 2), date(2026, 11, 25)), first, end, cal, 5)
    assert not reused  # same symbol, another company later
    early_end, _ = sv.judge_bars(
        _bars(cal, date(2015, 12, 1), date(2019, 6, 1)), first, end, cal, 5
    )
    assert not early_end


def test_try_bars_writes_only_the_span(cal, cfg, settings):
    t = _target()
    seen = []

    def fake_yf(ticker, start):
        seen.append(ticker)
        return _bars(cal, date(2015, 11, 1), date(2019, 12, 5))

    class NoHttp:
        def get(self, url, **kw):  # Stooq never reached: yfinance was accepted first
            raise AssertionError(url)

    tries = sv.try_bars(t, NoHttp(), cfg, cal, settings.config.collect.pit, fake_yf)
    (b,) = tries
    assert b.status == "accepted" and b.source == "yfinance_surv" and seen == ["STI"]
    days = [r[2] for r in b.rows]
    assert min(days).date() >= t.first and len(b.rows) == len(cal.sessions(t.first, t.last))
    assert all(r[-1] > r[2] for r in b.rows)  # released after the bar's date


def test_try_bars_stooq_bot_page_is_blocked(cal, cfg, settings):
    class Resp:
        text = "<!DOCTYPE html><html>verify</html>"

    class Http:
        def get(self, url, **kw):
            return Resp()

    tries = sv.try_bars(
        _target(), Http(), cfg, cal, settings.config.collect.pit, lambda tk, s: pd.DataFrame()
    )
    assert [b.status for b in tries] == ["empty", "blocked"]


# ---------------------------------------------------------------- overrides
def test_merge_entries_keeps_manual(settings):
    from qt.common.config import CikOverride

    old = [
        CikOverride(security_key="A", ticker="A", cik="0000000001", match="manual"),
        CikOverride(security_key="B", ticker="B", cik="0000000002", match="exact"),
    ]
    new = [
        {"security_key": "A", "ticker": "A", "cik": "0000000009", "match": "exact"},
        {"security_key": "C", "ticker": "C", "cik": "0000000003", "match": "fuzzy"},
    ]
    got = sv.merge_entries(old, new, lambda e: e.match == "manual")
    assert [(e["security_key"], e["cik"]) for e in got] == [
        ("A", "0000000001"),
        ("C", "0000000003"),
    ]


def test_write_entries_keeps_header(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("# header line\nentries: []\n", encoding="utf-8")
    sv.write_entries(p, [{"security_key": "A", "cik": "1", "score": None, "manual": False}])
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# header line\n") and "score" not in text and "manual" not in text


def test_apply_overrides(db_conn, settings):
    from qt.common.config import CikOverridesCfg, SectorOverridesCfg

    c = db_conn
    ins = (
        "INSERT INTO assets (ticker, asset_class, security_key, cik, gics_sector, sector_source) "
        "VALUES (%s, 'equity', %s, %s, %s, %s)"
    )
    c.execute(ins, ("STI", "SPX:STI:2015-12-31", None, None, None))
    c.execute(ins, ("XOM", "SPX:XOM:2015-12-31", "0000034088", "Energy", "wikipedia_gics"))
    cfg = settings.config.model_copy(
        update={
            "cik_overrides": CikOverridesCfg.model_validate(
                {
                    "entries": [
                        {
                            "security_key": "SPX:STI:2015-12-31",
                            "ticker": "STI",
                            "cik": "0000750556",
                            "match": "exact",
                        },
                        {
                            "security_key": "SPX:XOM:2015-12-31",
                            "ticker": "XOM",
                            "cik": "0000000001",
                            "match": "manual",
                        },
                    ]
                }
            ),
            "sector_overrides": SectorOverridesCfg.model_validate(
                {
                    "entries": [
                        {
                            "security_key": "SPX:STI:2015-12-31",
                            "ticker": "STI",
                            "sector": "Financials",
                            "source": "wikipedia_gics_hist",
                            "name": "SunTrust Banks",
                            "sic": "6021",
                        },
                        {
                            "security_key": "SPX:XOM:2015-12-31",
                            "ticker": "XOM",
                            "sector": "Materials",
                            "source": "sic_map",
                        },
                    ]
                }
            ),
        }
    )
    n = sv.apply_overrides(c, cfg)
    assert n == {"cik": 1, "cik_conflicts": 1, "sector": 1}
    rows = dict(
        (r[0], r[1:])
        for r in c.execute(
            "SELECT ticker, cik, gics_sector, sector_source, name, sic FROM assets"
        ).fetchall()
    )
    assert rows["STI"] == (
        "0000750556",
        "Financials",
        "wikipedia_gics_hist",
        "SunTrust Banks",
        "6021",
    )
    assert rows["XOM"] == ("0000034088", "Energy", "wikipedia_gics", None, None)  # current wins
    assert sv.apply_overrides(c, cfg) == {"cik": 0, "cik_conflicts": 1, "sector": 0}  # idempotent


def test_load_targets_retargets_generated_overrides(db_conn, settings):
    """A rerun re-matches what it filled before, so the overrides file is rewritten whole."""
    from qt.common.config import CikOverridesCfg

    c = db_conn
    ins = (
        "INSERT INTO assets (ticker, asset_class, security_key, cik, gics_sector, sector_source) "
        "VALUES (%s, 'equity', %s, %s, %s, %s) RETURNING asset_id"
    )
    rows = [
        ("STI", "SPX:STI:2015-12-31", "0000750556", "Financials", "wikipedia_gics_hist"),
        ("XOM", "SPX:XOM:2015-12-31", "0000034088", "Energy", "wikipedia_gics"),
        ("GONE", "SPX:GONE:2015-12-31", "0000000005", "Energy", "wikipedia_gics"),
    ]
    for r in rows:
        aid = c.execute(ins, r).fetchone()[0]
        c.execute(
            "INSERT INTO index_membership (index_code, snapshot_at, ticker, source, asset_id,"
            " released_at) VALUES ('SPX', '2016-01-04', %s, 'fja05680', %s, '2016-01-05')",
            (r[0], aid),
        )
    cfg = settings.config.model_copy(
        update={
            "cik_overrides": CikOverridesCfg.model_validate(
                {
                    "entries": [
                        {
                            "security_key": "SPX:STI:2015-12-31",
                            "ticker": "STI",
                            "cik": "0000750556",
                            "match": "exact",
                        }
                    ]
                }
            )
        }
    )
    got = {t.security_key: (t.cik, t.sector) for t in sv.load_targets(c, cfg)}
    assert got == {"SPX:STI:2015-12-31": (None, None)}  # XOM / GONE: SEC-map CIK + current sector
