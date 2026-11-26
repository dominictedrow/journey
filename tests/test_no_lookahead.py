"""No-lookahead tests for the as-of readers (component doc section 12)."""

from datetime import UTC, date, datetime, timedelta

import pytest

from qt.collect import pit
from qt.collect.base import insert_rows
from qt.common.calendar import TradingCalendar
from qt.db.pit import latest_rows, macro_asof

pytestmark = pytest.mark.db


def utc(*a):
    return datetime(*a, tzinfo=UTC)


def test_no_lookahead_macro(settings, db_conn):
    """Synthetic CPI, period 2026-09-01, three vintages (first print + two revisions)."""
    cfg = settings.config.collect.pit
    period = date(2026, 9, 1)
    vintages = [
        (date(2026, 10, 15), 310.0),  # first print
        (date(2026, 11, 13), 310.4),  # revision 1
        (date(2026, 12, 10), 310.2),  # revision 2
    ]
    rows = [
        ("CPIAUCSL", period, v, vint, pit.fred_vintage(cfg, "CPIAUCSL", vint), "alfred")
        for vint, v in vintages
    ]
    cols = ["series_id", "period", "value", "vintage", "released_at", "source"]
    assert insert_rows(db_conn, "raw_macro", cols, rows).inserted == 3

    eps = timedelta(seconds=1)
    assert macro_asof(db_conn, "CPIAUCSL", rows[0][4] - eps) == []
    for i, (vint, v) in enumerate(vintages):
        rel = rows[i][4]
        assert macro_asof(db_conn, "CPIAUCSL", rel) == [(period, v, vint)]
        if i + 1 < len(rows):
            # just before the next release, still the current vintage, never the revision
            assert macro_asof(db_conn, "CPIAUCSL", rows[i + 1][4] - eps) == [(period, v, vint)]


def test_no_lookahead_bars(settings, db_conn):
    cal = TradingCalendar(settings.config.calendar)
    aid = db_conn.execute(
        "INSERT INTO assets (ticker, asset_class) VALUES ('AAPL', 'equity') RETURNING asset_id"
    ).fetchone()[0]
    d = date(2026, 11, 24)
    bar_ts = cal.session_open(d)
    insert_rows(
        db_conn,
        "raw_prices",
        ["asset_id", "ts", "close", "source", "released_at"],
        [(aid, bar_ts, 230.0, "alpaca", pit.stock_bar(cal, d))],
    )
    before = utc(2026, 11, 24, 20, 59)  # 15:59 ET: session still open
    at_close = utc(2026, 11, 24, 21, 0)  # 16:00 ET
    decision = cal.decision_time(d)  # 16:30 ET
    assert latest_rows(db_conn, "raw_prices", before, where={"asset_id": aid}) == []
    assert len(latest_rows(db_conn, "raw_prices", at_close, where={"asset_id": aid})) == 1
    assert len(latest_rows(db_conn, "raw_prices", decision, where={"asset_id": aid})) == 1


def test_no_lookahead_news(db_conn):
    """A backdated published_at doesn't make an article visible before we first saw it."""
    seen = utc(2026, 11, 24, 15, 0)
    insert_rows(
        db_conn,
        "articles",
        ["url", "url_hash", "source", "published_at", "released_at", "title"],
        [("https://x/a", "h1", "bbc", utc(2026, 11, 20), pit.first_seen(seen), "t")],
    )
    assert latest_rows(db_conn, "articles", utc(2026, 11, 21)) == []
    assert latest_rows(db_conn, "articles", seen - timedelta(seconds=1)) == []
    assert len(latest_rows(db_conn, "articles", seen)) == 1


def test_readers_reject_naive_and_unknown_tables(db_conn):
    with pytest.raises(ValueError):
        macro_asof(db_conn, "CPIAUCSL", datetime(2026, 11, 24))
    with pytest.raises(ValueError):
        latest_rows(db_conn, "runs", utc(2026, 11, 24))
