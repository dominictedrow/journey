from datetime import UTC, date, datetime

import psycopg
import pytest

pytestmark = pytest.mark.db

T0 = datetime(2026, 11, 24, 21, 0, tzinfo=UTC)


def _seed(c):
    aid = c.execute(
        "INSERT INTO assets (ticker, asset_class) VALUES ('SPY', 'etf') RETURNING asset_id"
    ).fetchone()[0]
    c.execute(
        "INSERT INTO raw_prices (asset_id, ts, close, source, released_at) "
        "VALUES (%s, %s, 600, 'alpaca', %s)",
        (aid, T0, T0),
    )
    c.execute(
        "INSERT INTO raw_macro (series_id, period, value, vintage, released_at) "
        "VALUES ('BAMLH0A0HYM2', %s, 3.1, %s, %s)",
        (date(2026, 11, 24), date(2026, 11, 25), T0),
    )
    c.execute(
        "INSERT INTO raw_fundamentals (asset_id, fiscal_period, metric, value, accession, "
        "filed_at, released_at) VALUES (%s, '2026Q3', 'Revenues', 1, 'acc-1', %s, %s)",
        (aid, T0, T0),
    )
    c.execute(
        "INSERT INTO articles (url, url_hash, source, released_at) VALUES ('u', 'h', 'bbc', %s)",
        (T0,),
    )
    c.execute("INSERT INTO article_seen (url_hash) VALUES ('h')")
    return aid


MUTATIONS = {
    "raw_prices": "UPDATE raw_prices SET close = 1",
    "raw_macro": "UPDATE raw_macro SET value = 1",
    "raw_fundamentals": "UPDATE raw_fundamentals SET value = 2",
    "articles": "UPDATE articles SET title = 'x'",
    "article_seen": "UPDATE article_seen SET simhash = 1",
}


@pytest.mark.parametrize("table", sorted(MUTATIONS))
def test_update_delete_truncate_blocked(db_conn, table):
    _seed(db_conn)
    for stmt in (MUTATIONS[table], f"DELETE FROM {table}", f"TRUNCATE {table} CASCADE"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
            db_conn.execute(stmt)
    assert db_conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 1


def test_duplicate_insert_is_noop(db_conn):
    aid = _seed(db_conn)
    cur = db_conn.execute(
        "INSERT INTO raw_prices (asset_id, ts, close, source, released_at) "
        "VALUES (%s, %s, 999, 'alpaca', %s) ON CONFLICT DO NOTHING",
        (aid, T0, T0),
    )
    assert cur.rowcount == 0
    assert db_conn.execute("SELECT close FROM raw_prices").fetchone()[0] == 600


def test_market_feature_unique_with_null_asset(db_conn):
    ins = (
        "INSERT INTO features (asof, asset_id, feature_name, feature_version, value) "
        "VALUES (%s, NULL, 'vix', 1, 20) ON CONFLICT DO NOTHING"
    )
    assert db_conn.execute(ins, (T0,)).rowcount == 1
    assert db_conn.execute(ins, (T0,)).rowcount == 0
