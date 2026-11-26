"""Migration 005: the new raw tables are append-only; collector_state and aliases are mutable."""

from datetime import UTC, datetime

import psycopg
import pytest

pytestmark = pytest.mark.db

T0 = datetime(2026, 11, 24, 21, 0, tzinfo=UTC)

SEED = {
    "index_membership": (
        "INSERT INTO index_membership (index_code, snapshot_at, ticker, source, released_at) "
        "VALUES ('SPX', %(t)s, 'AAPL', 'wikipedia', %(t)s)",
        "UPDATE index_membership SET ticker = 'X'",
    ),
    "raw_events": (
        "INSERT INTO raw_events (source, external_id, content_hash, kind, released_at) "
        "VALUES ('fed', 'id1', 'h', 'fomc_statement', %(t)s)",
        "UPDATE raw_events SET title = 'x'",
    ),
    "raw_metrics": (
        "INSERT INTO raw_metrics (source, metric, ts, value, released_at) "
        "VALUES ('coingecko', 'btc_dominance', %(t)s, 57.1, %(t)s)",
        "UPDATE raw_metrics SET value = 1",
    ),
    "raw_crypto_derivs": (
        "INSERT INTO raw_crypto_derivs (venue, symbol, ts, kind, funding_rate, released_at) "
        "VALUES ('okx', 'BTC', %(t)s, 'funding', 0.0001, %(t)s)",
        "UPDATE raw_crypto_derivs SET funding_rate = 1",
    ),
}


@pytest.mark.parametrize("table", sorted(SEED))
def test_append_only_new_tables(db_conn, table):
    ins, upd = SEED[table]
    db_conn.execute(ins, {"t": T0})
    for stmt in (upd, f"DELETE FROM {table}", f"TRUNCATE {table}"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
            db_conn.execute(stmt)
    assert db_conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 1


@pytest.mark.parametrize("table", sorted(SEED))
def test_released_at_not_null(db_conn, table):
    ins, _ = SEED[table]
    with pytest.raises(psycopg.errors.NotNullViolation):
        db_conn.execute(ins.replace("%(t)s)", "NULL)"), {"t": T0})


def test_collector_state_mutable(db_conn):
    db_conn.execute("INSERT INTO collector_state (collector, last_status) VALUES ('news', 'ok')")
    db_conn.execute("UPDATE collector_state SET last_status = 'failed'")
    assert db_conn.execute("SELECT last_status FROM collector_state").fetchone()[0] == "failed"


def test_articles_new_columns(db_conn):
    db_conn.execute(
        "INSERT INTO articles (url, url_hash, source, released_at, word_count, extract_status, "
        "guid, title_hash) VALUES ('u', 'h', 'bbc', %s, 120, 'short', 'g', 'th')",
        (T0,),
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        db_conn.execute(
            "INSERT INTO articles (url, url_hash, source, released_at, extract_status) "
            "VALUES ('u2', 'h2', 'bbc', %s, 'bogus')",
            (T0,),
        )
