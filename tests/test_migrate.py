import shutil

import pytest

from qt.common.paths import MIGRATIONS_DIR
from qt.db.conn import connect
from qt.db.migrate import MigrationError, discover, migrate

pytestmark = pytest.mark.db

SPEC_TABLES = {
    "assets",
    "universe_weekly",
    "raw_prices",
    "raw_macro",
    "raw_fundamentals",
    "articles",
    "article_seen",
    "llm_outputs",
    "ngram_vocab",
    "feature_registry",
    "features",
    "predictions",
    "meta_scores",
    "regime",
    "positions",
    "portfolio_daily",
    "proposed_orders",
    "orders",
    "runs",
    "risk_events",
}
EXTRA_TABLES = {"alerts_log", "backup_log", "article_scores", "schema_migrations"}
COLLECTOR_TABLES = {  # 005
    "index_membership",
    "aliases",
    "raw_events",
    "raw_metrics",
    "raw_crypto_derivs",
    "collector_state",
}
FEATURE_TABLES = {"hmm_fits"}  # 008


def test_migrate_empty_db(settings, empty_db):
    with connect(settings, "primary", empty_db, autocommit=True) as c:
        applied = migrate(c)
        assert applied == [m.filename for m in discover()]
        tables = {
            r[0]
            for r in c.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname='public'"
            ).fetchall()
        }
        views = {
            r[0]
            for r in c.execute("SELECT viewname FROM pg_views WHERE schemaname='public'").fetchall()
        }
    assert SPEC_TABLES | EXTRA_TABLES | COLLECTOR_TABLES | FEATURE_TABLES <= tables
    assert "model_input_latest" not in views  # dropped in 008 (matrix.features_asof replaces it)


def test_migrate_idempotent(settings, db):
    with connect(settings, "primary", db, autocommit=True) as c:
        assert migrate(c) == []


def test_migrate_checksum_tamper(settings, db, tmp_path):
    mdir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, mdir, ignore=shutil.ignore_patterns("._*"))
    target = sorted(mdir.glob("0*.sql"))[-1]
    target.write_text(target.read_text() + "\n-- tampered\n")
    with connect(settings, "primary", db, autocommit=True) as c:
        with pytest.raises(MigrationError, match="checksum"):
            migrate(c, mdir)


def test_bad_filename_rejected(tmp_path):
    (tmp_path / "1_bad.sql").write_text("select 1;")
    with pytest.raises(MigrationError, match="filename"):
        discover(tmp_path)


def test_hypertables(settings, db_conn):
    rows = db_conn.execute(
        "SELECT hypertable_name, column_name, time_interval "
        "FROM timescaledb_information.dimensions ORDER BY 1"
    ).fetchall()
    got = {name: (col, iv.days) for name, col, iv in rows}
    ht = settings.config.db.hypertables
    assert got == {
        "features": ("asof", ht["features"].chunk_days),
        "predictions": ("asof", ht["predictions"].chunk_days),
        "raw_prices": ("ts", ht["raw_prices"].chunk_days),
    }
    policies = dict(
        db_conn.execute(
            "SELECT hypertable_name, (config->>'compress_after')::interval "
            "FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression'"
        ).fetchall()
    )
    assert {k: v.days for k, v in policies.items()} == {
        k: v.compress_after_days for k, v in ht.items()
    }
