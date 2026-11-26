from datetime import UTC, datetime

import pytest

from qt.db import backup
from qt.db.conn import connect

pytestmark = [pytest.mark.db, pytest.mark.slow]


def test_backup_roundtrip(settings, db, backup_db_name, tmp_path):
    with connect(settings, "primary", db) as c:
        aid = c.execute(
            "INSERT INTO assets (ticker, asset_class) VALUES ('BTC', 'crypto') RETURNING asset_id"
        ).fetchone()[0]
        t = datetime(2026, 11, 24, tzinfo=UTC)
        for i in range(3):
            c.execute(
                "INSERT INTO raw_prices (asset_id, ts, close, source, released_at) "
                "VALUES (%s, %s + make_interval(days => %s), 1, 'alpaca', %s)",
                (aid, t, i, t),
            )

    # Pre-existing old dumps to exercise pruning.
    for d in range(1, 10):
        (tmp_path / f"qt_202611{d:02d}.dump").write_bytes(b"old")

    res = backup.run_backup(
        settings,
        src_db=db,
        dst_db=backup_db_name,
        dump_dir=tmp_path,
        now=datetime(2026, 11, 26, 2, 0, tzinfo=UTC),  # 21:00 ET on the 25th
    )
    assert res.ok, res.mismatches
    assert res.row_counts["raw_prices"] == 3 and res.row_counts["assets"] == 1
    assert res.dump_file.exists() and res.dump_bytes > 0
    kept = sorted(p.name for p in tmp_path.glob("qt_*.dump"))
    assert len(kept) == settings.config.backup.keep_dumps and kept[-1] == "qt_20261125.dump"

    with connect(settings, "backup", backup_db_name) as b:
        assert b.execute("SELECT count(*) FROM raw_prices").fetchone()[0] == 3
        hts = {
            r[0]
            for r in b.execute(
                "SELECT hypertable_name FROM timescaledb_information.hypertables"
            ).fetchall()
        }
        assert hts == {"raw_prices", "features", "predictions"}


def test_refuses_same_cluster(settings, db, tmp_path, monkeypatch):
    # Point "backup" at the primary port: the system_identifier guard must stop the drop.
    same = settings.config.db.model_copy(update={"backup_port": settings.config.db.port})
    cfg = settings.config.model_copy(update={"db": same})
    secrets = settings.secrets.model_copy(update={"backup_database_url": None})
    bad = type(settings)(settings.profile, cfg, secrets, settings.config_hash)
    if secrets.postgres_password is None:
        pytest.skip("needs POSTGRES_PASSWORD")
    dump_file = tmp_path / "x.dump"
    dump_file.write_bytes(b"")
    with pytest.raises(backup.BackupError, match="SAME Postgres cluster"):
        backup.restore(bad, dump_file, db)
    with connect(settings, "primary", db) as c:  # primary DB untouched
        assert c.execute("SELECT 1 FROM schema_migrations LIMIT 1").fetchone()


def test_prune_ignores_appledouble(tmp_path):
    for d in range(1, 4):
        (tmp_path / f"qt_202611{d:02d}.dump").write_bytes(b"x")
    (tmp_path / "._qt_20261101.dump").write_bytes(b"x")
    doomed = backup.prune(tmp_path, keep=2)
    assert [p.name for p in doomed] == ["qt_20261101.dump"]
