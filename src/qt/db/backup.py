"""Evening backup (D-017): dump primary -> refresh the separate backup DB -> verify row counts.

1. ``pg_dump -Fc`` inside ``qt_db`` -> ``data/backups/qt_YYYYMMDD.dump`` (fine on exFAT)
2. In ``qt_db_backup``: drop + recreate the DB, ``timescaledb_pre_restore()``, ``pg_restore``,
   ``timescaledb_post_restore()``. Every refresh doubles as a test restore.
3. Exact ``count(*)`` per table on both sides; any mismatch is an ERROR (-> critical push).
4. Keep the newest ``keep_dumps`` dump files.
5. ``backup_log`` row on the primary + an info push on success.

The backup server must be a *different* Postgres cluster from the primary (checked via
``pg_control_system().system_identifier``) before anything is dropped.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import psycopg
from psycopg import sql

from qt.common import alerts
from qt.common.calendar import ET
from qt.common.config import Settings
from qt.common.paths import REPO_ROOT, repo_path
from qt.db import conn as dbconn

log = logging.getLogger("qt.backup")

_DUMP_GLOB = "qt_*.dump"


class BackupError(RuntimeError):
    pass


@dataclass
class BackupResult:
    dump_file: Path
    dump_bytes: int
    row_counts: dict[str, int]
    mismatches: dict[str, dict[str, int]] = field(default_factory=dict)
    pruned: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches


def _compose(service: str, *cmd: str, stdin=None, stdout=None) -> subprocess.CompletedProcess:
    args = ["docker", "compose", "exec", "-T", service, *cmd]
    return subprocess.run(
        args, cwd=REPO_ROOT, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, check=False
    )


def _system_id(conn: psycopg.Connection) -> int:
    return conn.execute("SELECT system_identifier FROM pg_control_system()").fetchone()[0]


def table_counts(conn: psycopg.Connection) -> dict[str, int]:
    names = [
        r[0]
        for r in conn.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY 1"
        ).fetchall()
    ]
    return {
        n: conn.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(n))).fetchone()[0]
        for n in names
    }


def dump(settings: Settings, src_db: str, dump_file: Path) -> int:
    dump_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = dump_file.with_name(dump_file.name + ".partial")
    user = settings.secrets.postgres_user
    with tmp.open("wb") as f:
        r = _compose("db", "pg_dump", "-U", user, "-d", src_db, "-Fc", stdout=f)
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise BackupError(f"pg_dump failed: {r.stderr.decode(errors='replace').strip()}")
    tmp.replace(dump_file)
    return dump_file.stat().st_size


def restore(settings: Settings, dump_file: Path, dst_db: str) -> None:
    with (
        dbconn.connect(settings, "primary", "postgres", autocommit=True) as p,
        dbconn.connect(settings, "backup", "postgres", autocommit=True) as b,
    ):
        if _system_id(p) == _system_id(b):
            raise BackupError(
                "backup target is the SAME Postgres cluster as the primary; refusing to drop "
                "anything. Check QT_DB_BACKUP_PORT / BACKUP_DATABASE_URL."
            )
        b.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dst_db)))
        b.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dst_db)))

    with dbconn.connect(settings, "backup", dst_db, autocommit=True) as b:
        b.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
        b.execute("SELECT timescaledb_pre_restore()")

    user = settings.secrets.postgres_user
    with dump_file.open("rb") as f:
        r = _compose(
            "db_backup",
            "pg_restore",
            "-U",
            user,
            "-d",
            dst_db,
            "--no-owner",
            stdin=f,
        )
    stderr = r.stderr.decode(errors="replace")

    with dbconn.connect(settings, "backup", dst_db, autocommit=True) as b:
        b.execute("SELECT timescaledb_post_restore()")

    if r.returncode != 0:
        # pg_restore keeps going on errors and exits 1. Real damage shows up in the row-count
        # check, but surface the messages either way.
        log.warning("pg_restore_warnings", extra={"stderr": stderr.strip()[-2000:]})


def prune(dump_dir: Path, keep: int) -> list[Path]:
    dumps = sorted(p for p in dump_dir.glob(_DUMP_GLOB) if not p.name.startswith("._"))
    doomed = dumps[:-keep] if len(dumps) > keep else []
    for p in doomed:
        p.unlink(missing_ok=True)
    return doomed


def run_backup(
    settings: Settings,
    src_db: str | None = None,
    dst_db: str | None = None,
    dump_dir: Path | None = None,
    now: datetime | None = None,
) -> BackupResult:
    src_db = src_db or settings.config.db.name
    dst_db = dst_db or settings.config.db.name
    dump_dir = dump_dir or repo_path(settings.config.backup.dump_dir)
    now = now or datetime.now(UTC)
    # Named by the ET trading day: the ~21:00 ET run is already the next day in UTC.
    dump_file = dump_dir / f"qt_{now.astimezone(ET):%Y%m%d}.dump"

    size = dump(settings, src_db, dump_file)
    log.info("backup_dumped", extra={"file": dump_file.name, "bytes": size})
    restore(settings, dump_file, dst_db)

    with dbconn.connect(settings, "primary", src_db) as p:
        src_counts = table_counts(p)
    with dbconn.connect(settings, "backup", dst_db) as b:
        dst_counts = table_counts(b)
    mismatches = {
        t: {"primary": src_counts.get(t, -1), "backup": dst_counts.get(t, -1)}
        for t in sorted(set(src_counts) | set(dst_counts))
        if src_counts.get(t) != dst_counts.get(t)
    }
    pruned = prune(dump_dir, settings.config.backup.keep_dumps)
    return BackupResult(dump_file, size, src_counts, mismatches, pruned)


def _record(settings: Settings, run_id: str, res: BackupResult | None, error: str | None) -> None:
    try:
        with dbconn.connect(settings) as p:
            p.execute(
                "INSERT INTO backup_log (run_id, dump_file, dump_bytes, status, row_counts, "
                "mismatches, error) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    run_id,
                    res.dump_file.name if res else None,
                    res.dump_bytes if res else None,
                    "ok" if res and res.ok and not error else "failed",
                    json.dumps(res.row_counts if res else {}),
                    json.dumps(res.mismatches if res else {}),
                    error,
                ),
            )
    except Exception:
        log.exception("backup_log_write_failed")


def backup_job(settings: Settings, connect) -> bool:
    """One backup under ``run_context`` (used by the CLI and the scheduler). Never raises."""
    from qt.common.runs import run_context

    try:
        with run_context("backup", settings, connect) as run:
            res: BackupResult | None = None
            try:
                res = run_backup(settings)
            except Exception as e:
                _record(settings, run.run_id, None, f"{type(e).__name__}: {e}")
                raise
            if not res.ok:
                _record(settings, run.run_id, res, "row-count mismatch")
                raise BackupError(f"backup row-count mismatch: {res.mismatches}")
            _record(settings, run.run_id, res, None)
            total = sum(res.row_counts.values())
            log.info("backup_ok", extra={"tables": len(res.row_counts), "rows": total})
            alerts.alert(
                "info",
                "backup_ok",
                "QT backup OK",
                f"{res.dump_file.name} ({res.dump_bytes / 1e6:.1f} MB), "
                f"{len(res.row_counts)} tables / {total:,} rows match.",
            )
    except Exception:
        return False
    return True


def main() -> int:
    from qt.common import config
    from qt.common import logging as qtlog

    settings = config.load()
    connect = lambda: dbconn.connect(settings)  # noqa: E731
    qtlog.setup(settings, connect=connect)
    return 0 if backup_job(settings, connect) else 1


if __name__ == "__main__":
    sys.exit(main())
