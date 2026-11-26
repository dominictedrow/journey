"""Migration runner (D-013).

Applies ``src/qt/db/migrations/NNN_<slug>.sql`` in order, one transaction per file, recording each
file's sha256 in ``schema_migrations``. An applied file whose checksum changed is a hard error:
applied migrations are immutable, fixes are new forward migrations.

    uv run python -m qt.db.migrate [--target primary|backup] [--dbname NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

from qt.common.paths import MIGRATIONS_DIR, visible_files

_NAME = re.compile(r"^(\d{3})_[a-z0-9_]+\.sql$")
_LOCK_KEY = 0x71745F6D6967  # "qt_mig": serialize concurrent runners

_BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     int         PRIMARY KEY,
    filename    text        NOT NULL,
    sha256      text        NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
)
"""


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Migration:
    version: int
    path: Path
    sha256: str

    @property
    def filename(self) -> str:
        return self.path.name


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    found: dict[int, Migration] = {}
    for p in visible_files(directory, "*.sql"):
        m = _NAME.match(p.name)
        if not m:
            raise MigrationError(f"bad migration filename: {p.name} (want NNN_slug.sql)")
        v = int(m.group(1))
        if v in found:
            raise MigrationError(f"duplicate migration version {v:03d}")
        found[v] = Migration(v, p, hashlib.sha256(p.read_bytes()).hexdigest())
    return [found[v] for v in sorted(found)]


def migrate(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply pending migrations. Returns the filenames applied (empty if already at head)."""
    migrations = discover(directory)
    conn.autocommit = True
    conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
    try:
        conn.execute(_BOOTSTRAP)
        applied = {
            v: (f, h)
            for v, f, h in conn.execute(
                "SELECT version, filename, sha256 FROM schema_migrations"
            ).fetchall()
        }
        on_disk = {m.version for m in migrations}
        missing = sorted(set(applied) - on_disk)
        if missing:
            raise MigrationError(f"applied migrations missing from disk: {missing}")
        for m in migrations:
            if m.version in applied and applied[m.version][1] != m.sha256:
                raise MigrationError(
                    f"{m.filename} changed after it was applied (checksum mismatch). "
                    "Applied migrations are immutable: write a new forward migration."
                )

        done: list[str] = []
        for m in migrations:
            if m.version in applied:
                continue
            sql = m.path.read_text(encoding="utf-8")
            try:
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, filename, sha256) "
                        "VALUES (%s, %s, %s)",
                        (m.version, m.filename, m.sha256),
                    )
            except psycopg.Error as e:
                raise MigrationError(f"{m.filename} failed: {e}") from e
            done.append(m.filename)
        return done
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))


def main(argv: list[str] | None = None) -> int:
    from qt.common import config
    from qt.db.conn import connect

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=["primary", "backup"], default="primary")
    ap.add_argument("--dbname", default=None)
    args = ap.parse_args(argv)

    settings = config.load()
    with connect(settings, args.target, args.dbname, autocommit=True) as conn:
        done = migrate(conn)
    if done:
        print(f"applied {len(done)} migration(s): {', '.join(done)}")
    else:
        print("already at head; nothing to apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
