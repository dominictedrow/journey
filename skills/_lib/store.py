"""sqlite3 (stdlib) storage — one DB file, all skills and the UI share it."""

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_cache (
    cache_key   TEXT PRIMARY KEY,
    response    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def db_path():
    path = Path.home() / "Library" / "Application Support" / "cwork" / "cwork.db"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def connect():
    path = db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    path.chmod(0o600)
    return conn
