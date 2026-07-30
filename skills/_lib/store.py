"""sqlite3 (stdlib) storage — one DB file, all skills and the UI share it."""

import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claude_cache (
    cache_key   TEXT PRIMARY KEY,
    response    TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS skill_runs (
    skill_id    TEXT PRIMARY KEY,
    envelope    TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    ran_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS classes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS assignments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    class_id    INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
    title       TEXT NOT NULL,
    due_date    TEXT NOT NULL,
    done        INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS delivered_items (
    item_hash   TEXT PRIMARY KEY,
    skill       TEXT NOT NULL,
    delivered_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS release_items (
    category      TEXT NOT NULL,
    item          TEXT NOT NULL,
    date          TEXT,
    confidence    TEXT,
    source_url    TEXT,
    checked_month TEXT,
    PRIMARY KEY (category, item)
);
"""
# `skill_runs`/`classes`/`assignments` are also created (idempotently) by
# app/db.js, since the Electron main process talks to this same file directly
# via node:sqlite rather than round-tripping through Python for plain reads/CRUD.


def db_path():
    path = Path.home() / "Library" / "Application Support" / "cwork" / "cwork.db"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def connect():
    path = db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    path.chmod(0o600)
    return conn
