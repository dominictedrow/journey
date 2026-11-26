"""Shared fixtures. DB tests run against throwaway databases and skip if Docker isn't up."""

from __future__ import annotations

import secrets
from collections.abc import Iterator

import psycopg
import pytest
from psycopg import sql

from qt.common import config
from qt.db.conn import connect
from qt.db.migrate import migrate

# exFAT: macOS drops ._* AppleDouble twins next to every file; never collect them.
collect_ignore_glob = ["._*"]


@pytest.fixture(scope="session")
def settings() -> config.Settings:
    return config.load(profile="dev")


def _reachable(settings: config.Settings, target: str) -> bool:
    try:
        with connect(settings, target, "postgres", autocommit=True):  # type: ignore[arg-type]
            return True
    except Exception:
        return False


def _create(settings, target, name):
    with connect(settings, target, "postgres", autocommit=True) as c:
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def _drop(settings, target, name):
    with connect(settings, target, "postgres", autocommit=True) as c:
        c.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def empty_db(settings) -> Iterator[str]:
    """Name of a fresh, empty database on the primary server."""
    if not _reachable(settings, "primary"):
        pytest.skip("primary DB not reachable: run `docker compose up -d`")
    name = "qt_test_" + secrets.token_hex(4)
    _create(settings, "primary", name)
    try:
        yield name
    finally:
        _drop(settings, "primary", name)


@pytest.fixture
def db(settings, empty_db) -> str:
    """Name of a fresh database on the primary, migrated to head."""
    with connect(settings, "primary", empty_db, autocommit=True) as c:
        migrate(c)
    return empty_db


@pytest.fixture
def db_conn(settings, db) -> Iterator[psycopg.Connection]:
    with connect(settings, "primary", db, autocommit=True) as c:
        yield c


@pytest.fixture
def backup_db_name(settings) -> Iterator[str]:
    """Name to use on the BACKUP server; dropped afterwards."""
    if not _reachable(settings, "backup"):
        pytest.skip("backup DB not reachable: run `docker compose up -d`")
    name = "qt_test_" + secrets.token_hex(4)
    try:
        yield name
    finally:
        _drop(settings, "backup", name)
