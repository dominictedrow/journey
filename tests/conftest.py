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


# ---------------------------------------------------------------- collector helpers
FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


class FakeAlerter:
    """Stands in for qt.common.alerts.Alerter: records pushes instead of sending them."""

    def __new__(cls, cfg):
        from qt.common import alerts

        class _Fake(alerts.Alerter):
            def _deliver(self, severity, event, title, body):
                self.sent.append((severity, title, body))
                return True

        a = _Fake(cfg, topic="t", http_post=lambda *a, **k: None, os_notify=lambda *a: True)
        a.sent = []
        return a


def with_collect(settings, **sections):
    """Copy of ``settings`` with ``collect.<section>`` fields overridden, e.g.
    ``with_collect(s, news={"feeds": {...}})``."""
    import dataclasses

    c = settings.config
    upd = {k: getattr(c.collect, k).model_copy(update=v) for k, v in sections.items()}
    collect = c.collect.model_copy(update=upd)
    return dataclasses.replace(settings, config=c.model_copy(update={"collect": collect}))


class Router:
    """httpx.MockTransport handler: canonical URL (no query) -> response or exception."""

    def __init__(self, routes):
        self.routes = dict(routes)
        self.calls = []

    def __call__(self, request):
        import httpx

        key = f"{request.url.scheme}://{request.url.host}{request.url.path}"
        self.calls.append(key)
        r = self.routes.get(key)
        if r is None:
            return httpx.Response(404)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, (bytes, str)):
            return httpx.Response(200, content=r if isinstance(r, bytes) else r.encode())
        return r
