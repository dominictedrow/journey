"""DB connections. Pools pre-ping and reconnect, so they survive the Mac sleeping (news_bot)."""

from __future__ import annotations

from typing import Literal

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg_pool import ConnectionPool

from qt.common.config import ConfigError, Settings

Target = Literal["primary", "backup"]


def dsn(settings: Settings, target: Target = "primary", dbname: str | None = None) -> str:
    """Connection string for the primary or backup server, optionally for another database."""
    s, db = settings.secrets, settings.config.db
    url = s.database_url if target == "primary" else s.backup_database_url
    if url is not None:
        base = url.get_secret_value()
        extra = {"dbname": dbname} if dbname else {}
        return make_conninfo(base, connect_timeout=db.connect_timeout_s, **extra)
    if s.postgres_password is None:
        raise ConfigError("POSTGRES_PASSWORD (or DATABASE_URL) is not set; see .env.example")
    return make_conninfo(
        host=db.host,
        port=db.port if target == "primary" else db.backup_port,
        dbname=dbname or db.name,
        user=s.postgres_user,
        password=s.postgres_password.get_secret_value(),
        connect_timeout=db.connect_timeout_s,
    )


def connect(
    settings: Settings,
    target: Target = "primary",
    dbname: str | None = None,
    autocommit: bool = False,
) -> psycopg.Connection:
    return psycopg.connect(dsn(settings, target, dbname), autocommit=autocommit)


def make_pool(settings: Settings, target: Target = "primary") -> ConnectionPool:
    p = settings.config.db.pool
    return ConnectionPool(
        dsn(settings, target),
        min_size=p.min_size,
        max_size=p.max_size,
        max_idle=p.max_idle_s,
        check=ConnectionPool.check_connection,  # pre-ping on checkout
        open=True,
    )
