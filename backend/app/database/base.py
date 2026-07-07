"""Async engine, session factory, and declarative base."""

from sqlalchemy import JSON, BigInteger, Integer, MetaData
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

# Deterministic constraint names so Alembic migrations stay stable across autogenerations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# BIGINT surrogate PK that still autoincrements on SQLite (used by the test suite);
# renders as plain BIGINT on Postgres.
BigIntPK = BigInteger().with_variant(Integer(), "sqlite")

# JSONB on Postgres, generic JSON elsewhere — lets the SQLite test suite create every table.
JsonB = JSON().with_variant(JSONB(), "postgresql")


def create_engine() -> AsyncEngine:
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)


engine = create_engine()
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)
