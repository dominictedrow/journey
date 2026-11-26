"""Point-in-time readers (component doc section 6.3). Phase 3 features read raw data only here.

The rule is one line: a reader for decision time T sees a row only if ``released_at <= T``.
Every function takes ``asof`` explicitly and refuses naive datetimes; nothing reads the clock.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd
import psycopg
from psycopg import sql

# Raw tables that carry released_at. Table names are whitelisted, never interpolated from input.
PIT_TABLES = frozenset(
    {
        "raw_prices",
        "raw_macro",
        "raw_fundamentals",
        "articles",
        "index_membership",
        "raw_events",
        "raw_metrics",
        "raw_crypto_derivs",
    }
)


def _aware(asof: datetime) -> datetime:
    if not isinstance(asof, datetime) or asof.tzinfo is None or asof.utcoffset() is None:
        raise ValueError("asof must be a timezone-aware datetime")
    return asof


def macro_asof(
    conn: psycopg.Connection,
    series_id: str,
    asof: datetime,
    *,
    source: str | None = None,
    start: date | None = None,
) -> list[tuple[date, float | None, date]]:
    """Per period, the latest vintage known at ``asof``: ``[(period, value, vintage), ...]``.

    A later revision of a period stays invisible until its own ``released_at``.
    """
    q = sql.SQL(
        """
        SELECT DISTINCT ON (period) period, value, vintage
        FROM raw_macro
        WHERE series_id = %s AND released_at <= %s {src} {start}
        ORDER BY period, vintage DESC, released_at DESC
        """
    ).format(
        src=sql.SQL("AND source = %s") if source else sql.SQL(""),
        start=sql.SQL("AND period >= %s") if start else sql.SQL(""),
    )
    params: list[Any] = [series_id, _aware(asof)]
    if source:
        params.append(source)
    if start:
        params.append(start)
    return [(r[0], r[1], r[2]) for r in conn.execute(q, params).fetchall()]


def latest_rows(
    conn: psycopg.Connection,
    table: str,
    asof: datetime,
    *,
    where: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None,
    order_by: Sequence[str] = ("released_at",),
) -> list[dict[str, Any]]:
    """All rows of ``table`` with ``released_at <= asof`` (and equality filters in ``where``)."""
    if table not in PIT_TABLES:
        raise ValueError(f"{table!r} is not a point-in-time table")
    cols = sql.SQL(", ").join(map(sql.Identifier, columns)) if columns else sql.SQL("*")
    conds = [sql.SQL("released_at <= %s")]
    params: list[Any] = [_aware(asof)]
    for k, v in (where or {}).items():
        conds.append(sql.SQL("{} = %s").format(sql.Identifier(k)))
        params.append(v)
    q = sql.SQL("SELECT {} FROM {} WHERE {} ORDER BY {}").format(
        cols,
        sql.Identifier(table),
        sql.SQL(" AND ").join(conds),
        sql.SQL(", ").join(map(sql.Identifier, order_by)),
    )
    cur = conn.execute(q, params)
    names = [d.name for d in cur.description or []]
    return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


def frame_asof(
    conn: psycopg.Connection,
    table: str,
    asof: datetime,
    *,
    where: Mapping[str, Any] | None = None,
    where_in: Mapping[str, Sequence[Any]] | None = None,
    columns: Sequence[str] | None = None,
    order_by: Sequence[str] = ("released_at",),
) -> pd.DataFrame:
    """Bulk form of :func:`latest_rows` as a DataFrame, with ``IN`` filters for many keys.

    Same rule: only rows with ``released_at <= asof``. Phase 3 loads a whole history once with the
    run's last decision time, then cuts each earlier decision time in memory on ``released_at``.
    """
    if table not in PIT_TABLES:
        raise ValueError(f"{table!r} is not a point-in-time table")
    cols = sql.SQL(", ").join(map(sql.Identifier, columns)) if columns else sql.SQL("*")
    conds = [sql.SQL("released_at <= %s")]
    params: list[Any] = [_aware(asof)]
    for k, v in (where or {}).items():
        conds.append(sql.SQL("{} = %s").format(sql.Identifier(k)))
        params.append(v)
    for k, vs in (where_in or {}).items():
        conds.append(sql.SQL("{} = ANY(%s)").format(sql.Identifier(k)))
        params.append(list(vs))
    q = sql.SQL("SELECT {} FROM {} WHERE {} ORDER BY {}").format(
        cols,
        sql.Identifier(table),
        sql.SQL(" AND ").join(conds),
        sql.SQL(", ").join(map(sql.Identifier, order_by)),
    )
    cur = conn.execute(q, params)
    names = [d.name for d in cur.description or []]
    return pd.DataFrame(cur.fetchall(), columns=names)
