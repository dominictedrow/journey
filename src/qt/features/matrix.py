"""Read the stored features back as model inputs (component doc 03 section 8).

:func:`features_asof` replaces the old ``model_input_latest`` view: for a decision time T and a
pinned ``(feature, version)`` set it returns the wide market vector and asset matrix. Each value is
the newest row with ``asof <= T`` within ``lookback`` (stocks carry Friday's values into the
weekend, coins have their own weekend rows). Missing stays NaN; nothing is filled here.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import psycopg


@dataclass(frozen=True)
class Matrix:
    asof: pd.Timestamp
    market: pd.Series  # index: feature name (or "name@vN" when two versions are requested)
    assets: pd.DataFrame  # index: asset_id; columns as above


def _label(name: str, ver: int, multi: set[str]) -> str:
    return f"{name}@v{ver}" if name in multi else name


def features_asof(
    conn: psycopg.Connection,
    asof: datetime,
    versions: Iterable[tuple[str, int]],
    *,
    lookback: timedelta = timedelta(days=5),
) -> Matrix:
    keys = sorted(set(versions))
    names = [k[0] for k in keys]
    multi = {n for n in names if names.count(n) > 1}
    t = pd.Timestamp(asof)
    rows = conn.execute(
        "SELECT DISTINCT ON (asset_id, feature_name, feature_version) asset_id, feature_name, "
        "feature_version, value FROM features "
        "WHERE asof <= %s AND asof > %s "
        "AND (feature_name, feature_version) IN (SELECT * FROM unnest(%s::text[], %s::int[])) "
        "ORDER BY asset_id, feature_name, feature_version, asof DESC",
        (t, t - lookback, names, [k[1] for k in keys]),
    ).fetchall()
    df = pd.DataFrame(rows, columns=["asset_id", "name", "ver", "value"])
    df["col"] = [_label(n, int(v), multi) for n, v in zip(df["name"], df["ver"], strict=True)]
    df["value"] = df["value"].astype(float)
    cols = [_label(n, v, multi) for n, v in keys]
    mk = df[df["asset_id"].isna()].set_index("col")["value"]
    a = df[df["asset_id"].notna()]
    assets = (a.pivot(index="asset_id", columns="col", values="value") if not a.empty
              else pd.DataFrame(columns=cols))  # fmt: skip
    assets.index = assets.index.astype("int64")
    assets.columns.name = None
    return Matrix(asof=t, market=mk, assets=assets)
