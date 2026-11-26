"""Standing check for reused-symbol contamination (D-035), reported in the daily summary.

Alpaca serves a symbol's history for its current holder, so a delisted S&P 500 member whose
symbol was reused can end up carrying a small company's bars. Real index members trade far more
than a few million dollars a day, so a member whose median daily dollar volume *during its
membership spells* is below the threshold is flagged for a look (and an identity override).
Bars hidden by an override (``exclude_sources``) don't count.
"""

from __future__ import annotations

from typing import Any

import psycopg

from qt.common.config import CollectCfg

QUERY = """
WITH mem AS (
    SELECT asset_id, min(snapshot_at) AS first_in, max(snapshot_at) AS last_in
    FROM index_membership
    WHERE index_code = %(index)s AND asset_id IS NOT NULL
    GROUP BY 1
), dv AS (
    SELECT p.asset_id,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY p.close * p.volume) AS med_dv,
           count(*) AS n
    FROM raw_prices p JOIN mem m USING (asset_id)
    WHERE p.timeframe = '1d' AND p.source = ANY(%(sources)s) AND p.volume > 0
      AND p.ts BETWEEN m.first_in AND m.last_in
      AND (p.asset_id, p.source) NOT IN (
          SELECT * FROM unnest(%(ex_ids)s::bigint[], %(ex_src)s::text[]))
    GROUP BY 1
)
SELECT a.ticker, a.security_key, d.med_dv / 1e6, d.n
FROM dv d JOIN assets a USING (asset_id)
WHERE d.n >= %(min_bars)s AND d.med_dv < %(min_dv)s
ORDER BY d.med_dv
"""


def suspects(conn: psycopg.Connection, c: CollectCfg) -> list[tuple[Any, ...]]:
    """Rows ``(ticker, security_key, median $M/day, bars)`` for members that look misidentified."""
    cfg = c.identity_check
    keys = {o.security_key: o.exclude_sources for o in c.identity_overrides if o.exclude_sources}
    ex_ids: list[int] = []
    ex_src: list[str] = []
    if keys:
        for key, aid in conn.execute(
            "SELECT security_key, asset_id FROM assets WHERE security_key = ANY(%s)", (list(keys),)
        ).fetchall():
            for s in keys[key]:
                ex_ids.append(aid)
                ex_src.append(s)
    rows = conn.execute(
        QUERY,
        {
            "index": c.universe.index,
            "sources": cfg.sources,
            "ex_ids": ex_ids,
            "ex_src": ex_src,
            "min_bars": cfg.min_bars,
            "min_dv": cfg.min_median_dollar_volume,
        },
    ).fetchall()
    return [(t, k, round(float(dv), 1), int(n)) for t, k, dv, n in rows]
