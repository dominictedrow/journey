from app.db import fetchall_dict, fetchone_dict


def get_latest():
    rows = fetchall_dict("""
        SELECT DISTINCT ON (series_id)
            series_id, metric_name, value, text_value, tier, category, recorded_at, fetched_at
        FROM market_data
        ORDER BY series_id, fetched_at DESC
    """)
    return {r['series_id']: r for r in rows}


def get_latest_by_tier(tier_num):
    return fetchall_dict("""
        SELECT DISTINCT ON (series_id)
            series_id, metric_name, value, text_value, tier, category, recorded_at, fetched_at
        FROM market_data
        WHERE tier = %s
        ORDER BY series_id, fetched_at DESC
    """, (tier_num,))


def get_series_history(series_id, limit=30):
    rows = fetchall_dict("""
        SELECT value, recorded_at
        FROM market_data
        WHERE series_id = %s AND value IS NOT NULL
        ORDER BY recorded_at DESC
        LIMIT %s
    """, (series_id, limit))
    rows.reverse()
    return rows


def get_multi_series_history(series_ids, limit=80):
    """Fetch history for multiple series in one query. Returns {series_id: [row_dicts]}."""
    if not series_ids:
        return {}
    rows = fetchall_dict("""
        SELECT series_id, value, recorded_at
        FROM (
            SELECT series_id, value, recorded_at,
                   ROW_NUMBER() OVER (PARTITION BY series_id ORDER BY recorded_at DESC) AS rn
            FROM market_data
            WHERE series_id = ANY(%s) AND value IS NOT NULL
        ) t
        WHERE t.rn <= %s
        ORDER BY series_id, recorded_at ASC
    """, (series_ids, limit))
    result = {}
    for r in rows:
        result.setdefault(r['series_id'], []).append(r)
    return result


def get_multi_pct_change(series_ids, days):
    """Compute % change over N calendar days for multiple series in one query."""
    if not series_ids:
        return {}
    rows = fetchall_dict("""
        SELECT t.sid AS series_id,
            (SELECT value FROM market_data WHERE series_id = t.sid AND value IS NOT NULL
             ORDER BY fetched_at DESC LIMIT 1) AS latest,
            (SELECT value FROM market_data WHERE series_id = t.sid AND value IS NOT NULL
               AND recorded_at <= NOW() - (%s * INTERVAL '1 day')
             ORDER BY recorded_at DESC LIMIT 1) AS past
        FROM (SELECT unnest(%s) AS sid) t
    """, (days, series_ids))
    result = {}
    for r in rows:
        latest = float(r['latest']) if r['latest'] is not None else None
        past = float(r['past']) if r['past'] is not None else None
        if latest is None or past is None or past == 0:
            result[r['series_id']] = None
        else:
            result[r['series_id']] = (latest - past) / abs(past) * 100
    return result


def get_biggest_shifts(limit=10):
    return fetchall_dict("""
        WITH latest AS (
            SELECT DISTINCT ON (series_id)
                series_id, value, fetched_at
            FROM market_data
            WHERE value IS NOT NULL
            ORDER BY series_id, fetched_at DESC
        ),
        prev AS (
            SELECT DISTINCT ON (series_id)
                series_id, value AS prev_value
            FROM market_data
            WHERE value IS NOT NULL
              AND fetched_at < NOW() - INTERVAL '12 hours'
            ORDER BY series_id, fetched_at DESC
        )
        SELECT
            l.series_id,
            l.value::FLOAT8,
            p.prev_value::FLOAT8,
            CASE WHEN p.prev_value != 0
                 THEN ((l.value - p.prev_value) / ABS(p.prev_value) * 100)::FLOAT8
                 ELSE NULL END AS pct_change
        FROM latest l
        JOIN prev p ON l.series_id = p.series_id
        ORDER BY ABS(
            CASE WHEN p.prev_value != 0
                 THEN (l.value - p.prev_value) / ABS(p.prev_value) * 100
                 ELSE NULL END
        ) DESC NULLS LAST
        LIMIT %s
    """, (limit,))
