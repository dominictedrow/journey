import psycopg2.extras
from app.db import get_conn


def save_market_data(rows):
    """rows: list of dicts with keys: series_id, metric_name, value, text_value,
    source, category, tier, recorded_at"""
    if not rows:
        return
    sql = """
        INSERT INTO market_data
            (series_id, metric_name, value, text_value, source, category, tier, recorded_at)
        VALUES
            (%(series_id)s, %(metric_name)s, %(value)s, %(text_value)s,
             %(source)s, %(category)s, %(tier)s, %(recorded_at)s)
        ON CONFLICT (series_id, recorded_at) DO UPDATE SET
            metric_name = EXCLUDED.metric_name,
            value       = EXCLUDED.value,
            text_value  = EXCLUDED.text_value,
            source      = EXCLUDED.source,
            category    = EXCLUDED.category,
            tier        = EXCLUDED.tier,
            fetched_at  = NOW()
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, rows)


def save_news(items):
    """items: list of dicts with keys: source_name, title, url, summary,
    published_at, category, tier"""
    if not items:
        return
    sql = """
        INSERT INTO news_items
            (source_name, title, url, summary, published_at, category, tier)
        VALUES
            (%(source_name)s, %(title)s, %(url)s, %(summary)s,
             %(published_at)s, %(category)s, %(tier)s)
        ON CONFLICT (url) DO NOTHING
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, items)
