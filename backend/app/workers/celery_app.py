"""Celery application. Worker and beat both load this module.

Beat schedule entries are added per phase (e.g. markets sync in Phase 1,
weekly optimizer in Phase 6).
"""

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "copytrader",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.workers.tasks.system",
        "app.workers.tasks.ingest",
        "app.workers.tasks.discovery",
        "app.workers.tasks.analytics",
        "app.workers.tasks.ranking",
        "app.workers.tasks.backtests",
        "app.workers.tasks.copy",
        "app.workers.tasks.strategies",
    ],
)

celery_app.conf.timezone = "UTC"
celery_app.conf.beat_schedule = {
    "sync-markets-hourly": {"task": "ingest.sync_markets", "schedule": 3600.0},
    # 60s (was 30s): the only RPC-hitting beat — halves QuickNode credit drip on the free tier.
    # Still well under copy_max_trade_age_seconds (300s) so live copies keep firing.
    "chain-live-tail": {"task": "chain.live_tail", "schedule": 60.0},
    "discovery-sweep-6h": {"task": "discovery.run", "schedule": 21600.0},
    "refresh-metrics-hourly": {"task": "analytics.refresh_all_metrics", "schedule": 3600.0},
    "rebuild-rankings-hourly": {"task": "ranking.rebuild", "schedule": 3600.0},
    "copy-detect-15s": {"task": "copy.detect_and_copy", "schedule": 15.0},
    "revalue-portfolio-60s": {"task": "copy.revalue_portfolio", "schedule": 60.0},
    "reconcile-live-120s": {"task": "copy.reconcile_live", "schedule": 120.0},
    "optimize-strategies-weekly": {"task": "strategies.optimize", "schedule": 604800.0},
}
