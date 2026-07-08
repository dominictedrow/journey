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
        "app.workers.tasks.autonomy",
        "app.workers.tasks.redemption",
    ],
)

celery_app.conf.timezone = "UTC"
celery_app.conf.beat_schedule = {
    "sync-markets-hourly": {"task": "ingest.sync_markets", "schedule": 3600.0},
    # 15s: end-to-end copy lag must clear the 60s TimingRule gate. Worst case is now
    # ~5 confirmations (~10s) + 15s tail + 15s detect ≈ 40s; at the old 60s cadence most
    # copies arrived >60s stale and were denied. RPC cost stays trivial on Alchemy's free
    # tier: ~5760 runs/day x (eth_blockNumber 10 CU + ~1 eth_getLogs 75 CU + a get_block
    # per fill-bearing block) ≈ 15M CU/month vs a 100M+ monthly allowance. If a QuickNode
    # free URL is ever primary again, drop this back to 60s.
    "chain-live-tail": {"task": "chain.live_tail", "schedule": 15.0},
    # Tightened from 6h: bare-address sweeps (no backfill) are cheap, and the "constantly
    # searching" autonomy goal wants a wallet pool that grows faster than once a quarter-day.
    "discovery-sweep-2h": {"task": "discovery.run", "schedule": 7200.0},
    # Budgeted discovery→analysis bridge: track + backfill + score a few new wallets each
    # hour, so rankings grow autonomously without bursting the Data-API.
    "discovery-autopromote-hourly": {"task": "discovery.autopromote", "schedule": 3600.0},
    "refresh-metrics-hourly": {"task": "analytics.refresh_all_metrics", "schedule": 3600.0},
    "rebuild-rankings-hourly": {"task": "ranking.rebuild", "schedule": 3600.0},
    "copy-detect-15s": {"task": "copy.detect_and_copy", "schedule": 15.0},
    "revalue-portfolio-60s": {"task": "copy.revalue_portfolio", "schedule": 60.0},
    # Live equity must be snapshotted too, or `total_equity` stays 0 forever in live mode:
    # the exposure-cap % and the loss-limit auto-kill-switch both read from these snapshots.
    # No-ops cheaply (see revalue_portfolio_task) on a box that isn't actually armed live.
    "revalue-portfolio-live-60s": {
        "task": "copy.revalue_portfolio",
        "schedule": 60.0,
        "kwargs": {"mode": "live"},
    },
    # Close positions in resolved markets at true $0/$1 payout so the equity curve (and the
    # loss limits and sizing built on it) tracks reality, not the last stale trade print.
    # Targeted Gamma fetch of only the held markets — a request or two per run.
    "settle-resolutions-10m": {"task": "copy.settle_resolutions", "schedule": 600.0},
    # Separate from settle-resolutions-10m so on-chain confirmation latency never blocks the
    # internal-books settlement sweep (Module E). No-ops cheaply on a paper-only deployment.
    "redeem-settled-positions-10m": {
        "task": "copy.redeem_settled_positions",
        "schedule": 600.0,
    },
    "reconcile-live-120s": {"task": "copy.reconcile_live", "schedule": 120.0},
    # Tightened from weekly: optimizer_min_improvement's 10% adoption hurdle already prevents
    # strategy thrashing, so daily re-evaluation reacts to fresh rankings/backtests without
    # any added flip-flop risk.
    "optimize-strategies-daily": {"task": "strategies.optimize", "schedule": 86400.0},
    # Enforcement half of the optimizer loop: converge wallet copy flags back to the adopted
    # strategy (rotations, restored DBs, accidental manual drift) without waiting a week.
    "apply-strategy-hourly": {"task": "strategies.apply_active", "schedule": 3600.0},
    # Module D: widen the wallet funnel autonomously when trailing 24h trade activity has
    # gone quiet, so the ≥1 trade/day floor holds without a human touching a dial.
    "activity-governor-6h": {"task": "autonomy.activity_governor", "schedule": 21600.0},
}
