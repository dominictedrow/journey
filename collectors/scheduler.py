"""Background scheduler — one thread per collector, isolated exception handling."""
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import resource_path

SOURCES_PATH = resource_path("config/sources.json")

# Minimum refresh interval per collector name (derived from sources.json)
def _build_intervals():
    with open(SOURCES_PATH) as f:
        series = json.load(f)["series"]
    intervals = {}
    for s in series:
        c = s["collector"]
        intervals[c] = min(intervals.get(c, 999999), s["refresh_seconds"])
    return intervals


def _run_collector(name, collect_fn, interval):
    print(f"scheduler: {name} thread started (interval={interval}s)")
    while True:
        try:
            collect_fn()
        except Exception:
            print(f"scheduler: {name} ERROR:")
            traceback.print_exc()
        time.sleep(interval)


def _run_daily_snapshot():
    """Wakes at midnight, takes portfolio snapshots and expires old news."""
    print("scheduler: portfolio-snapshot thread started")
    while True:
        now = datetime.now()
        next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0)
        time.sleep((next_midnight - now).total_seconds())
        try:
            from app.models.portfolio import take_daily_snapshot
            take_daily_snapshot("stock")
            take_daily_snapshot("crypto")
            print("scheduler: portfolio snapshots taken")
        except Exception:
            print("scheduler: portfolio-snapshot ERROR:")
            traceback.print_exc()
        try:
            from app.db import execute
            execute(
                "UPDATE news_items SET is_active = FALSE WHERE published_at < NOW() - INTERVAL '2 days'",
                []
            )
            print("scheduler: stale news expired")
        except Exception:
            print("scheduler: news-expiry ERROR:")
            traceback.print_exc()


def main():
    intervals = _build_intervals()

    # Import all collectors
    from collectors import fred, bls, prices, crypto, etf_flows, onchain, news, imf, cftc, holdings

    collectors = [
        ("fred",       fred.collect),
        ("bls",        bls.collect),
        ("prices",     prices.collect),
        ("crypto",     crypto.collect),
        ("etf_flows",  etf_flows.collect),
        ("onchain",    onchain.collect),
        ("news",       news.collect),
        ("imf",        imf.collect),
        ("cftc",       cftc.collect),
        ("holdings",   holdings.collect),
    ]

    # The pid file is only consumed by scripts/stop.sh in the browser workflow.
    # When frozen, the bundle is read-only and the process is app-managed, so skip it.
    if not getattr(sys, "frozen", False):
        pid_path = os.path.join(os.path.dirname(__file__), "..", ".scheduler.pid")
        with open(pid_path, "w") as f:
            f.write(str(os.getpid()))

    threads = []
    for i, (name, fn) in enumerate(collectors):
        interval = intervals.get(name, 3600)
        t = threading.Thread(
            target=_run_collector,
            args=(name, fn, interval),
            daemon=True,
            name=f"collector-{name}",
        )
        threads.append(t)
        t.start()
        time.sleep(2)  # 2-second stagger

    # Daily portfolio snapshot at midnight
    snap_thread = threading.Thread(
        target=_run_daily_snapshot, daemon=True, name="portfolio-snapshot"
    )
    snap_thread.start()
    threads.append(snap_thread)

    print(f"scheduler: {len(threads)} threads running ({len(threads)-1} collectors + 1 snapshot)")

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("scheduler: shutting down")


if __name__ == "__main__":
    main()
