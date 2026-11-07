"""Native macOS window for the Market Research dashboard (browser-independent).

Boots Flask + the collector scheduler in-process, then opens one pywebview
window backed by macOS's system WebKit. Closing the window exits the process,
and the daemon threads die with it.
"""
import os
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# --- Environment (mirror scripts/start.sh) ---
# Dev run: project .env next to this file. Frozen .app: __file__ lives inside the
# read-only bundle, so read keys from ~/Library/Application Support/Market Research/.env.
if getattr(sys, "frozen", False):
    _env_path = os.path.expanduser(
        "~/Library/Application Support/Market Research/.env"
    )
else:
    _env_path = os.path.join(ROOT, ".env")

if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("DATABASE_URL", "postgresql://localhost/market_research")

PORT = 5001
URL = f"http://127.0.0.1:{PORT}/"


def _wait_for_server(timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(URL, timeout=1)
            return
        except Exception:
            time.sleep(0.25)


def main():
    from db.init import init_db
    init_db()

    # Import shared third-party deps and build the app on the main thread before
    # spawning threads — concurrent first-imports of requests race and yield a
    # half-initialized module (NameError: name 'sys' is not defined).
    import requests  # noqa: F401
    import bs4  # noqa: F401
    from collectors import scheduler
    from app import create_app
    app = create_app()

    threading.Thread(target=scheduler.main, daemon=True, name="scheduler").start()
    threading.Thread(
        target=lambda: app.run(port=PORT, threaded=True, use_reloader=False),
        daemon=True,
        name="flask",
    ).start()

    _wait_for_server()

    import webview
    webview.create_window("Market Research", URL, width=1400, height=900)
    webview.start()


if __name__ == "__main__":
    main()
