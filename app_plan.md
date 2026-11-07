# Plan: Package Market Research as a standalone macOS `.app` (browser-independent)

## Context

The app is a local Flask + PostgreSQL dashboard. A prior plan made it an **installable PWA**,
and all of that work is implemented (manifest, service worker, icons, `/sw.js` + `/offline`
route, `base.html` wiring). **However, a PWA is always browser-backed** — "installing" it via
Chrome creates a Chrome app; via Safari it's a WebKit/Safari web app. No browser → no app. That
is incompatible with the requirement:

> A Mac application **completely independent of a browser** aside from data fetching.

So the remaining work is a different architecture: wrap the existing Flask server in a native
desktop window.

**Confirmed decisions:**
- **Engine = `pywebview`** — renders in macOS's built-in WebKit framework (a system component,
  *not* the Safari app). No Chrome/Safari needed. Pure-Python, fits the Flask stack.
- **Server lifecycle = app-managed** — launching the `.app` starts Flask + the collector
  scheduler in-process; quitting stops them. Self-contained; the LaunchAgent is retired.

The existing PWA layer is harmless and stays (it's just markup the window also loads); it is no
longer the mechanism that makes this "an app."

**External prerequisite (out of scope, document only):** a local PostgreSQL must be running
(e.g. Postgres.app / `brew services start postgresql`). The `.app` connects to it via
`DATABASE_URL` (default `postgresql://localhost/market_research`, mirroring `scripts/start.sh:14`).

---

## What we add

A `pywebview` entry point that boots the server in-process and opens one native window, plus a
PyInstaller build that bundles it (with templates/static/config/schema and an `.icns` icon) into
`dist/Market Research.app`.

### 1. Entry point — `desktop.py` (new, project root)
Runs entirely on launch; `webview.start()` blocks the main thread (required on macOS) and returns
on window close, at which point the process exits (daemon threads die with it).

- **Env:** load `.env` from the project dir if present (dev) else from
  `~/Library/Application Support/Market Research/.env` (frozen); set `DATABASE_URL` default to
  `postgresql://localhost/market_research` if unset (mirror `scripts/start.sh:14`).
- **DB init:** call `init_db()` (see step 2) — idempotent via the existing `.initialized` flag.
- **Scheduler:** run `collectors.scheduler.main()` in a `daemon=True` thread. `main()` already
  spins up per-collector daemon threads (`collectors/scheduler.py:63`); wrapping it in a daemon
  thread means it dies cleanly when the window closes.
- **Flask:** run `create_app().run(port=5001, threaded=True, use_reloader=False)` in a
  `daemon=True` thread. Poll `http://127.0.0.1:5001/` until it answers (or short sleep).
- **Window:** `webview.create_window("Market Research", "http://127.0.0.1:5001/", width=1400,
  height=900)` then `webview.start()`.
- Port **5001** fixed (5000 is macOS AirPlay).

### 2. Make DB init importable — `db/init.py` (modify)
Wrap the current module-level body (`db/init.py:6-26`) in `def init_db():` (keep the
`.initialized` flag check and schema execution), and keep a `if __name__ == "__main__": init_db()`
guard so `scripts/start.sh:16` still works unchanged. `desktop.py` imports `init_db`.

### 3. Resolve bundled resources when frozen — `app/__init__.py` (modify)
`Flask(__name__)` finds `templates/` and `static/` relative to the package, which breaks under
PyInstaller. Add a small `resource_path(rel)` helper that returns `sys._MEIPASS/rel` when
`getattr(sys, "frozen", False)` else the normal path, and pass explicit
`template_folder` / `static_folder` to the `Flask(...)` constructor (`app/__init__.py:6`).
`config/sources.json` and `db/schema.sql` reads (in `collectors/scheduler.py:12`, `db/init.py:15`)
should resolve through the same helper when frozen.

### 4. Build script — `scripts/build_app.sh` (new) + `MarketResearch.spec` (new)
- **Icon:** generate `build/AppIcon.icns` from `app/static/icons/icon-512.png` by creating an
  `.iconset` (via `sips` to the standard sizes) then `iconutil -c icns`. Native tools, no new dep.
- **PyInstaller** (`--windowed`, `--name "Market Research"`, `--icon build/AppIcon.icns`) with
  `--add-data` for `app/templates`, `app/static`, `config/sources.json`, `db/schema.sql`, and
  `--hidden-import` for the dynamically-imported collectors (`fred,bls,prices,crypto,etf_flows,
  onchain,news,imf,cftc` — imported by name in `collectors/scheduler.py:67`) and `psycopg2`.
  Output: `dist/Market Research.app`. Capture the flags in `MarketResearch.spec` for repeatable
  builds.

### 5. Dependencies — `requirements.txt` (modify) + `requirements-build.txt` (new)
- Add `pywebview` to `requirements.txt` (runtime; pulls `pyobjc` on macOS).
- `pyinstaller` goes in a new `requirements-build.txt` (build-time only).
- **Note the CLAUDE.md "4 packages only" rule:** this is a deliberate, user-approved expansion
  for the desktop wrapper. Document the exception in `CLAUDE.md` so the constraint stays honest.

### 6. Retire the LaunchAgent + docs
- The `.app` now owns port 5001, so the background server must not also run. Document
  `launchctl unload ~/Library/LaunchAgents/com.newsbot.flask.plist` (and remove the plist) in
  `update.md`; otherwise the `.app`'s Flask bind fails.
- Add a short "Running as a Mac app" section to `update.md`: dev run = `python3 desktop.py`;
  build = `bash scripts/build_app.sh` → drag `dist/Market Research.app` to `/Applications`.

---

## Files touched

| File | Change |
|---|---|
| `desktop.py` | **new** — pywebview entry: env, `init_db()`, scheduler thread, Flask thread, window |
| `db/init.py` | wrap body in `init_db()`, keep `__main__` guard |
| `app/__init__.py` | `resource_path()` helper + explicit `template_folder`/`static_folder` for frozen mode |
| `scripts/build_app.sh` | **new** — `.icns` generation + PyInstaller build |
| `MarketResearch.spec` | **new** — PyInstaller spec (add-data, hidden imports, windowed, icon) |
| `requirements.txt` | add `pywebview` |
| `requirements-build.txt` | **new** — `pyinstaller` |
| `update.md` | "Running as a Mac app" + LaunchAgent retirement note |
| `CLAUDE.md` | note the deps exception for the desktop wrapper |

No schema, collector logic, route, or PWA changes. Postgres remains an external local service.

---

## Verification (end-to-end)

1. **Dev run:** `pip install pywebview && python3 desktop.py` → a native window opens (no browser),
   Home renders with live data, sidebar/charts work, and `Market Research` appears as the app
   menu — confirm via Activity Monitor that **no Chrome/Safari** process is involved.
2. **Server managed:** market data loads (scheduler running); **close the window → the Python
   process and its threads exit** (no orphaned Flask/scheduler; check `ps`).
3. **Build:** `bash scripts/build_app.sh` → `dist/Market Research.app` exists with the custom icon.
4. **Launch from Finder:** double-click the `.app` → it opens its own window with a Dock/Launchpad
   icon, no browser, no Terminal. Data loads (Postgres running). Quit → server stops.
5. **Browser independence:** quit Chrome and Safari entirely, relaunch the `.app` → still works.
6. **No port conflict:** confirm the `com.newsbot.flask` LaunchAgent is unloaded so the `.app`
   binds 5001 cleanly.
