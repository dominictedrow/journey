# Updating the Site

## Running as a Mac app

The dashboard can also run as a standalone macOS `.app` (a native window backed by
system WebKit — no browser, no Terminal). It boots Flask + the collector scheduler
in-process and stops them when the window closes. PostgreSQL must be running locally
(e.g. Postgres.app or `brew services start postgresql`).

**Dev run (from source):**
```bash
pip install -r requirements.txt   # one-time: adds pywebview
python3 desktop.py                 # opens the window on port 5001
```

**Build the distributable app:**
```bash
pip install -r requirements-build.txt   # one-time: adds pyinstaller
bash scripts/build_app.sh                # → dist/Market Research.app
```
Then drag `dist/Market Research.app` to `/Applications` and double-click to launch.

**API keys for the packaged app.** A dev run (`python3 desktop.py`) reads the project
`.env`, but the frozen `.app` can't — its `__file__` lives inside the read-only bundle.
The built app instead reads keys from:
```
~/Library/Application Support/Market Research/.env
```
Seed it once (same `DATABASE_URL` / `FRED_API_KEY` / `BLS_API_KEY` keys as the project `.env`):
```bash
mkdir -p ~/Library/"Application Support"/"Market Research"
cp .env ~/Library/"Application Support"/"Market Research"/.env
```
Without it the app still runs (DB defaults to `postgresql://localhost/market_research`),
but FRED and BLS collectors have no key and skip.

> **LaunchAgent retired.** The `.app` (and `desktop.py`) own port **5001**, so the old
> `com.newsbot.flask` background LaunchAgent must not also run — otherwise Flask can't
> bind the port. It has been unloaded and its plist removed:
> ```bash
> launchctl unload ~/Library/LaunchAgents/com.newsbot.flask.plist
> rm ~/Library/LaunchAgents/com.newsbot.flask.plist
> ```
> The repo still contains `com.newsbot.flask.plist` for historical reference; it is
> defunct — do not re-install it. Use `scripts/start.sh` (browser) or the `.app` instead.

---

## The Basic Loop

1. Edit files in the terminal (or with Claude Code)
2. Restart Flask to pick up the changes
3. Refresh the browser

Flask does **not** hot-reload in this setup, so every code change requires a restart.

---

## Restart Commands

**Quick restart (stop + start):**
```bash
bash scripts/stop.sh && PORT=5001 bash scripts/start.sh
```

**Just stop:**
```bash
bash scripts/stop.sh
```

**Just start:**
```bash
PORT=5001 bash scripts/start.sh
```

**Open the site:**
```
http://localhost:5001/
```

---

## What Restarts Are Needed

| Change type | Action needed |
|---|---|
| Python routes, models, collectors | Restart Flask |
| Jinja2 HTML templates | Restart Flask |
| CSS / JS (static files) | Hard-refresh browser (`Cmd+Shift+R`) — no restart. **Note:** the PWA service worker (`app/static/js/sw.js`) caches static assets, so an edit can lag one reload. After changing CSS/JS, bump `CACHE` (`mr-v1` → `mr-v2`) in `sw.js`, or use DevTools → Application → Service Workers → "Update on reload" during development. |
| Database schema (`db/schema.sql`) | Run `python3 db/init.py` manually, then restart |
| `config/sources.json` | Restart Flask (loaded at startup) |
| Environment variables (`.env`) | Restart Flask |

---

## Database Schema Changes

If you add a new table or column to `db/schema.sql`:

```bash
# Option 1 — re-run init (safe, uses IF NOT EXISTS)
python3 db/init.py

# Option 2 — run a specific ALTER directly
psql market_research -c "ALTER TABLE market_data ADD COLUMN new_col TEXT;"
```

Then restart the app.

---

## Running Collectors Manually

Trigger all data collectors immediately (useful after fixing a broken collector):
```bash
bash scripts/collect_now.sh
```

---

## Checking Logs

Flask and scheduler output goes to the terminal where you ran `start.sh`. To check for errors:
```bash
# Flask log
cat logs/flask.log 2>/dev/null

# Scheduler log
cat logs/scheduler.log 2>/dev/null

# Or tail live (if start.sh pipes output there)
tail -f logs/flask.log
```

---

## Common Edit Targets

| What you want to change | File |
|---|---|
| Page layout / HTML | `app/templates/<section>/<page>.html` |
| Styles | `app/static/css/main.css` |
| Client-side JS | `app/static/js/app.js` |
| Route logic | `app/routes/<name>.py` |
| Database queries | `app/models/<name>.py` |
| Data collectors | `collectors/<name>.py` |
| Which series are tracked | `config/sources.json` |
| DB schema | `db/schema.sql` |
| Flask app setup / Jinja filters | `app/__init__.py` |

---

## Backup Before Big Changes

```bash
bash scripts/backup_db.sh
```

Dumps to `backups/market_research_YYYYMMDD.sql`.
