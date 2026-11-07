# Market Research Dashboard — Current Version

*Scanned: 2026-05-31 — All 10 phases complete and running*

---

## Implementation Status

| Phase | Name | Status |
|---|---|---|
| 1 | Setup, Schema, Scripts | ✅ Complete |
| 2 | Data Collection Engine | ✅ Complete |
| 3 | Models + Home Page | ✅ Complete |
| 4 | Tier Pages + Full Framework Flow | ✅ Complete |
| 5 | Standalone Tier Mode | ✅ Complete |
| 6 | Routine Pages | ✅ Complete |
| 7 | Asset Pages (US / International / Crypto) | ✅ Complete |
| 8 | Journal System | ✅ Complete |
| 9 | Portfolio System | ✅ Complete |
| 10 | API endpoints, Freshness badges, Print export | ✅ Complete |

---

## How to Run

```bash
# Set environment variables (required)
export DATABASE_URL="postgresql://localhost/market_research"
export FRED_API_KEY="<your_key>"
export BLS_API_KEY="<your_key>"

# Start everything (DB init + scheduler + Flask)
PORT=5001 bash scripts/start.sh   # port 5000 is taken by macOS AirPlay

# Stop
bash scripts/stop.sh

# Force-collect all data immediately (first-run population)
bash scripts/collect_now.sh

# Backup database
bash scripts/backup_db.sh
```

Open `http://localhost:5001/` in the browser.

---

## Directory Structure

```
news_bot/
├── app/
│   ├── __init__.py                   Flask factory; 7 blueprint registrations; 3 Jinja filters
│   ├── db.py                         ThreadedConnectionPool(2,10); 5 DB helper functions
│   ├── routes/
│   │   ├── home.py                   GET /
│   │   ├── framework.py              GET /framework/ (landing), GET /framework/new,
│   │   │                             GET+POST /framework/tier/<n>,
│   │   │                             GET+POST /framework/complete/<id>,
│   │   │                             POST /framework/exit/<id>,
│   │   │                             GET+POST /tiers/<n>  (standalone mode)
│   │   ├── assets.py                 GET /assets/us, /assets/international, /assets/crypto
│   │   ├── routines.py               GET+POST /routines/<daily|weekly|monthly>
│   │   ├── journal.py                GET /journal/, /journal/<id>, /journal/<id>/export;
│   │   │                             POST /journal/<id>/comment
│   │   ├── portfolio.py              GET /portfolio/, /portfolio/<stocks|crypto>;
│   │   │                             POST /portfolio/buy, /portfolio/sell
│   │   └── api.py                    GET /api/data-freshness, /api/ticker-price/<ticker>
│   ├── models/
│   │   ├── market_data.py            get_latest(), get_latest_by_tier(),
│   │   │                             get_series_history(), get_multi_series_history(),
│   │   │                             get_multi_pct_change(), get_biggest_shifts()
│   │   ├── journal.py                get_recent_entries(), get_entries_grouped(),
│   │   │                             get_entry_with_comments()
│   │   └── portfolio.py              get_cash_balance(), compute_pnl(),
│   │                                 get_open_positions(), get_snapshot_history(),
│   │                                 get_trade_history(), take_daily_snapshot()
│   ├── templates/
│   │   ├── base.html                 Sliding sidebar + sticky topbar; all pages extend this
│   │   ├── home/index.html           7-section dashboard
│   │   ├── framework/
│   │   │   ├── start.html            Landing page: tier map preview, Resume/Start Fresh
│   │   │   ├── tier.html             All 5 tiers + standalone mode (shared template)
│   │   │   └── complete.html         Final review + optional notes + journal submission
│   │   ├── assets/
│   │   │   ├── us.html               Major indices, 11 sector ETFs heatmap, macro signals
│   │   │   ├── international.html    FX, commodities, IMF GDP/CPI table, COT, news
│   │   │   └── crypto.html           Prices, on-chain signals, 4 dedicated charts
│   │   ├── routines/
│   │   │   ├── daily.html            8 key prices/signals + 24h news + journal textarea
│   │   │   ├── weekly.html           Tier 1 series + sector ETF heatmap + 7d news
│   │   │   └── monthly.html          Tier 1+2, IMF tables, portfolio summary, 30d news
│   │   ├── journal/
│   │   │   ├── list.html             Date/type filters; entries grouped by month
│   │   │   └── entry.html            Snapshot table, interpretations, portfolio entries,
│   │   │                             written notes, comments + add-comment form; print mode
│   │   └── portfolio/
│   │       └── stocks.html           Shared for stocks + crypto via asset_type variable
│   └── static/
│       ├── css/main.css              Single stylesheet; all design tokens as CSS vars
│       ├── js/
│       │   ├── chartjs.min.js        Local Chart.js copy (offline-safe)
│       │   └── app.js                toggleSidebar(), submenu expand, formatPct(),
│       │                             formatCurrency(), showModal(), hideModal()
│       └── icons/                    flag.svg, globe.svg, bitcoin.svg
├── collectors/
│   ├── base.py                       save_market_data(), save_news() — shared upsert helpers
│   ├── fred.py                       FRED API (8 series, hourly)
│   ├── bls.py                        BLS API (CPI YoY, PPI YoY, Payrolls, daily)
│   ├── crypto.py                     CoinGecko, Alternative.me, DefiLlama (15min/1h)
│   ├── etf_flows.py                  Farside HTML scrape — BTC ETF net flows (1h)
│   ├── onchain.py                    CoinGlass MVRV (daily), Blockchaincenter altcoin season
│   ├── prices.py                     TradingView Scanner API — 40+ tickers (15min)
│   ├── news.py                       RSS via xml.etree.ElementTree (3 working feeds)
│   ├── imf.py                        IMF DataMapper API — 9 countries × GDP + CPI (weekly)
│   ├── cftc.py                       CFTC COT CSV ZIP — SPX net + Gold net (weekly)
│   └── scheduler.py                  9 collector threads (2s stagger) + midnight snapshot thread
├── db/
│   ├── schema.sql                    11 tables with IF NOT EXISTS (idempotent)
│   └── init.py                       Checks .initialized flag; runs schema; seeds cash
├── config/
│   ├── settings.py                   DATABASE_URL, FRED_API_KEY, BLS_API_KEY, PORT from env
│   └── sources.json                  ~70 series definitions + 4 RSS feed entries
├── scripts/
│   ├── start.sh                      db/init.py → scheduler (bg) → flask run
│   ├── stop.sh                       Kills .scheduler.pid and .flask.pid
│   ├── collect_now.sh                Runs all collectors immediately
│   └── backup_db.sh                  pg_dump → backups/market_research_YYYYMMDD.sql
└── requirements.txt                  flask, psycopg2-binary, requests, beautifulsoup4
```

---

## Database Tables

All tables use `CREATE TABLE IF NOT EXISTS` — schema is fully idempotent.

| Table | Purpose |
|---|---|
| `market_data` | Append-only time-series. `DISTINCT ON (series_id) ORDER BY fetched_at DESC` for latest. |
| `news_items` | Headlines. Deduplicated on `url`. `is_active=FALSE` after 2 days (nightly scheduler). |
| `framework_sessions` | One row per research session. Tracks tier progress + completion state. |
| `tier_interpretations` | User's typed notes per series per tier per session. Deleted + re-inserted on each tier POST. |
| `session_portfolio_entries` | Ticker/price/qty entered during a session (feeds journal snapshot). |
| `journal_entries` | Immutable records with three JSONB columns: `auto_data_snapshot`, `interpretations_snapshot`, `portfolio_entries_snapshot`. |
| `journal_comments` | Mutable follow-up notes attached to a journal entry. |
| `portfolio_positions` | Paper-trade positions. `status`: `open` or `closed`. |
| `trade_history` | Every buy and sell, including P&L on sells. |
| `portfolio_cash` | Running cash balance per asset type. Seeds at $10,000 each. |
| `portfolio_snapshots` | Daily total value snapshot for P&L chart. Upserted on page load + midnight scheduler. |

**Key indexes:**
- `idx_md_series_fetched` on `(series_id, fetched_at DESC)` — powers all "get latest" queries
- `idx_md_tier_fetched` on `(tier, fetched_at DESC)` — powers tier page data loading
- `idx_news_cat_pub` on `(category, published_at DESC)` — powers news feeds by category

---

## Data Series Collected

| Series ID | Collector | Source | Refresh |
|---|---|---|---|
| `T10Y2Y`, `BAMLH0A0HYM2`, `DGS10`, `DFII10`, `M2SL`, `WALCL`, `DEXJPUS`, `DEXUSAL` | `fred.py` | FRED API | 1h |
| `CPI_YOY`, `PPI_YOY`, `PAYROLLS` | `bls.py` | BLS API | Daily |
| `VIX`, `DXY`, `COPPER`, `GOLD`, `COPPER_GOLD_RATIO`, `MOVE`, `OIL_WTI` | `prices.py` | TradingView Scanner | 15m |
| `EQUITY_SPY/QQQ/IWM/DIA` | `prices.py` | TradingView Scanner | 15m |
| `SECTOR_XLK/XLF/XLE/XLV/XLY/XLP/XLI/XLB/XLRE/XLU/XLC` | `prices.py` | TradingView Scanner | 15m |
| `BOND_TLT/IEF/SHY` | `prices.py` | TradingView Scanner | 15m |
| `INTL_EWJ/EWZ/EEM/VEA/MCHI/EWG` | `prices.py` | TradingView Scanner | 15m |
| `FX_EURUSD`, `FX_GBPUSD` | `prices.py` | TradingView Scanner | 15m |
| `BTC_USD`, `ETH_USD`, `SOL_USD`, `BTC_DOMINANCE` | `crypto.py` | CoinGecko | 15m |
| `FEAR_GREED` | `crypto.py` | Alternative.me | 1h |
| `STABLECOIN_MCAP`, `DEFI_TVL` | `crypto.py` | DefiLlama | 1h |
| `BTC_ETF_FLOWS_NET` | `etf_flows.py` | Farside HTML scrape | 1h |
| `MVRV_ZSCORE` | `onchain.py` | CoinMetrics community API (MVRV ratio; no true Z-Score source available free) | Daily |
| `ALTCOIN_SEASON` | `onchain.py` | Blockchaincenter HTML | Daily |
| `IMF_GDP_*`, `IMF_CPI_*` (9 countries) | `imf.py` | IMF DataMapper API | Weekly |
| `COT_SPX_NET`, `COT_GOLD_NET` | `cftc.py` | CFTC CSV ZIP | Weekly |

**News feeds (RSS):** MarketWatch, BBC Business, CNBC Markets, CoinDesk, CoinTelegraph (5 feeds, all working).

---

## app/db.py — Database Helpers

`ThreadedConnectionPool(2, 10)` initialised at import time from `DATABASE_URL`. `get_conn()` is a `@contextmanager` that commits on clean exit and rolls back on exception.

| Function | Returns | Notes |
|---|---|---|
| `get_conn()` | context manager | Yields raw psycopg2 connection; commits/rollbacks; returns to pool |
| `execute(sql, params)` | None | Fire-and-forget write |
| `execute_returning(sql, params)` | tuple | Raw cursor `.fetchone()` — used for `RETURNING id` |
| `fetchone_dict(sql, params)` | dict or None | RealDictCursor |
| `fetchall_dict(sql, params)` | list of dicts | RealDictCursor — used throughout all routes and models |

---

## app/__init__.py — Factory + Jinja Filters

`create_app()` registers 7 blueprints and installs three Jinja2 template filters:

| Filter | Usage | Behaviour |
|---|---|---|
| `time_ago` | `{{ dt \| time_ago }}` | Returns `"5m ago"`, `"3h ago"`, `"2d ago"`; handles tz-naive datetimes |
| `fmt_num` | `{{ value \| fmt_num }}` | M suffix ≥1M; comma-thousands ≥1k; 5dp for tiny values; `—` for None |
| `sector_color` | `{{ pct \| sector_color }}` | `hsl(0/120, 70%, 15–35%)` for sector heatmap cells; clamped ±5% |

---

## Phase 1 — Setup, Schema, Scripts ✅

- Full directory tree; `db/schema.sql` (11 tables, all indexes, fully idempotent)
- `db/init.py` — checks `db/.initialized` flag; runs schema; seeds `portfolio_cash` with $10k each
- `config/settings.py` — raises `RuntimeError` if `DATABASE_URL` is missing
- `config/sources.json` — ~70 series definitions + 4 RSS feed entries (3 working)
- `app/db.py` — `ThreadedConnectionPool(2,10)` + 5 helper functions
- `app/__init__.py` — Flask factory, 7 blueprint registrations, 3 Jinja filters
- `scripts/start.sh` — loads `.env`; runs `db/init.py`; starts scheduler in background; starts Flask
- `scripts/stop.sh` — kills `.scheduler.pid` and `.flask.pid`
- `base.html` — sliding sidebar (hamburger → panel with dark overlay); sticky topbar (52px); hover-submenus for Framework/Assets/Routines
- `main.css` — CSS custom properties (`--bg`, `--surface`, `--accent`, `--green`, `--red`, etc.); `.card`; `.positive`/`.negative`; modal overlay; gradient button variants (`.btn-primary`, `.btn-daily`, `.btn-weekly`, `.btn-monthly`, `.btn-orange`, `.btn-teal`, `.btn-danger`); `.grid-2/3/4`; tier map nodes; freshness dots; `@keyframes rise-in` + `.anim-rise` (page-in animation)
- `app.js` — `toggleSidebar()`, submenu expand/collapse, `formatPct()`, `formatCurrency()`, `showModal()`, `hideModal()`, `animatePageIn()` (page-in animation)

---

## Phase 2 — Data Collection Engine ✅

`collectors/base.py`:
- `save_market_data(rows)` — `psycopg2.extras.execute_batch`; append-only insert
- `save_news(items)` — `ON CONFLICT (url) DO NOTHING` deduplication

`collectors/fred.py` — fetches last observation for 8 FRED series; skips `value == '.'` entries.

`collectors/bls.py` — POSTs to BLS v2 API with `calculations: true`; extracts 12-month `pct_changes` for CPI/PPI YoY; stores raw value for payrolls.

`collectors/crypto.py` — batches BTC/ETH/SOL prices in one CoinGecko call; BTC dominance from CoinGecko global endpoint; Fear & Greed from Alternative.me; sums stablecoin market caps from DefiLlama; sums DeFi TVL across all chains.

`collectors/etf_flows.py` — BeautifulSoup parses Farside HTML table; uses `Total` column if present; else sums all numeric `<td>` values per row; saves only the most recent row.

`collectors/onchain.py` — CoinGlass MVRV: tries API endpoint, returns HTTP 500 — logs error and continues. Blockchaincenter altcoin season: four regex patterns then BeautifulSoup fallback.

`collectors/prices.py` — **TradingView Scanner API** (`https://scanner.tradingview.com/{screener}/scan`); groups symbols by screener (`america`, `cfd`, `futures`, `forex`); POSTs one batch request per screener with `["close"]` column; handles `america`, `cfd`, `futures`, `forex` screeners; computes `COPPER_GOLD_RATIO` from live copper and gold prices; 1s sleep between screener groups; 40+ symbols total.

`collectors/news.py` — `xml.etree.ElementTree` RSS parser; `email.utils.parsedate_to_datetime` for RFC 2822 dates; `published_at` stored as NULL on parse failure; caps at 20 items per feed.

`collectors/imf.py` — IMF DataMapper REST API for `NGDP_RPCH` and `PCPIPCH` across 9 countries; picks most recent year with a non-null value.

`collectors/cftc.py` — downloads financial futures + commodity futures ZIP files; uses `Lev_Money_Positions_*` columns for SPX net; `M_Money_Positions_*` for Gold net; cleans up temp files in `finally` blocks.

`collectors/scheduler.py` — one `threading.Thread(daemon=True)` per collector (9 total); reads per-collector minimum interval from `sources.json`; 2-second stagger between thread starts; exceptions isolated per thread (full traceback logged, loop continues); midnight thread takes portfolio snapshots + expires news older than 2 days; writes `.scheduler.pid`.

---

## Phase 3 — Models + Home Page ✅

`app/models/market_data.py`:
- `get_latest()` — `DISTINCT ON (series_id) ORDER BY fetched_at DESC`; returns `{series_id: dict}` map
- `get_latest_by_tier(tier_num)` — all series for a tier, one row per series_id
- `get_series_history(series_id, limit=30)` — ordered ASC, returns `[{value, recorded_at}]`
- `get_multi_series_history(series_ids, limit=80)` — fetches history for multiple series in one query using `ROW_NUMBER() OVER (PARTITION BY series_id)`; returns `{series_id: [rows]}`
- `get_multi_pct_change(series_ids, days)` — computes % change over N calendar days for multiple series in one query using correlated subqueries; returns `{series_id: pct_float}`
- `get_biggest_shifts(limit=10)` — CTE comparing latest vs prior-12h values; ranks by `ABS(pct_change)`

`app/models/journal.py`:
- `get_recent_entries(limit=4)` — newest entries for home page widget
- `get_entries_grouped()` — groups up to 200 entries by `strftime('%B %Y')` (unused; route does grouping inline)
- `get_entry_with_comments(entry_id)` — entry dict + ordered comments list

`app/routes/home.py` — assembles: `get_latest()`, `get_biggest_shifts(8)`, `get_recent_entries(4)`, active session query, tier activity map from `journal_entries`. Increments `engagement_score` for displayed stories.

`app/templates/home/index.html` — 7 sections:
1. Gradient greeting + "Resume Session" button if active framework session in progress
2. 3 asset icon-cards (flag/globe/bitcoin SVGs): US (SPY + VIX), International (EEM + DXY), Crypto (BTC + Fear & Greed with colour coding)
3. 3 routine buttons (Daily/Weekly/Monthly) with gradient colour variants
4. Top 6 news stories with source badge and `time_ago` timestamp
5. Biggest Data Shifts — Chart.js horizontal bar chart (green positive, red negative), sized by shift count
6. Tier map — 5 clickable nodes linked to `/tiers/<n>`; green = has journal entry, red = in-progress session
7. Recent Journal — 4 clickable date rectangles showing entry type + date

---

## Phase 4 — Tier Pages + Full Framework Flow ✅

`app/routes/framework.py` routes:

| Route | Behaviour |
|---|---|
| `GET /framework/` | Renders `start.html` with active session data (Resume or Start Fresh) |
| `GET /framework/new` | Deactivates any existing active session; creates new `full_framework` session; redirects to tier 1 |
| `GET /framework/tier/<n>` | Loads tier series with 30-day sparklines via `_tier_series()`; fetches saved interpretations; fetches tier-relevant news; renders `tier.html` |
| `POST /framework/tier/<n>` | Saves interpretations (DELETE + re-INSERT); saves portfolio candidates; updates `completed_tiers[]` and `last_active_tier`; handles `action=next/prev/exit` |
| `GET /framework/complete/<id>` | Renders `complete.html` with all interpretations grouped by tier + portfolio candidates |
| `POST /framework/complete/<id>` | Builds full `auto_data_snapshot` from all series; builds JSONB snapshots; inserts `journal_entries`; marks session complete; redirects to journal entry |
| `POST /framework/exit/<id>` | Updates `last_active_tier`; redirects to home (session stays active for resume) |

`app/templates/framework/start.html` — centred layout; tier map preview (non-clickable nodes); if active session: blue-bordered card with session time/progress + "Resume" button + "Start Fresh" secondary button; else: single "Start Framework →" primary button.

`app/templates/framework/tier.html` (shared for full framework and standalone):
- Full framework mode: progress bar (T1–T5, coloured blue=current/green=done/grey=pending)
- Two-column layout (main + 340px sticky news panel)
- Left: 2-column grid of series cards (metric name, value, text_value, 60px sparkline) + interpretation textarea per series
- Freshness dots per series: 7px coloured dot polled from `/api/data-freshness` every 60s
- Portfolio candidate table with JS add-row / remove-row buttons
- Standalone mode only: written notes textarea
- Sticky bottom action bar: Exit (triggers modal) / ← Back / Next → or "Complete Framework"
- Exit modal with save confirmation

`app/templates/framework/complete.html` — per-tier interpretation summary cards (badge showing note count); portfolio candidates table; optional written notes textarea; "Save to Journal & Finish" button.

Private helpers: `_tier_series()` (builds series list + chart_data), `_tier_news()` (fetches by category), `_existing_interps()`, `_save_interps()`, `_save_portfolio_entries()`, `_create_tier_journal_entry()`.

---

## Phase 5 — Standalone Tier Mode ✅

`GET /tiers/<n>` — renders `tier.html` with `standalone=True`: hides progress bar and exit button; shows "Save & Create Journal Entry" orange button instead of Next.

`POST /tiers/<n>` — creates a one-off `tier_N` session; saves interpretations + portfolio entries; calls `_create_tier_journal_entry()`; marks session complete; redirects to journal entry.

`_create_tier_journal_entry(session_id, tier_num, written_text)` — shared by both standalone and full-framework complete route: builds `auto_data_snapshot` for tier's series, builds JSONB interpretations + portfolio snapshots, inserts `journal_entries` with `entry_type='tier_N'`, returns new entry ID.

---

## Phase 6 — Routine Pages ✅

`app/routes/routines.py` — `_get_routine_data(period)` assembles:

| Period | Data assembled |
|---|---|
| `daily` | BTC, ETH, SPY, QQQ, DXY, GOLD, VIX, FEAR_GREED + 24h news |
| `weekly` | All WEEKLY_KEYS (Tier 1 FRED+BLS+prices) + sector ETFs (7-day perf) + 7d news |
| `monthly` | `get_latest_by_tier(1)` + `get_latest_by_tier(2)` split into IMF GDP, IMF CPI, other T2 + portfolio summary + 30d news |

`_with_history(series_list, days)` — attaches `chart_data` (80 records via `get_multi_series_history`) and `week_pct` (% change over N days via `get_multi_pct_change`) to each series.

POST handler: if `written_text` is non-empty, creates `journal_entries` row with `entry_type=period` and `auto_data_snapshot`; redirects to new entry.

**Templates:**
- `daily.html` — 4-column card grid with sparklines; 24h news; notes textarea; "Save to Journal" (red gradient) / "← Skip"
- `weekly.html` — Tier 1 series in 3-column grid; sector ETF cards coloured via `sector_color` filter; 7d news; notes textarea; "Save to Journal" (amber gradient) / "← Skip"
- `monthly.html` — 4-card portfolio summary; Tier 1+2 data cards; IMF GDP/CPI side-by-side tables; 30d news; notes textarea; "Save to Journal" (green gradient) / "← Skip"

---

## Phase 7 — Asset Pages ✅

`app/routes/assets.py`:

**`GET /assets/us`** — loads:
- Major indices (SPY, QQQ, IWM, DIA) with 80-day sparklines + change_pct from history
- All `SECTOR_*` ETFs with 80-day sparklines, `pct_1m` (30d change), `pct_1w` (7d change) via `get_multi_pct_change`; sorted by metric_name
- Macro signals (VIX, T10Y2Y, BAMLH0A0HYM2, DGS10, DFII10, MOVE) with 80-day sparklines
- News: 72h window, up to 40 items, any category

**`GET /assets/international`** — loads:
- FX (DXY, DEXJPUS, DEXUSAL) with 120-day sparklines
- Commodities (COPPER, GOLD, OIL_WTI, COPPER_GOLD_RATIO) with 80-day sparklines
- IMF GDP + CPI merged by country into `imf_rows` (country, gdp dict, cpi dict)
- CFTC COT rows (COT_SPX_NET, COT_GOLD_NET)
- News: 168h window

**`GET /assets/crypto`** — loads:
- Crypto prices (BTC_USD, ETH_USD, SOL_USD) with 200-day sparklines + change_pct
- Crypto signals (BTC_DOMINANCE, FEAR_GREED, STABLECOIN_MCAP, DEFI_TVL, BTC_ETF_FLOWS_NET, MVRV_ZSCORE, ALTCOIN_SEASON) with 80-day sparklines; FEAR_GREED and MVRV_ZSCORE have colour-coded values
- Dedicated chart data: `btc_dom_chart` (200pts), `fg_chart`, `etf_chart`, `stbl_chart`
- Crypto news: 168h window

All three templates share: gradient page title; Data/News slider toggle (JS `showPanel()`); all metric cards with Chart.js sparklines; back link to `/`.

---

## Phase 8 — Journal System ✅

`app/routes/journal.py`:

| Route | Behaviour |
|---|---|
| `GET /journal/` | `?start=`, `?end=`, `?type=` filters via parameterised SQL; groups by month; renders `list.html` |
| `GET /journal/<id>` | Fetches entry + comments; renders `entry.html`; 302 to list if not found |
| `GET /journal/<id>/export` | Same render with `print_mode=True` (sidebar stripped for print-to-PDF) |
| `POST /journal/<id>/comment` | Guards FK; inserts into `journal_comments`; redirects back |

`journal/list.html` — filter bar (date pickers + type dropdown + Filter button + Clear link); entries grouped under gradient green→blue month-year headings; each entry row: day number + month + vertical divider + type label + text snippet (truncated at 130 chars) + time_ago.

`journal/entry.html`:
- Market Data Snapshot table from `auto_data_snapshot` JSONB (series name, value via `fmt_num`, category, tier)
- Interpretations section from `interpretations_snapshot` JSONB, grouped by tier key
- Portfolio Entries table from `portfolio_entries_snapshot` JSONB
- Written notes in `<pre>` with `white-space: pre-wrap` (read-only)
- Comments list + "Add Comment" form with teal gradient submit button
- Print mode: `window.print()` auto-called on load; sidebar/nav hidden via inline `<style>`

**JSONB snapshot formats:**

`auto_data_snapshot`:
```json
{"T10Y2Y": {"value": 0.45, "text_value": null, "label": "2s/10s Yield Curve", "category": "credit", "tier": 1}}
```

`interpretations_snapshot`:
```json
{"1": [{"key": "T10Y2Y", "label": "2s/10s Yield Curve", "interpretation": "Curve steepening..."}]}
```

`portfolio_entries_snapshot`:
```json
[{"ticker": "BTC", "entry_price": 95000.0, "quantity": 0.1, "asset_type": "crypto"}]
```

---

## Phase 9 — Portfolio System ✅

`app/models/portfolio.py`:
- `get_cash_balance(asset_type)` — queries `portfolio_cash`
- `compute_pnl(entry_price, current_price, quantity)` — pure arithmetic function
- `get_open_positions(asset_type)` — joins with `get_latest()` for live prices; attaches `current_price`, `current_value`, `pnl`, `pnl_pct`, `has_live_price` to each row; falls back to `entry_price` if no market data. Series ID heuristic: crypto → `{TICKER}_USD`, stocks → `EQUITY_{TICKER}`.
- `get_snapshot_history(asset_type, limit=365)` — ordered ASC for P&L chart
- `get_trade_history(asset_type, limit=50)` — JOINs `trade_history` + `portfolio_positions`
- `take_daily_snapshot(asset_type)` — sums open positions × live prices + cash; `ON CONFLICT DO UPDATE` upsert into `portfolio_snapshots`

`app/routes/portfolio.py`:
- `GET /portfolio/` → redirect to `/portfolio/stocks`
- `GET /portfolio/<asset_type>` — loads positions + cash + snapshots + trades; calls `take_daily_snapshot` (idempotent); renders `stocks.html`
- `POST /portfolio/buy` — validates; checks cash; inserts position + buy trade record; debits cash; creates `portfolio` journal entry; links position to journal entry via `journal_entry_id`
- `POST /portfolio/sell` — finds open position; computes P&L; inserts sell trade; closes position; credits cash; creates sell journal entry with P&L in written_text

`app/templates/portfolio/stocks.html` (shared for stocks + crypto via `asset_type` variable):
- Stocks/Crypto tab switcher (blue highlight on active)
- 4 summary cards: Total Value, Cash Available, Invested, Unrealized P&L (green/red)
- P&L line chart: 1D/1W/1M/1Y time range buttons; client-side filter from embedded JSON; colour (green/red) based on direction
- Allocation doughnut chart with colour-coded legend showing each position + cash
- Open Positions list: ticker, entry price, qty, current price (amber `~` if no live data), current value, P&L+% (green/red), Sell button
- Sell modal: exit price input + real-time P&L preview via `updatePnlPreview()` using current and entry prices
- Trade History table: date, ticker, type (green BUY / red SELL), price, qty, value, P&L with %
- Buy modal: ticker, entry price, quantity, available cash display

---

## Phase 10 — API Endpoints + Polish ✅

`app/routes/api.py`:
- `GET /api/data-freshness` — returns `{series_id: {seconds_since_fetch, refresh_seconds, status}}` for all series; status `ok`/`amber`/`red` at 2× and 5× refresh interval thresholds; loads intervals from `sources.json` at startup
- `GET /api/ticker-price/<ticker>` — live price via TradingView; checks `market_data` first (instant, no external call) then falls back to TradingView symbol search → screener scan for arbitrary tickers; used by sell modal to pre-fill current price for positions not covered by the scheduled collector

**Freshness badges in `tier.html`** — each series card has a 6px dot (`id="dot-{series_id}"`); JS block polls `/api/data-freshness` every 60s, updates dot colour and title tooltip.

**Print export** — `GET /journal/<id>/export` renders `entry.html` with `print_mode=True`; inline `<style>` hides sidebar/nav; `window.print()` fires on load.

**Daily scheduler jobs (midnight thread):**
- `take_daily_snapshot('stock')` and `take_daily_snapshot('crypto')`
- `UPDATE news_items SET is_active = FALSE WHERE published_at < NOW() - INTERVAL '2 days'`

---

## Page-in Animation

Every page animates its content elements in on load — rising up and fading in, top-to-bottom.

**CSS (`main.css`):**
```css
@keyframes rise-in {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
.anim-rise {
  animation: rise-in 0.6s cubic-bezier(0.16, 1, 0.3, 1) backwards;
  will-change: transform, opacity;
}
```
- `animation-fill-mode: backwards` — element stays invisible during its delay, returns to normal CSS after completion (hover/transition effects work as usual)
- `cubic-bezier(0.16, 1, 0.3, 1)` — expo ease-out: quick rise, smooth settle

**JS (`app.js`) — `animatePageIn()`:**
- Runs on `DOMContentLoaded` (or immediately if DOM is already ready)
- Selects: `.card`, `.page-title`, `.tier-node`, `.tier-connector`, `h2`, `.btn`, `textarea`
- **Deduplication** — elements contained within another selected element are removed (e.g. a `.btn` inside a `.card` is excluded; the card animates as a unit)
- **Sort** — by `getBoundingClientRect().top` then `.left` (same row within 12px = left-to-right sweep)
- **Stagger** — total spread capped at 0.8 s regardless of element count; step = `spread / (n-1)`
- No HTML template changes required — animation is applied entirely via JS class + inline `animationDelay`

**What animates on each page:**
- Home: page title, Resume button, 3 asset cards, 3 routine buttons, section headings, news story cards, shifts chart card, tier nodes + connectors, journal entry cards
- Tier (framework/standalone): page title, all series data cards, their interpretation textareas (each animates just after its card), the portfolio candidates card, the sticky news card, the action bar buttons
- Asset pages: page title, equity/sector/signal cards, chart cards
- Routines: page title, data cards, news card, notes card
- Journal list: page title, filter bar (`.btn`), month heading (`h2`), each entry row (`.card`)
- Journal entry: page title, snapshot table card, interpretation cards, portfolio card, notes card, comments card
- Portfolio: page title, tab buttons, summary cards, chart cards, position cards, trade history card

---

## Known Issues

None currently known.
