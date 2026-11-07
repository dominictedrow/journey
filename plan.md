# Market Research Dashboard — Implementation Plan

## Context

A local-only personal finance research website that implements the Global Macro Master Investment Framework. The app guides Dominic through a 5-tier investment analysis process (macro regime → country scoring → sector rotation → stock selection → risk management), stores all market data in PostgreSQL, tracks a paper-trade portfolio ($10k stocks + $10k crypto), and auto-generates journal entries after each research session. The site runs 100% locally — internet connection is used only for outbound data collection.

---

## Tech Stack & Dependencies

**Backend (Python):** 4 packages only.
| Package | Why it's included |
|---|---|
| `flask` | Web framework — routing, Jinja2 templates, Blueprint structure |
| `psycopg2-binary` | PostgreSQL driver — direct SQL, no ORM |
| `requests` | HTTP client for all API calls and web scraping fetches |
| `beautifulsoup4` | HTML parsing for Farside ETF flows, Blockchaincenter, LookIntoBitcoin (no structured API) |

Python stdlib covers everything else: `xml.etree.ElementTree` (RSS parsing), `csv` (CFTC data), `json`, `threading`, `time`, `datetime`, `hashlib`, `os`, `email.utils` (date parsing).

**Frontend:** `Chart.js` downloaded once to `static/js/chartjs.min.js` — no CDN (app must work offline). Vanilla JS only — no React, jQuery, Alpine.

**Database:** PostgreSQL (local instance).

---

## Directory Structure

```
market-research/
├── app/
│   ├── __init__.py           # Flask app factory, registers blueprints, Jinja filters
│   ├── db.py                 # psycopg2 ThreadedConnectionPool, get_conn(), fetchone(), fetchall(), execute()
│   ├── routes/
│   │   ├── home.py           # GET /
│   │   ├── framework.py      # GET/POST /framework/, /framework/tier/<n>, /tiers/<n>, /framework/complete/<id>
│   │   ├── assets.py         # GET /assets/<us|international|crypto>
│   │   ├── routines.py       # GET/POST /routines/<daily|weekly|monthly>
│   │   ├── journal.py        # GET /journal/, /journal/<id>; POST /journal/<id>/comment
│   │   └── portfolio.py      # GET /portfolio/; POST /portfolio/buy, /portfolio/sell
│   ├── models/
│   │   ├── market_data.py    # get_latest(), get_latest_by_tier(), get_series_history(), get_biggest_shifts()
│   │   ├── journal.py        # get_recent_entries(), get_entries_grouped(), get_entry_with_comments()
│   │   └── portfolio.py      # get_open_positions(), get_cash_balance(), get_snapshot_history(), compute_pnl()
│   ├── templates/
│   │   ├── base.html         # Sidebar + topbar layout — all pages extend this
│   │   ├── home/index.html
│   │   ├── framework/
│   │   │   ├── start.html    # "Start Framework" button page
│   │   │   ├── tier.html     # Reused for all 5 tiers (+ standalone mode)
│   │   │   └── complete.html # Final review + journal submission
│   │   ├── assets/
│   │   │   ├── us.html
│   │   │   ├── international.html
│   │   │   └── crypto.html
│   │   ├── routines/
│   │   │   ├── daily.html
│   │   │   ├── weekly.html
│   │   │   └── monthly.html
│   │   ├── journal/
│   │   │   ├── list.html
│   │   │   └── entry.html
│   │   └── portfolio/
│   │       ├── stocks.html
│   │       └── crypto.html
│   └── static/
│       ├── css/main.css      # Single stylesheet — all design tokens as CSS variables
│       ├── js/
│       │   ├── chartjs.min.js  # Local Chart.js copy
│       │   └── app.js          # Sidebar toggle, submenu hover, shared utilities
│       └── icons/            # SVG files: flag.svg, globe.svg, bitcoin.svg
├── collectors/
│   ├── base.py               # save_market_data(), save_news() — shared upsert logic
│   ├── fred.py               # FRED API — T10Y2Y, BAMLH0A0HYM2, M2SL, DGS10, DFII10, DEXJPUS, DEXUSAL, WALCL
│   ├── bls.py                # BLS API — CPI (CUUR0000SA0), PPI (WPUFD49104), Payrolls (CES0000000001)
│   ├── crypto.py             # CoinGecko global+prices, Alternative.me Fear&Greed, DefiLlama stablecoins/TVL
│   ├── etf_flows.py          # Farside.co.uk Bitcoin ETF flows (BeautifulSoup HTML table)
│   ├── onchain.py            # CoinGlass MVRV, Blockchaincenter altcoin season index (BeautifulSoup)
│   ├── prices.py             # Yahoo Finance hidden JSON API for stocks/ETFs/commodities/FX
│   ├── news.py               # RSS feeds via xml.etree.ElementTree — no beautifulsoup
│   ├── imf.py                # IMF DataMapper REST API — GDP, CPI, current account
│   ├── cftc.py               # CFTC COT weekly CSV download (ZIP extract + csv module)
│   └── scheduler.py          # threading.Thread-based scheduler — no APScheduler
├── db/
│   ├── schema.sql            # Full schema with IF NOT EXISTS — idempotent
│   └── init.py               # Runs schema.sql, seeds portfolio_cash
├── config/
│   ├── settings.py           # DATABASE_URL, FRED_API_KEY, BLS_API_KEY, FLASK_PORT
│   └── sources.json          # All series definitions (id, label, tier, collector, refresh interval) + RSS feeds
├── scripts/
│   ├── start.sh              # Runs db/init.py → scheduler.py (background) → flask run
│   ├── stop.sh               # Kills .scheduler.pid and .flask.pid
│   ├── collect_now.sh        # Runs all collectors immediately (for first-run data population)
│   └── backup_db.sh          # pg_dump to backups/market_research_YYYYMMDD.sql
└── requirements.txt          # flask, psycopg2-binary, requests, beautifulsoup4
```

---

## Database Schema

All tables use `CREATE TABLE IF NOT EXISTS`. Run once via `db/init.py`.

```sql
-- Append-only time-series. "Latest" queries use DISTINCT ON (series_id) ORDER BY fetched_at DESC.
CREATE TABLE market_data (
    id           BIGSERIAL PRIMARY KEY,
    series_id    TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    value        NUMERIC,
    text_value   TEXT,
    source       TEXT NOT NULL,
    category     TEXT,           -- 'macro','credit','fx','volatility','crypto','equity','bond','commodity'
    tier         INTEGER,        -- 1-5; NULL = cross-tier
    recorded_at  TIMESTAMPTZ NOT NULL,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_md_series_fetched ON market_data(series_id, fetched_at DESC);
CREATE INDEX idx_md_tier_fetched   ON market_data(tier, fetched_at DESC);

-- News; URL is unique — ON CONFLICT DO NOTHING deduplicates across collector runs.
CREATE TABLE news_items (
    id               BIGSERIAL PRIMARY KEY,
    source_name      TEXT NOT NULL,
    title            TEXT NOT NULL,
    url              TEXT NOT NULL UNIQUE,
    summary          TEXT,
    published_at     TIMESTAMPTZ,
    category         TEXT,       -- 'macro','crypto','equity'
    tier             INTEGER,
    engagement_score INTEGER DEFAULT 0,
    is_active        BOOLEAN DEFAULT TRUE,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_news_cat_pub ON news_items(category, published_at DESC);
CREATE INDEX idx_news_active   ON news_items(is_active, engagement_score DESC);

-- One row per user-initiated research session.
CREATE TABLE framework_sessions (
    id               BIGSERIAL PRIMARY KEY,
    session_type     TEXT NOT NULL,   -- 'full_framework','tier_1'..'tier_5','daily','weekly','monthly'
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    last_active_tier INTEGER,
    completed_tiers  INTEGER[] DEFAULT '{}',
    is_active        BOOLEAN DEFAULT TRUE,
    is_complete      BOOLEAN DEFAULT FALSE
);

-- One row per data point per tier per session (user's typed interpretation).
CREATE TABLE tier_interpretations (
    id              BIGSERIAL PRIMARY KEY,
    session_id      BIGINT NOT NULL REFERENCES framework_sessions(id) ON DELETE CASCADE,
    tier            INTEGER NOT NULL,
    data_item_key   TEXT NOT NULL,    -- matches series_id
    data_item_label TEXT NOT NULL,
    interpretation  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_interp_session ON tier_interpretations(session_id, tier);

-- Asset selections made during any session (feeds into journal + portfolio).
CREATE TABLE session_portfolio_entries (
    id          BIGSERIAL PRIMARY KEY,
    session_id  BIGINT NOT NULL REFERENCES framework_sessions(id),
    ticker      TEXT NOT NULL,
    entry_price NUMERIC(14,4),
    quantity    NUMERIC(14,4),
    asset_type  TEXT NOT NULL    -- 'stock' or 'crypto'
);

-- Immutable after creation. Edits are forbidden — only comments allowed post-creation.
CREATE TABLE journal_entries (
    id                         BIGSERIAL PRIMARY KEY,
    session_id                 BIGINT REFERENCES framework_sessions(id),
    entry_date                 DATE NOT NULL DEFAULT CURRENT_DATE,
    entry_type                 TEXT NOT NULL, -- 'full_framework','tier_1'..'tier_5','daily','weekly','monthly'
    tier_number                INTEGER,
    auto_data_snapshot         JSONB,    -- market_data at time of submission
    interpretations_snapshot   JSONB,    -- all tier_interpretations keyed by tier
    portfolio_entries_snapshot JSONB,    -- session_portfolio_entries array
    written_text               TEXT,
    created_at                 TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_journal_date    ON journal_entries(entry_date DESC);
CREATE INDEX idx_journal_session ON journal_entries(session_id);

CREATE TABLE journal_comments (
    id           BIGSERIAL PRIMARY KEY,
    entry_id     BIGINT NOT NULL REFERENCES journal_entries(id) ON DELETE CASCADE,
    comment_text TEXT NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_comments_entry ON journal_comments(entry_id);

-- Paper trading. status: 'open' | 'closed'.
CREATE TABLE portfolio_positions (
    id               BIGSERIAL PRIMARY KEY,
    asset_type       TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    entry_price      NUMERIC(14,4) NOT NULL,
    quantity         NUMERIC(14,4) NOT NULL,
    cost_basis       NUMERIC(14,4) NOT NULL,  -- entry_price * quantity, set at insert
    entry_date       DATE NOT NULL DEFAULT CURRENT_DATE,
    status           TEXT NOT NULL DEFAULT 'open',
    journal_entry_id BIGINT REFERENCES journal_entries(id),
    session_id       BIGINT REFERENCES framework_sessions(id)
);
CREATE INDEX idx_pos_status_type ON portfolio_positions(status, asset_type);

CREATE TABLE trade_history (
    id          BIGSERIAL PRIMARY KEY,
    position_id BIGINT NOT NULL REFERENCES portfolio_positions(id),
    trade_type  TEXT NOT NULL,    -- 'buy' or 'sell'
    price       NUMERIC(14,4) NOT NULL,
    quantity    NUMERIC(14,4) NOT NULL,
    trade_value NUMERIC(14,4) NOT NULL,  -- price * quantity
    pnl         NUMERIC(14,4),
    pnl_percent NUMERIC(8,4),
    trade_date  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_trades_date ON trade_history(trade_date DESC);

-- Starting cash, never reset.
CREATE TABLE portfolio_cash (
    asset_type TEXT PRIMARY KEY,
    balance    NUMERIC(14,4) NOT NULL DEFAULT 10000.00
);
INSERT INTO portfolio_cash VALUES ('stock', 10000.00), ('crypto', 10000.00) ON CONFLICT DO NOTHING;

-- Daily snapshot for portfolio P&L chart.
CREATE TABLE portfolio_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    snapshot_date   DATE NOT NULL,
    asset_type      TEXT NOT NULL,
    total_value     NUMERIC(14,4),
    cash_balance    NUMERIC(14,4),
    positions_value NUMERIC(14,4),
    UNIQUE(snapshot_date, asset_type)
);
```

---

## Data Source Map

| Framework Need | Series IDs | Collector | API/Source | Refresh |
|---|---|---|---|---|
| Yield curve | `T10Y2Y` | `fred.py` | FRED API | 1h |
| HY OAS spread | `BAMLH0A0HYM2` | `fred.py` | FRED API | 1h |
| 10Y Treasury | `DGS10` | `fred.py` | FRED API | 1h |
| 10Y TIPS real yield | `DFII10` | `fred.py` | FRED API | 1h |
| M2 money supply | `M2SL` | `fred.py` | FRED API | 1h |
| Fed balance sheet | `WALCL` | `fred.py` | FRED API | 1h |
| USD/JPY | `DEXJPUS` | `fred.py` | FRED API | 1h |
| AUD/USD | `DEXUSAL` | `fred.py` | FRED API | 1h |
| VIX | `VIX` | `prices.py` | Yahoo Finance (`^VIX`) | 15m |
| DXY | `DXY` | `prices.py` | Yahoo Finance (`DX-Y.NYB`) | 15m |
| Copper futures | `COPPER` | `prices.py` | Yahoo Finance (`HG=F`) | 15m |
| Gold futures | `GOLD` | `prices.py` | Yahoo Finance (`GC=F`) | 15m |
| Copper/Gold ratio | `COPPER_GOLD_RATIO` | `prices.py` | Computed | 15m |
| MOVE index | `MOVE` | `prices.py` | Yahoo Finance (`^MOVE`) | 1h |
| Sector ETFs (XLK, XLF…) | `SECTOR_*` | `prices.py` | Yahoo Finance | 15m |
| Crude Oil | `OIL_WTI` | `prices.py` | Yahoo Finance (`CL=F`) | 15m |
| SPY, QQQ, IWM, DIA | `EQUITY_*` | `prices.py` | Yahoo Finance | 15m |
| CPI, PPI, Payrolls | `CPI_YOY`, `PPI_YOY`, `PAYROLLS` | `bls.py` | BLS API | Daily |
| BTC/ETH/SOL prices | `BTC_USD`, `ETH_USD`, `SOL_USD` | `crypto.py` | CoinGecko API | 15m |
| BTC dominance | `BTC_DOMINANCE` | `crypto.py` | CoinGecko global | 15m |
| Fear & Greed | `FEAR_GREED` | `crypto.py` | Alternative.me API | 1h |
| Stablecoin market cap | `STABLECOIN_MCAP` | `crypto.py` | DefiLlama API | 1h |
| DeFi TVL | `DEFI_TVL` | `crypto.py` | DefiLlama API | 1h |
| BTC ETF net flows | `BTC_ETF_FLOWS_NET` | `etf_flows.py` | Farside HTML scrape | 1h |
| MVRV Z-Score | `MVRV_ZSCORE` | `onchain.py` | CoinGlass open API | Daily |
| Altcoin season index | `ALTCOIN_SEASON` | `onchain.py` | Blockchaincenter HTML | Daily |
| IMF GDP/CPI forecasts | `IMF_GDP_*`, `IMF_CPI_*` | `imf.py` | IMF DataMapper API | Weekly |
| CFTC COT positioning | `COT_SPX_NET`, `COT_GOLD_NET` | `cftc.py` | CFTC CSV ZIP | Weekly |
| Macro/crypto news | — | `news.py` | RSS (Reuters, CNBC, CoinDesk, CoinTelegraph) | 30m |

**Tickers for portfolio pricing** (fetched on-demand when viewing portfolio): any user-entered ticker via Yahoo Finance v8 chart API.

---

## Phase 1 — Project Setup, Database Schema, Scripts

**Backend:**
- Create full directory tree
- Write `db/schema.sql` with all tables and indexes (shown above)
- Write `db/init.py`: check `db/.initialized` flag, run schema.sql via psycopg2, seed `portfolio_cash`, write flag
- Write `config/settings.py`: reads env vars `DATABASE_URL`, `FRED_API_KEY`, `BLS_API_KEY`, `PORT`
- Write `config/sources.json`: all series definitions (id, label, tier, category, collector, refresh_seconds) + RSS feed list
- Write `app/db.py`: `ThreadedConnectionPool(2, 10)`, context-manager `get_conn()`, `fetchone()`, `fetchall()`, `execute()`, `execute_returning()`
- Write `app/__init__.py`: factory function, register all blueprints, register `time_ago` Jinja filter
- Write empty blueprint stubs for all 6 route modules (so Flask starts)
- Write `scripts/start.sh`, `stop.sh`, `collect_now.sh`, `backup_db.sh`
- Write `requirements.txt`

**Frontend:**
- Write `app/templates/base.html`: sidebar HTML (hamburger → sliding panel, all nav links with hover-submenus for Routines and Tiers), topbar, `{% block content %}`, Chart.js script tag, `app.js` script tag
- Write `app/static/css/main.css`: CSS custom properties for all colors; base reset; sidebar/topbar styles; `.card`, `.page-header` gradient text, `.positive`/`.negative` color classes, modal overlay, routine button gradient variants, responsive grid utilities
- Write `app/static/js/app.js`: sidebar open/close, overlay click-to-close, submenu hover show/hide, shared `formatPct(val)` utility

**Verification:** `bash scripts/start.sh` → Flask serves a 200 at `localhost:5000`. Database has all tables. Sidebar opens/closes.

---

## Phase 2 — Data Collection Engine

**Backend:**
- Write `collectors/base.py`: `save_market_data(rows)` using `psycopg2.extras.execute_batch`, `save_news(items)` with `ON CONFLICT (url) DO NOTHING`
- Write `collectors/fred.py`: `collect()` fetches last observation for all 8 FRED series (from `FRED_SERIES` dict), calls `save_market_data()`
- Write `collectors/bls.py`: POST to BLS v2 API with 3 series IDs, parse nested JSON response, compute YoY % for CPI and PPI
- Write `collectors/crypto.py`: `collect()` calls (a) CoinGecko `/global` for BTC dominance and total market cap, (b) CoinGecko `/simple/price` for BTC/ETH/SOL/BNB in one request, (c) Alternative.me `/fng/` for Fear & Greed, (d) DefiLlama for stablecoin cap sum
- Write `collectors/etf_flows.py`: GET Farside page, BeautifulSoup find table, parse rows, sum all numeric columns per date for net daily flow, skip rows with "Total" in first column
- Write `collectors/onchain.py`: CoinGlass public endpoint for MVRV Z-Score; Blockchaincenter page — regex for embedded JSON containing altcoin season score
- Write `collectors/prices.py`: `fetch_yahoo(ticker)` via `query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=5d` with `User-Agent` header; `collect()` fetches all configured tickers, computes `COPPER_GOLD_RATIO` from HG=F / GC=F; `fetch_fundamentals(ticker)` via Yahoo v10 quoteSummary for P/E, P/B, ROE, debt/equity
- Write `collectors/news.py`: `parse_rss(url, source, category)` using `xml.etree.ElementTree`; `email.utils.parsedate_to_datetime` for RFC 2822 dates; deduplicates on URL; caps at 20 items per feed
- Write `collectors/imf.py`: IMF DataMapper REST API for `NGDP_RPCH` (GDP growth) and `PCPIPCH` (CPI) across 9 countries; parse nested `values` dict to extract most recent year
- Write `collectors/cftc.py`: download COT ZIP for current year, extract disaggregated futures file, `csv.DictReader` parse, filter for S&P 500 E-Mini and Gold futures, compute net non-commercial position, delete temp file
- Write `collectors/scheduler.py`: one `threading.Thread` per collector with its own sleep-loop; 2-second stagger between thread starts; catches all exceptions per collector so one failure never kills the scheduler

**Key implementation notes:**
- FRED: check `obs['value'] != '.'` before `float()` — FRED returns `"."` for missing data points
- Yahoo Finance: keep two fallback User-Agent strings; if v8 fails try v7 `/quote`; log failures, continue
- CoinGecko: batch all price requests into one call; 10 req/min limit on free tier
- Farside: sum all numeric td values per row (don't hardcode ETF column names — they add new ETFs)
- CFTC ZIP URL: derive year from `datetime.now().year` — `fut_fin_txt_{year}.zip`
- News: `published_at` can be None if date parse fails — store NULL, don't skip the item

**Verification:** Run `bash scripts/collect_now.sh`. Query `SELECT series_id, value, fetched_at FROM market_data ORDER BY fetched_at DESC LIMIT 20;` — should show real values. Query `SELECT source_name, title FROM news_items LIMIT 10;` — should show headlines.

---

## Phase 3 — Flask Core and Home Page

**Backend:**
- Write `app/models/market_data.py`:
  - `get_latest(series_id)` — `DISTINCT ON (series_id) ORDER BY fetched_at DESC`
  - `get_latest_by_tier(tier)` — all series for a tier, one row per series_id
  - `get_series_history(series_id, days)` — `WHERE recorded_at >= NOW() - INTERVAL 'N days'`
  - `get_biggest_shifts(hours=24)` — CTE joining latest vs. prior-period values, returns top 10 by `ABS(pct_change)`
- Write `app/models/journal.py`:
  - `get_recent_entries(limit)`, `get_entries_grouped()` (grouped by month-year for listing)
  - `get_entry_with_comments(entry_id)` — single query with LEFT JOIN on comments
- Write `app/routes/home.py`: assembles top_stories (latest 6 active news), data_shifts, recent_journal (4 entries), active_session (latest incomplete full_framework session), tier completion map from past sessions

**Frontend:**
- Write `templates/home/index.html` extending `base.html`:
  - **Row 1:** Green-gradient `h1` "Welcome back Dominic!"
  - **Row 2:** 3 icon cards (inline SVGs for flag, globe, bitcoin) linking to `/assets/us`, `/assets/international`, `/assets/crypto`
  - **Row 3:** 3 rectangular routine buttons — Daily (low-opacity red gradient), Weekly (amber), Monthly (green) — linking to `/routines/*`
  - **Row 4:** "Top Stories" section — list of 6 news items with source badge, title (external link), timestamp
  - **Row 5:** "Biggest Data Shifts" — horizontal bar Chart.js chart + table of top 10 moves (green/red bars)
  - **Row 6:** Tier map — 5 clickable nodes in a horizontal flow; completed tiers show green dot + date; the tier the user abandoned shows red; active session shows "Resume →" banner above
  - **Row 7:** "Journal Entries" — grid of clickable date rectangles linking to `/journal/<id>`

**CSS additions:** Gradient button variants (`.routine-daily`, `.routine-weekly`, `.routine-monthly` with `background: linear-gradient(...)` + `opacity: 0.85`); tier map node styles (`.tier-node.completed`, `.tier-node.active`); story card layout; journal rect cards.

**Verification:** Load `localhost:5000` — all 7 sections render with real data from Phase 2. Tier map shows 5 nodes. Charts render.

---

## Phase 4 — Tier Pages and Full Framework Flow

**Backend:**
- Extend `app/models/market_data.py`:
  - `get_tier_data(tier_num, session_id=None)` — returns series list + their 90-day histories + saved interpretations for the session (if session_id provided)
- Write full `app/routes/framework.py`:
  - `GET /framework/` — checks for active `full_framework` session; if found, redirects to `last_active_tier`; else creates new session row, redirects to `/framework/tier/1?session_id=<id>`
  - `GET /framework/tier/<n>` — loads tier data, renders `framework/tier.html` in "framework" mode
  - `POST /framework/tier/<n>` — saves `tier_interpretations` rows (one per `interp_<series_id>` form field); saves `session_portfolio_entries` if ticker/price/qty provided; updates `framework_sessions.completed_tiers` and `last_active_tier`; if `n < 5` redirect to tier `n+1`; if `n == 5` redirect to `/framework/complete/<session_id>`
  - `GET /framework/complete/<session_id>` — renders complete.html with all interpretations and a final `written_text` textarea
  - `POST /framework/complete/<session_id>` — builds `auto_data_snapshot` JSONB from latest market_data for tiers 1–5; builds `interpretations_snapshot` from `tier_interpretations`; builds `portfolio_entries_snapshot`; inserts `journal_entries` row; marks session `is_complete=TRUE, is_active=FALSE`; redirects to `/journal/<entry_id>`
  - `POST /framework/exit/<session_id>` — sets `is_active=FALSE` (preserves progress for resume), redirects to home

**Frontend:**
- Write `templates/framework/start.html` — centered button "Start Framework" with green-to-light-green gradient; if active session exists, shows "Resume from Tier N" button instead
- Write `templates/framework/tier.html` (handles both full-framework mode and standalone mode via `mode` variable):
  - Header: gradient tier title + framework progress dots (T1→T5, colored by completion state)
  - "Exit Framework" button top-right (only in framework mode) — triggers JS modal
  - **Two-column layout** below header: left column "Data", right column "News"
    - Data column: one `.tier-card` per series — shows metric name, current value (colored positive/negative), 90-day sparkline (Chart.js line), data source name above it, freshness indicator, `<textarea>` for interpretation (in session mode)
    - News column: news items filtered by tier — source name above, title, summary snippet, `<textarea>` for interpretation
  - Footer: "Back" link (if tier > 1) + form with hidden interpretation inputs + "Next →" button (tiers 1-4) or "Complete Framework" (tier 5)
  - **Exit modal** (hidden div, shown by JS): "Are you sure?" + Yes/No buttons
- Write `templates/framework/complete.html` — shows all interpretations organized by tier, portfolio entries entered, final `<textarea>` for journal text, "Submit Journal Entry" button (orange-to-light-orange gradient, bold)
- **JS in tier.html:** Before form submit, copy all textarea values to hidden inputs (so the form POST carries them). Exit button shows modal. Sparkline charts initialized per series from `chart_data` JSON passed via `tojson` filter.

**Verification:** Navigate through all 5 tiers entering interpretations. Exit mid-framework, reload — home page shows incomplete session in red on tier map. Resume, complete, verify journal entry created with full snapshot.

---

## Phase 5 — Individual Tier Research Mode

**Backend (additions to `framework.py`):**
- `GET /tiers/<n>` — loads tier data with `session_id=None`, renders `tier.html` in `mode='standalone'`
- `POST /tiers/<n>` — creates a `tier_N` session, saves interpretations, calls `_create_tier_journal_entry(session_id, tier_num, written_text)`, marks session complete, redirects to new journal entry

`_create_tier_journal_entry()` is a shared helper (also used by full framework completion): builds `auto_data_snapshot` for just the one tier, builds `interpretations_snapshot`, inserts `journal_entries` with `entry_type='tier_N'`.

**Frontend:**
- `tier.html` in standalone mode: hides progress bar, hides exit button, shows single "Save & Create Journal Entry" button (orange gradient) at bottom instead of "Next →"
- No popup confirmation needed — standalone tiers have no multi-step flow to interrupt

**Verification:** Click "Tier 2" from sidebar dropdown. Fill interpretations. Submit. Journal entry appears with `entry_type='tier_2'`.

---

## Phase 6 — Routine Research Pages

**Backend:**
- Write `app/routes/routines.py`:
  - `GET /routines/<period>` — calls `_get_routine_data(period)` which assembles appropriate series from the DB; renders corresponding template
  - `POST /routines/<period>` — if `written_text` is non-empty, inserts a `journal_entries` row with `entry_type=period` and `auto_data_snapshot`; redirects to journal
  - `_get_routine_data('daily')`: prices for BTC, ETH, SPY, QQQ, DXY, GOLD; last 24h news; Fear & Greed; VIX
  - `_get_routine_data('weekly')`: all Tier 1 series; BTC ETF flows 30-day history; CFTC COT; sector ETF 1W performance
  - `_get_routine_data('monthly')`: Tier 1+2 series; BLS CPI/PPI/payrolls; IMF GDP forecasts; portfolio summary (total value, P&L)

**Frontend:**
- Three templates (`daily.html`, `weekly.html`, `monthly.html`) — each:
  - Gradient header with routine name
  - Data cards (sparklines where historical data exists)
  - News list (last 24h for daily, last 7d for weekly, last 30d for monthly)
  - Optional journal section at bottom: `<textarea>` + "Save to Journal" submit button + "Skip" link back to home
- Routine pages are **data-only** (no interpretation textareas per item, unlike tier pages)

**Verification:** Visit `/routines/daily` — shows today's prices and news. Submit a note — journal entry created.

---

## Phase 7 — Asset Pages (US, International, Crypto)

**Backend:**
- Write `app/routes/assets.py`:
  - `GET /assets/us` — loads: major US equity indices (SPY/QQQ/IWM/DIA), all 11 sector ETFs with 1W/1M performance, bond ETFs (TLT/IEF/SHY), VIX, T10Y2Y, HY spread, macro news
  - `GET /assets/international` — loads: country ETFs (EWJ/EWZ/EEM/VEA/MCHI/EWG), currency pairs (EUR/USD, GBP/USD, AUD/USD, USD/JPY), IMF GDP/CPI by country, international news
  - `GET /assets/crypto` — loads: BTC/ETH/SOL/BNB/top-20 prices from DB, BTC dominance 90-day history, stablecoin market cap history, Fear & Greed history, ETF flows history, MVRV Z-Score, altcoin season index, crypto news

**Frontend:**
- Each asset template shares this structure:
  - Gradient page title ("US Assets", "International Assets", "Crypto Assets")
  - **Slider toggle** at top: "Data" | "News" — JS class swap between two `<div>` panels
  - **Data panel**: charts-first layout. Every metric with history gets a Chart.js chart (line charts for time series, bar charts for categorical comparisons). US: sector heatmap grid (colored cells by 1M % change using HSL mapping). Crypto: multi-series chart for TOTAL/TOTAL2/TOTAL3 dominance over time.
  - **News panel**: top 10 most recent items from the relevant category + a "Top 5" pinned section for highest-engagement items
- Back button on each page links to home (`/`)

**US Assets sector heatmap:** Each sector ETF is a grid cell with `background-color: hsl(<hue>, 70%, 15%)` where hue is 0 (red) for negative, 120 (green) for positive, scaled by magnitude. Computed in a Jinja2 `sector_color(pct)` filter registered in `app/__init__.py`.

**Verification:** Toggle Data/News slider on each page. Charts render for BTC dominance, sector performance, yield curve history.

---

## Phase 8 — Journal System

**Backend:**
- Write full `app/routes/journal.py`:
  - `GET /journal/` — accepts `?start=`, `?end=`, `?type=` query params; builds filtered SQL; groups results by `TO_CHAR(entry_date, 'YYYY-MM')`; renders `journal/list.html`
  - `GET /journal/<id>` — fetches entry + all comments via LEFT JOIN; renders `journal/entry.html`
  - `POST /journal/<id>/comment` — validates `comment_text` non-empty; inserts into `journal_comments`; redirects back to entry

**Frontend:**
- `templates/journal/list.html`:
  - Search bar at top: date range pickers (HTML `<input type="date">`) + entry type filter dropdown + "Search" button
  - Entries grouped by month-year under gradient `<h2>` headers ("April 2026")
  - Each entry is a clickable rectangle showing date, entry type badge, and first 80 chars of `written_text`
- `templates/journal/entry.html`:
  - Header: entry date + type badge
  - **Auto-populated data section**: rendered from `auto_data_snapshot` JSONB — table of all captured series values at time of entry
  - **Interpretations section**: organized by tier, showing label + user's interpretation text
  - **Portfolio entries section** (if any): table of tickers, prices, quantities entered during the session
  - **Journal text**: `<pre>` wrapped, read-only
  - **Comments section** below: list of existing comments (timestamp + text); "Add Comment" textarea + teal-to-light-blue gradient "Add Comment" button
  - No edit capability — entry text is display-only after creation
  - Back button → `/journal/`

**Verification:** Journal list shows all entries grouped. Clicking an entry shows full snapshot + interpretations. Adding a comment persists and redisplays.

---

## Phase 9 — Portfolio System

**Backend:**
- Write `app/models/portfolio.py`:
  - `get_open_positions(asset_type)` — with current price lookup from latest `market_data` for the ticker
  - `compute_pnl(position, current_price)` — `(current_price - entry_price) * quantity`
  - `get_snapshot_history(asset_type, days)` — for P&L line chart
  - `get_trade_history(asset_type)` — all trades for the asset type, newest first
  - `take_daily_snapshot(asset_type)` — sums open positions × current prices + cash; upserts into `portfolio_snapshots`
- Write full `app/routes/portfolio.py`:
  - `GET /portfolio/` — redirects to `/portfolio/stocks` (default)
  - `GET /portfolio/<asset_type>` — loads open positions with current prices, computes P&L per position, loads snapshots for chart, loads trade history
  - `POST /portfolio/buy` — validates ticker, price, quantity; checks cash >= cost; inserts `portfolio_positions` and `trade_history`; debits `portfolio_cash`; creates `journal_entries` row simultaneously with `entry_type='portfolio'`; redirects back
  - `POST /portfolio/sell` — finds open position by id; validates exit_price; computes pnl and pnl_percent; inserts sell row into `trade_history`; updates position `status='closed'`; credits `portfolio_cash`; updates linked journal entry's `portfolio_entries_snapshot`; redirects back
- Add `take_daily_snapshot` to scheduler — runs once per day at midnight

**Frontend:**
- `templates/portfolio/stocks.html` (reused for crypto with `asset_type` variable):
  - **Slider at top**: "Stocks" | "Crypto" — navigates between `/portfolio/stocks` and `/portfolio/crypto`
  - **Two-box row**:
    - Left box: Chart.js line chart of portfolio total value over time (green line if P&L > 0, red if negative); time-range buttons "1D / 1W / 1M / 1Y" — JS re-renders chart with filtered snapshot data; total portfolio value displayed above chart
    - Right box: Doughnut chart showing allocation percentages by ticker (different colors per position) — `Chart.js type: 'doughnut'`
  - **"Current Positions"** header + list of position rectangles: ticker left-aligned, entry price, quantity, current value, unrealized P&L (green/red); "Sell" button in red on far right
    - Sell button opens a popup (modal div): exit price input + red "Sell" button + P&L preview (computed client-side as user types exit price)
  - **"Trade History"** section: list of all trades, buys in green, sells with P&L in green/red
  - **"Add Position"** button — opens modal: ticker input, entry price, quantity, asset type (auto-set), "Buy" submit button

**Portfolio-journal integration:** Every buy creates a `journal_entries` row with `entry_type='portfolio'` containing the ticker, price, quantity, and the current date. Every sell appends to the existing journal entry for that position (if the buy journal entry still exists in the same day, update `portfolio_entries_snapshot`; else create new row).

**Verification:** Buy 0.1 BTC at $95,000. Portfolio shows open position. Sell at $100,000 — P&L shows +$500. Journal entry created for buy. Portfolio chart updates.

---

## Phase 10 — Integration, Refresh Indicators, and Polish

**Backend:**
- Add `GET /api/data-freshness` JSON endpoint: returns `{series_id: seconds_since_fetch}` for all series — called by frontend every 60s for staleness badges
- Add `GET /api/ticker-price/<ticker>` JSON endpoint: fetches live price from Yahoo Finance v8 on-demand (used by portfolio sell modal for P&L preview)
- Expire old news: add to scheduler a daily cleanup — `UPDATE news_items SET is_active = FALSE WHERE published_at < NOW() - INTERVAL '2 days'`
- Engagement scoring: each time news is viewed (journal snapshot or asset page), increment `engagement_score`. In `get_biggest_shifts`, weight by `engagement_score` for Top Stories ordering.
- Export to HTML: `GET /journal/<id>/export` renders `journal/entry.html` with `?print=1` — strips sidebar/nav, sets print-friendly CSS. Browser print-to-PDF does the rest.

**Frontend polish:**
- Add freshness badges to tier cards: if `seconds_since_fetch > refresh_interval * 2`, show amber dot; if `> refresh_interval * 5`, show red dot
- Home page tier map: if `active_session` exists and `last_active_tier` is set, completed tiers show green, `last_active_tier` shows red (abandoned)
- Portfolio time-range selector: when user clicks "1D", filter `snapshots` client-side from the JSON array already in the page (no extra request)
- Ensure all back buttons present and correct: asset pages → `/`, routine pages → `/`, tier pages (standalone) → `/`, framework tier pages → `/framework/`, journal entry → `/journal/`, portfolio → `/`

**Verification:** Run full framework end-to-end. Verify journal has complete snapshot. Check portfolio P&L chart across day/week/month. Confirm all back buttons navigate correctly. Confirm offline operation (disconnect internet, page still loads).

---

## dependencies.txt Summary

```
flask
psycopg2-binary
requests
beautifulsoup4
```

API keys needed (both free):
- `FRED_API_KEY` — register at fred.stlouisfed.org/api/key
- `BLS_API_KEY` — register at bls.gov/developers/home.htm

---

## End-to-End Verification Checklist

1. `bash scripts/start.sh` → Flask at `:5000`, scheduler logging collection activity
2. Wait 5 minutes → `SELECT COUNT(*) FROM market_data;` returns > 50 rows
3. Home page renders with top stories, data shifts bar chart, tier map, journal rectangles
4. Navigate Full Framework → all 5 tiers → submit journal entry → journal entry page shows full snapshot
5. Navigate to `/tiers/1` standalone → submit → journal entry created with `entry_type='tier_1'`
6. `/routines/daily` → data loads → submit note → journal list shows new entry
7. `/assets/crypto` → data/news toggle works → BTC dominance chart renders
8. `/journal/` → entries grouped by month → search by date filters correctly → click entry → add comment persists
9. `/portfolio/stocks` → buy SPY → position appears with cost basis → sell → P&L calculated → trade history shows both
10. Disconnect internet → reload home page → all pages still functional (data from DB)
