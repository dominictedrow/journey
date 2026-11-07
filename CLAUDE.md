# Market Research Dashboard — CLAUDE.md

## Project Overview

Local-only personal finance research website implementing the **Global Macro Master Investment Framework** (see full framework below). Guides Dominic through a 5-tier investment analysis process, stores market data in PostgreSQL, tracks a paper-trade portfolio ($10k stocks + $10k crypto), and auto-generates journal entries per research session. Runs 100% locally — internet used only for data collection.

---

## Tech Stack

**Backend (Python — 4 packages only):**
- `flask` — routing, Jinja2, Blueprints
- `psycopg2-binary` — PostgreSQL, direct SQL, no ORM
- `requests` — all HTTP/API calls
- `beautifulsoup4` — HTML parsing (Farside, Blockchaincenter, LookIntoBitcoin)

**Desktop-wrapper exception to the "4 packages only" rule** (user-approved, see `app_plan.md`):
- `pywebview` — runtime; renders the dashboard in macOS system WebKit so the `.app` is browser-independent (pulls `pyobjc` on macOS). In `requirements.txt`.
- `pyinstaller` — build-time only; bundles `desktop.py` into `dist/Market Research.app`. In `requirements-build.txt`, not `requirements.txt`.

Stdlib: `xml.etree.ElementTree` (RSS), `csv` (CFTC), `json`, `threading`, `time`, `datetime`, `hashlib`, `os`, `email.utils`

**Frontend:** Chart.js downloaded once to `static/js/chartjs.min.js` (offline). Vanilla JS only.

**Database:** PostgreSQL (local instance).

**Setup commands (run once):**
```bash
curl https://cdn.jsdelivr.net/npm/chart.js/dist/chart.umd.min.js -o app/static/js/chartjs.min.js
```
API keys needed (both free): `FRED_API_KEY` (fred.stlouisfed.org/api/key), `BLS_API_KEY` (bls.gov/developers)

**To open the site:** Always run `PORT=5001 bash scripts/start.sh` then open `http://localhost:5001/` in the browser. Port 5000 is taken by macOS AirPlay Receiver.

---

## Directory Structure

```
market-research/
├── app/
│   ├── __init__.py           # Flask factory, registers blueprints, sector_color/time_ago Jinja filters
│   ├── db.py                 # ThreadedConnectionPool(2,10), get_conn(), fetchone(), fetchall(), execute(), execute_returning()
│   ├── routes/
│   │   ├── home.py           # GET /
│   │   ├── framework.py      # GET/POST /framework/, /framework/tier/<n>, /tiers/<n>, /framework/complete/<id>
│   │   ├── assets.py         # GET /assets/<us|international|crypto>
│   │   ├── routines.py       # GET/POST /routines/<daily|weekly|monthly>
│   │   ├── journal.py        # GET /journal/, /journal/<id>; POST /journal/<id>/comment
│   │   └── portfolio.py      # GET /portfolio/; POST /portfolio/buy, /portfolio/sell
│   ├── models/
│   │   ├── market_data.py    # get_latest(), get_latest_by_tier(), get_series_history(), get_biggest_shifts(), get_tier_data()
│   │   ├── journal.py        # get_recent_entries(), get_entries_grouped(), get_entry_with_comments()
│   │   └── portfolio.py      # get_open_positions(), get_cash_balance(), get_snapshot_history(), compute_pnl(), take_daily_snapshot()
│   ├── templates/
│   │   ├── base.html         # Sidebar + topbar — all pages extend this
│   │   ├── home/index.html
│   │   ├── framework/
│   │   │   ├── start.html
│   │   │   ├── tier.html     # Reused for all 5 tiers + standalone mode
│   │   │   └── complete.html
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
│       ├── css/main.css      # Single stylesheet, all design tokens as CSS variables
│       ├── js/
│       │   ├── chartjs.min.js
│       │   └── app.js        # Sidebar toggle, submenu hover, formatPct() utility
│       └── icons/            # flag.svg, globe.svg, bitcoin.svg
├── collectors/
│   ├── base.py               # save_market_data(), save_news() — shared upsert logic
│   ├── fred.py               # FRED API — T10Y2Y, BAMLH0A0HYM2, M2SL, DGS10, DFII10, DEXJPUS, DEXUSAL, WALCL
│   ├── bls.py                # BLS API — CPI (CUUR0000SA0), PPI (WPUFD49104), Payrolls (CES0000000001)
│   ├── crypto.py             # CoinGecko global+prices, Alternative.me Fear&Greed, DefiLlama stablecoins/TVL
│   ├── etf_flows.py          # Farside Bitcoin ETF flows (BeautifulSoup HTML table)
│   ├── onchain.py            # CoinGlass MVRV; Blockchaincenter altcoin season (BeautifulSoup)
│   ├── prices.py             # Yahoo Finance v8 hidden JSON API for stocks/ETFs/commodities/FX
│   ├── news.py               # RSS feeds via xml.etree.ElementTree
│   ├── imf.py                # IMF DataMapper REST API — GDP, CPI, current account (9 countries)
│   ├── cftc.py               # CFTC COT weekly CSV ZIP
│   └── scheduler.py          # threading.Thread per collector, 2s stagger, exceptions isolated per thread
├── db/
│   ├── schema.sql            # Full schema, IF NOT EXISTS (idempotent)
│   └── init.py               # Checks .initialized flag, runs schema.sql, seeds portfolio_cash
├── config/
│   ├── settings.py           # DATABASE_URL, FRED_API_KEY, BLS_API_KEY, FLASK_PORT from env
│   └── sources.json          # Series definitions (id, label, tier, category, collector, refresh_seconds) + RSS feeds
├── scripts/
│   ├── start.sh              # db/init.py → scheduler.py (bg) → flask run
│   ├── stop.sh               # Kills .scheduler.pid and .flask.pid
│   ├── collect_now.sh        # Runs all collectors immediately
│   └── backup_db.sh          # pg_dump → backups/market_research_YYYYMMDD.sql
└── requirements.txt          # flask psycopg2-binary requests beautifulsoup4
```

---

## Database Schema

```sql
CREATE TABLE market_data (
    id           BIGSERIAL PRIMARY KEY,
    series_id    TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    value        NUMERIC,
    text_value   TEXT,
    source       TEXT NOT NULL,
    category     TEXT,           -- macro|credit|fx|volatility|crypto|equity|bond|commodity
    tier         INTEGER,        -- 1-5; NULL = cross-tier
    recorded_at  TIMESTAMPTZ NOT NULL,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_md_series_fetched ON market_data(series_id, fetched_at DESC);
CREATE INDEX idx_md_tier_fetched   ON market_data(tier, fetched_at DESC);

CREATE TABLE news_items (
    id               BIGSERIAL PRIMARY KEY,
    source_name      TEXT NOT NULL,
    title            TEXT NOT NULL,
    url              TEXT NOT NULL UNIQUE,
    summary          TEXT,
    published_at     TIMESTAMPTZ,
    category         TEXT,
    tier             INTEGER,
    engagement_score INTEGER DEFAULT 0,
    is_active        BOOLEAN DEFAULT TRUE,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_news_cat_pub ON news_items(category, published_at DESC);
CREATE INDEX idx_news_active   ON news_items(is_active, engagement_score DESC);

CREATE TABLE framework_sessions (
    id               BIGSERIAL PRIMARY KEY,
    session_type     TEXT NOT NULL,   -- full_framework|tier_1..tier_5|daily|weekly|monthly
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    last_active_tier INTEGER,
    completed_tiers  INTEGER[] DEFAULT '{}',
    is_active        BOOLEAN DEFAULT TRUE,
    is_complete      BOOLEAN DEFAULT FALSE
);

CREATE TABLE tier_interpretations (
    id              BIGSERIAL PRIMARY KEY,
    session_id      BIGINT NOT NULL REFERENCES framework_sessions(id) ON DELETE CASCADE,
    tier            INTEGER NOT NULL,
    data_item_key   TEXT NOT NULL,
    data_item_label TEXT NOT NULL,
    interpretation  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_interp_session ON tier_interpretations(session_id, tier);

CREATE TABLE session_portfolio_entries (
    id          BIGSERIAL PRIMARY KEY,
    session_id  BIGINT NOT NULL REFERENCES framework_sessions(id),
    ticker      TEXT NOT NULL,
    entry_price NUMERIC(14,4),
    quantity    NUMERIC(14,4),
    asset_type  TEXT NOT NULL
);

CREATE TABLE journal_entries (
    id                         BIGSERIAL PRIMARY KEY,
    session_id                 BIGINT REFERENCES framework_sessions(id),
    entry_date                 DATE NOT NULL DEFAULT CURRENT_DATE,
    entry_type                 TEXT NOT NULL, -- full_framework|tier_1..5|daily|weekly|monthly|portfolio
    tier_number                INTEGER,
    auto_data_snapshot         JSONB,
    interpretations_snapshot   JSONB,
    portfolio_entries_snapshot JSONB,
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

CREATE TABLE portfolio_positions (
    id               BIGSERIAL PRIMARY KEY,
    asset_type       TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    entry_price      NUMERIC(14,4) NOT NULL,
    quantity         NUMERIC(14,4) NOT NULL,
    cost_basis       NUMERIC(14,4) NOT NULL,
    entry_date       DATE NOT NULL DEFAULT CURRENT_DATE,
    status           TEXT NOT NULL DEFAULT 'open',
    journal_entry_id BIGINT REFERENCES journal_entries(id),
    session_id       BIGINT REFERENCES framework_sessions(id)
);
CREATE INDEX idx_pos_status_type ON portfolio_positions(status, asset_type);

CREATE TABLE trade_history (
    id          BIGSERIAL PRIMARY KEY,
    position_id BIGINT NOT NULL REFERENCES portfolio_positions(id),
    trade_type  TEXT NOT NULL,
    price       NUMERIC(14,4) NOT NULL,
    quantity    NUMERIC(14,4) NOT NULL,
    trade_value NUMERIC(14,4) NOT NULL,
    pnl         NUMERIC(14,4),
    pnl_percent NUMERIC(8,4),
    trade_date  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_trades_date ON trade_history(trade_date DESC);

CREATE TABLE portfolio_cash (
    asset_type TEXT PRIMARY KEY,
    balance    NUMERIC(14,4) NOT NULL DEFAULT 10000.00
);
INSERT INTO portfolio_cash VALUES ('stock', 10000.00), ('crypto', 10000.00) ON CONFLICT DO NOTHING;

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

## Data Series Map

| Series ID | Collector | Source | Refresh |
|---|---|---|---|
| `T10Y2Y`, `BAMLH0A0HYM2`, `M2SL`, `DGS10`, `DFII10`, `DEXJPUS`, `DEXUSAL`, `WALCL` | `fred.py` | FRED API | 1h |
| `CPI_YOY`, `PPI_YOY`, `PAYROLLS` | `bls.py` | BLS API | Daily |
| `VIX`, `DXY`, `COPPER`, `GOLD`, `COPPER_GOLD_RATIO`, `MOVE`, `SECTOR_*`, `OIL_WTI`, `EQUITY_*` | `prices.py` | Yahoo Finance v8 | 15m |
| `BTC_USD`, `ETH_USD`, `SOL_USD`, `BTC_DOMINANCE` | `crypto.py` | CoinGecko | 15m |
| `FEAR_GREED` | `crypto.py` | Alternative.me | 1h |
| `STABLECOIN_MCAP`, `DEFI_TVL` | `crypto.py` | DefiLlama | 1h |
| `BTC_ETF_FLOWS_NET` | `etf_flows.py` | Farside HTML scrape | 1h |
| `MVRV_ZSCORE` | `onchain.py` | CoinGlass open API | Daily |
| `ALTCOIN_SEASON` | `onchain.py` | Blockchaincenter HTML | Daily |
| `IMF_GDP_*`, `IMF_CPI_*` | `imf.py` | IMF DataMapper API | Weekly |
| `COT_SPX_NET`, `COT_GOLD_NET` | `cftc.py` | CFTC CSV ZIP | Weekly |

**Collector gotchas:**
- FRED: skip obs where `value == '.'` before calling `float()`
- Yahoo Finance: two fallback User-Agent strings; if v8 fails try v7 `/quote`; log, continue
- CoinGecko: batch all price requests into one call; 10 req/min free tier
- Farside: sum all numeric td values per row (don't hardcode ETF columns — they add new ones)
- CFTC ZIP URL: `fut_fin_txt_{datetime.now().year}.zip`
- News: `published_at` can be None on parse failure — store NULL, don't skip the item
- `DISTINCT ON (series_id) ORDER BY fetched_at DESC` is the pattern for "get latest" queries

---

## Build Phases

### Phase 1 — Setup, Schema, Scripts
- Create directory tree; write `db/schema.sql`, `db/init.py` (idempotent via `.initialized` flag)
- Write `config/settings.py`, `config/sources.json`
- Write `app/db.py` (ThreadedConnectionPool), `app/__init__.py` (factory + blueprints + Jinja filters)
- Write empty blueprint stubs for all 6 route modules
- Write `scripts/start.sh`, `stop.sh`, `collect_now.sh`, `backup_db.sh`; `requirements.txt`
- Download Chart.js; write `base.html` (sidebar hamburger → sliding panel, hover-submenus for Routines + Tiers), `main.css` (CSS vars, `.card`, `.positive`/`.negative`, modal, gradient button variants), `app.js` (sidebar, submenu, `formatPct()`)
- **Check:** `bash scripts/start.sh` → 200 at `localhost:5000`, all tables exist, sidebar opens

### Phase 2 — Data Collection
- Write all collectors (see directory map above) and `collectors/scheduler.py`
- **Check:** `bash scripts/collect_now.sh` → `SELECT series_id, value FROM market_data ORDER BY fetched_at DESC LIMIT 20` returns real values; news_items has headlines

### Phase 3 — Models + Home Page
- Write `app/models/market_data.py`: `get_latest()`, `get_latest_by_tier()`, `get_series_history()`, `get_biggest_shifts()`
- Write `app/models/journal.py`: `get_recent_entries()`, `get_entries_grouped()`, `get_entry_with_comments()`
- Write `app/routes/home.py`: top_stories (6), data_shifts, recent_journal (4), active_session, tier completion map
- Write `home/index.html` — 7 rows: greeting, 3 asset icon-cards, 3 routine buttons, top stories, biggest shifts bar chart, tier map (5 nodes), journal date rects
- **Check:** All 7 sections render with real data; Chart.js bar chart renders

### Phase 4 — Tier Pages + Full Framework Flow
- Extend `market_data.py` with `get_tier_data(tier_num, session_id=None)`
- Write full `framework.py`:
  - `GET /framework/` — resumes or creates session
  - `GET/POST /framework/tier/<n>` — saves interpretations + portfolio entries, advances tiers
  - `GET/POST /framework/complete/<session_id>` — builds snapshots, creates journal entry, marks session complete
  - `POST /framework/exit/<session_id>` — saves progress, goes home
- Write `start.html`, `tier.html` (two-column: data left, news right; sparklines; textarea per series; exit modal), `complete.html`
- JS in tier.html: copy textareas to hidden inputs before submit; sparklines from `chart_data|tojson`
- **Check:** Navigate all 5 tiers; exit mid-flow; home shows incomplete session red on tier map; resume works; journal entry has full snapshot

### Phase 5 — Standalone Tier Mode
- Add `GET/POST /tiers/<n>` to `framework.py` (mode='standalone', no progress bar, single "Save & Create Journal Entry" button)
- Shared `_create_tier_journal_entry()` helper used by both standalone and full framework
- **Check:** `/tiers/2` → fill → submit → `entry_type='tier_2'` in journal

### Phase 6 — Routine Pages
- Write `routines.py` with `GET/POST /routines/<period>` and `_get_routine_data(period)`:
  - Daily: BTC/ETH/SPY/QQQ/DXY/GOLD prices, 24h news, Fear & Greed, VIX
  - Weekly: all Tier 1 series, BTC ETF flows 30d, CFTC COT, sector ETF 1W performance
  - Monthly: Tier 1+2, BLS data, IMF GDP forecasts, portfolio summary
- Three templates: gradient header, data cards with sparklines, news list, optional journal textarea + "Save to Journal" / "Skip"
- Data-only: no per-item interpretation textareas
- **Check:** `/routines/daily` shows prices + news; submit creates journal entry

### Phase 7 — Asset Pages
- Write `assets.py`:
  - US: equity indices, 11 sector ETFs (1W/1M perf), bond ETFs, VIX, T10Y2Y, HY spread, macro news
  - International: country ETFs, FX pairs, IMF GDP/CPI, international news
  - Crypto: top prices, BTC dominance history, stablecoin cap, Fear & Greed, ETF flows, MVRV, altcoin season, crypto news
- Three templates: slider toggle "Data|News", charts-first layout, US sector heatmap (`hsl(0-120, 70%, 15%)` by % change), TOTAL/TOTAL2/TOTAL3 multi-series chart
- `sector_color(pct)` Jinja filter registered in `__init__.py`
- **Check:** Data/News toggle on each page; BTC dominance chart renders

### Phase 8 — Journal System
- Write full `journal.py` routes: `GET /journal/` (filtered by `?start=&end=&type=`), `GET /journal/<id>`, `POST /journal/<id>/comment`
- `list.html`: date pickers + type filter dropdown, entries grouped by month under gradient `<h2>` headers, clickable rects
- `entry.html`: auto-data table from `auto_data_snapshot` JSONB, interpretations by tier, portfolio entries, `<pre>` read-only text, comments + "Add Comment" button (teal gradient), back → `/journal/`
- **Check:** Grouped list renders; click entry shows snapshot + interpretations; comment persists

### Phase 9 — Portfolio System
- Write full `portfolio.py` model and routes
- `GET /portfolio/<asset_type>`: open positions with live P&L, snapshot chart (line: green if P&L>0, else red), doughnut allocation, trade history
- `POST /portfolio/buy`: validate, check cash, insert position + trade_history, debit cash, create `journal_entries` row
- `POST /portfolio/sell`: compute pnl, insert sell trade, close position, credit cash, update journal snapshot
- Sell modal: exit price input + client-side P&L preview
- Time-range buttons (1D/1W/1M/1Y): filter snapshot JSON client-side (no extra request)
- Add `take_daily_snapshot` to scheduler (midnight)
- **Check:** Buy 0.1 BTC at $95k; P&L shows; sell at $100k → +$500; journal entry exists

### Phase 10 — Polish
- `GET /api/data-freshness` JSON: `{series_id: seconds_since_fetch}`, polled every 60s for staleness badges (amber >2× interval, red >5× interval)
- `GET /api/ticker-price/<ticker>`: on-demand Yahoo Finance for portfolio sell modal
- Daily scheduler cleanup: `UPDATE news_items SET is_active=FALSE WHERE published_at < NOW() - INTERVAL '2 days'`
- `GET /journal/<id>/export`: strips sidebar for print-to-PDF
- Verify all back-button routes: asset → `/`, routines → `/`, tier standalone → `/`, framework tier → `/framework/`, journal entry → `/journal/`, portfolio → `/`
- **Final check:** Full framework end-to-end; journal has complete snapshot; portfolio P&L chart; offline operation (disconnect internet, all pages still load from DB)

---

## Global Macro Master Investment Framework

*This is the analytical framework the app implements. Refer to it when naming tiers, labeling data, writing template copy, and structuring the framework flow.*

### Tier Structure

| Tier | Process | Frequency | Output |
|---|---|---|---|
| 1 | Macro regime classification | Weekly + post-CB meeting | Risk-on/off backdrop |
| 2A | Country & region scoring | Monthly | Overweight/underweight countries |
| 2B | Crypto regime & cycle | Weekly | BTC or altcoin phase |
| 3 | Sector & asset-class rotation | Monthly + regime change | Sector overweights |
| 4 | Individual selection (stocks + crypto) | Monthly | Entry candidates |
| 5 | Risk management & monitoring | Weekly ongoing | Exit triggers & sizing |

**Rule:** Never select individual stocks or crypto without completing Tiers 1–3 first.

---

### Central Bank Primer

| Bank | Mandate | Tools | Watch |
|---|---|---|---|
| Fed | Dual: employment + 2% inflation | Fed funds rate, QE/QT | FOMC dates, dot plot, H.4.1 balance sheet |
| ECB | Single: 2% price stability | Deposit rate, TPI | BTP-Bund spread >200bps = ECB risk |
| BoJ | 2% price stability | YCC, rate | USD/JPY drop >10% in 30d = carry unwind |
| PBoC | Growth + stability + FX | RRR, LPR, CNY fix | Total Social Financing, USD/CNY fix |

**CB Divergence Matrix:**

| | Fed Tightening | Fed Holding | Fed Easing |
|---|---|---|---|
| BoJ Easing | USD surges, JPY crashes | USD/JPY stable, carry attractive | Best EM period |
| BoJ Tightening | Major carry unwind. Reduce book. | Watch USD/JPY >10% drop | Yen repatriation offsets US easing — volatile |
| PBoC Easing | Commodities squeezed, EM stress | EM mixed, AUD/USD as proxy | Powerful risk-on: commodity surge + EM rally |
| PBoC Tightening | Deflationary global pressure | China drag, Copper/AUD fall | Net EM positive but volatile |

---

### Tier 1 — Macro Regime Classification

**Four-Quadrant Regime Matrix:**

| Regime | Condition | Overweight | Underweight | Confirmation |
|---|---|---|---|---|
| 🟢 Goldilocks | Growth↑ + Inflation↓ | Equities, EM, Tech, Credit | USD, Gold, Commodities | VIX <15, HY tight, curve steepening |
| 🟡 Reflation | Growth↑ + Inflation↑ | Commodities, Energy, Financials, TIPS | Long-duration bonds, Utilities, REITs | CRB rising, 10yr rising, break-evens up |
| 🔴 Stagflation | Growth↓ + Inflation↑ | Gold, Energy, Agri, TIPS | Equities, EM, Tech | PMI <50 + CPI >5%, flat/inverted curve |
| ⚫ Deflation | Growth↓ + Inflation↓ | Long-duration Treasuries, USD, Defensives | Equities, Commodities, EM debt | Curve inverted 12+ months, HY blowing out |

**Three Composites:**

**Composite 1 — Monetary Conditions (highest weight)**
- Real rate = Fed funds rate − CPI. Positive AND rising = genuinely restrictive.
- Fed balance sheet trend (H.4.1): QT drains liquidity even when rate unchanged.
- Dot plot = forward guidance. Watch for pivot language.
- Global M2 (lynalden.com monthly): leads global equities ~6 months.
- Sources: federalreserve.gov · fred.stlouisfed.org (M2SL, DGS10) · lynalden.com

**Composite 2 — Credit & Risk Appetite**
- HY spread (BAMLH0A0HYM2): CRITICAL. Widening >150bps in 6 weeks = reduce equities.
- 2s/10s yield curve (T10Y2Y): inversion signals recession. Early cycle steepening = bullish banks.
- VIX: <15 complacent, 15–25 normal, >35 panic/structural stress.
- MOVE Index: bond VIX. Sustained >120 precedes equity volatility.
- Sources: fred.stlouisfed.org (BAMLH0A0HYM2, T10Y2Y, DFII10) · cboe.com/vix

**Composite 3 — Dollar Regime & Cross-Asset**
- DXY vs 50d/200d MA. Spike >5% in 15 trading days = risk-off.
- Copper/gold ratio: rising = growth > fear. Falling = fear > growth.
- AUD/USD: China demand proxy. Strengthening = China healthy + global risk-on.
- USD/JPY: rapid fall >10% in 30d = carry unwind = regime-change trigger.
- Sources: tradingview.com · lme.com (copper)

**Cross-Asset Quick Reference:**

| Signal | Bullish | Bearish | Source |
|---|---|---|---|
| 2s/10s Yield Curve | Positive & steepening | Inverted >6 months | fred.stlouisfed.org |
| HY Credit Spread | Tight & narrowing (<400bps) | Widening >150bps in 6 weeks | fred.stlouisfed.org |
| VIX | Below 15 and falling | Above 30 and rising | cboe.com/vix |
| DXY | Falling / below 200d MA | Rising fast / above 200d MA | tradingview.com |
| Copper/Gold Ratio | Rising with equities | Falling | lme.com |
| USD/JPY | Rising (carry being added) | Rapid fall >10% in 30d | tradingview.com |
| Global M2 | Expanding YoY | Contracting YoY | lynalden.com |

---

### Tier 2A — Country & Region Scoring

Score 1–5 per dimension. Total >16 = OVERWEIGHT · 10–16 = NEUTRAL · <10 = UNDERWEIGHT/AVOID.

**Dimension 1 — Growth Trajectory**
- PMI Composite above/below 50 and direction; GDP vs trend, surprise vs consensus
- Sources: spglobal.com/pmi · data.oecd.org · tradingeconomics.com

**Dimension 2 — Inflation & CB Stance**
- Core CPI vs target; PPI leads CPI 3–6 months; CB easing room
- Sources: tradingeconomics.com · federalreserve.gov · ecb.europa.eu · boj.or.jp

**Dimension 3 — Financial Health (Current Account + Fiscal)**
- Surplus countries (Germany, Japan, Korea) are creditors — currencies strengthen in stress
- Debt/GDP trend; dollar-denominated debt + deteriorating CA = red flag
- Sources: imf.org/en/Publications/FM · imf.org COFER

**Dimension 4 — Political Risk & Capital Flows**
- Election calendar, CB independence, sanctions; ETF fund flows; CFTC COT extremes = contrarian
- Sources: cfr.org/global-conflict-tracker · etf.com · cftc.gov/COT · pages.stern.nyu.edu/~adamodar

Countries: United States, Eurozone, Japan, China, India, Brazil, Mexico, South Korea, Australia

---

### Tier 2B — Crypto Regime & Cycle

Always apply Tier 1 results as the first overlay.

**Phase 1 — Cycle Position (BTC Halving + BTC.D)**

| Cycle Phase | BTC.D | Strategy |
|---|---|---|
| Pre-halving accumulation | Rising/high | Accumulate BTC only |
| Post-halving early bull | Rising | Overweight BTC, watch ETH/BTC |
| Mid bull expansion | Peaking, rolling | Rotate BTC → ETH → large-cap L1s |
| Late bull blow-off | Falling hard | Mid/small alts, begin reduction |
| Bear / post-peak | Irrelevant | Stablecoins and BTC only |

Sources: lookintobitcoin.com · tradingview.com (BTC.D, ETHBTC, TOTAL/TOTAL2/TOTAL3)

**Phase 2 — Macro Overlay**

| Signal | Bullish | Bearish |
|---|---|---|
| DXY | Falling below 200d MA | Spike >5% in 15d |
| BTC ETF net flows | Consistent daily net inflows | Net outflows 3+ consecutive days |
| Stablecoin market cap | Rising (dry powder) | Falling (exiting) |
| Global M2 | Expanding YoY | Contracting YoY |
| Fear & Greed | Extreme Fear (0–25): accumulate | Greed (75–100): reduce |

**Phase 3 — Crypto Sector Rotation**

| Phase | Buy | Avoid |
|---|---|---|
| Early bull (BTC.D rising) | BTC primarily, small ETH | All altcoins |
| Mid bull (BTC.D peaking) | ETH, SOL, AVAX; watch DeFi TVL | Meme coins, micro-caps |
| Late bull (TOTAL3 surging) | Selective mid-caps with on-chain growth | Overleveraged — trim funding |
| Bear | Stablecoins, BTC if holding | All altcoins |

**Phase 4 — Position Sizing**

| Conviction | BTC max | ETH/L1 max | Altcoin max |
|---|---|---|---|
| High (macro tailwind + ETF inflows + BTC.D + tech) | 5% | 3% | 2% |
| Medium (2–3 factors) | 3% | 2% | 1% |
| Low (1 factor / conflicting) | 2% | 1% | 0.5% |

Exit: MVRV Z-Score >5–6 → begin systematic crypto reduction. Leverage: extreme positive funding + high OI = squeeze risk.

---

### Tier 3 — Sector & Asset-Class Rotation

**Business Cycle Sector Clock:**

| Phase | Overweight | Avoid | US ETFs |
|---|---|---|---|
| Early Cycle | Financials, Discretionary, Real Estate, Industrials | Energy, Utilities, Staples | XLF · XLY · XLRE · XLI |
| Mid Cycle | Technology, Industrials, Materials, Comm Services | Real Estate, Utilities | XLK · XLI · XLB · XLC |
| Late Cycle | Energy, Healthcare, Staples, Materials | Discretionary, Real Estate, Financials | XLE · XLV · XLP · XLB |
| Recession | Utilities, Healthcare, Staples, Long-duration bonds | All cyclicals, Discretionary, EM | XLU · XLV · XLP · TLT |

**Commodity Overlay:**

| Commodity | Demand-led | Supply-shock |
|---|---|---|
| Oil WTI/Brent | Reflationary | Stagflationary |
| Copper | Rising = industrial activity accelerating | Falling = slowdown |
| Gold | Rallies when real yields fall / DXY weakens | Fails to rally with rising real yields |

---

### Tier 4 — Individual Selection

**4-Factor Checklist (require ≥3 of 4):**

| Factor | Criteria | Source |
|---|---|---|
| Quality | ROE >15%, ROIC > WACC, Debt/Equity below sector median, positive FCF 3+ years | macrotrends.net · tikr.com · stockanalysis.com |
| Valuation | Forward P/E below 5yr avg AND sector median; check EV/EBITDA and P/FCF | finviz.com · tikr.com · koyfin.com |
| Earnings Momentum + Catalyst | Positive EPS surprise last 2 qtrs + upward revisions + near-term catalyst | seekingalpha.com · tipranks.com |
| Price Momentum | Above 200d MA, RS vs local index improving, base or breakout forming | tradingview.com |

**Regime-Filtered Rules:**

| Regime | Bias | Avoid |
|---|---|---|
| 🟢 Goldilocks | High-growth, high-multiple tech + EM | Nothing — risk-on |
| 🟡 Reflation | Commodity producers, energy, banks | Unprofitable growth (rates compress multiples) |
| 🔴 Stagflation | Highest-quality, lowest-debt, pricing power | EM and tech |
| ⚫ Recession | Capital preservation, cash, inelastic demand | All cyclicals, discretionary, financials, EM |

---

### Tier 5 — Risk Management

**Position Sizing Rules:**

| Instrument | Max Size |
|---|---|
| Single country bet | 15% of book |
| Single sector bet | 25% of book |
| Single stock | 5% at cost |
| BTC | 3–5% of portfolio |
| Single altcoin | 1–3% |

**Regime-Change Triggers (re-run Tier 1 immediately):**

| Trigger | Threshold | Action |
|---|---|---|
| HY credit spreads | Widen >150bps in 6 weeks | Reduce equity exposure |
| USD/JPY carry unwind | Drop >10% in 30 days | Reduce entire book |
| DXY spike | >5% in 15 trading days | Reduce EM, commodities, crypto; raise USD cash |
| Fed pivot (surprise) | Hike or cut outside cycle | Re-run Tier 1 |
| G7 PMI deterioration | 2+ G7 economies below 48 | Shift late-cycle defensive |
| Oil spike | >25% in 60 days | Stagflationary — overweight energy + gold |
| BTC ETF outflows | 3+ consecutive days net | Reduce total crypto |
| MVRV Z-Score | Above 6 | Begin systematic crypto reduction |

**Monitoring Cadence:**
- **Daily (~10 min):** DXY direction, VIX thresholds (15/25/30), HY spread vs prior day, BTC ETF flows, headlines, open positions vs stops
- **Weekly (~60 min, Sunday):** Full Tier 1 re-run; crypto BTC.D/ETH/BTC/Fear&Greed/MVRV/ETF flows 7d; CB speeches; CFTC COT; earnings surprises; funding rates
- **Monthly (~2h, first weekend):** Re-score countries; adjust sector weights; 4-factor checklist on holdings + candidates; CAPE valuations; MVRV trajectory; full IMF WEO review (April + October); portfolio attribution; top 3 tail risks

---

## Data Source Reference

### Free Public APIs (no key)

| Source | Endpoint | Data |
|---|---|---|
| Alternative.me | `https://api.alternative.me/fng/` | Crypto Fear & Greed |
| Alternative.me history | `https://api.alternative.me/fng/?limit=30` | Last 30 days Fear & Greed |
| DefiLlama | `https://stablecoins.llama.fi/stablecoins` | Stablecoin market caps |
| DefiLlama TVL | `https://api.llama.fi/v2/chains` | TVL by chain |
| IMF DataMapper | `https://www.imf.org/external/datamapper/api/v1/` | GDP, inflation, fiscal |
| CFTC COT | `https://www.cftc.gov/dea/newcot/c_disagg.txt` | COT CSV |

### API Key Required (free tier)

| Source | Base URL | Data |
|---|---|---|
| FRED | `https://api.stlouisfed.org/fred/series/observations` | T10Y2Y, BAMLH0A0HYM2, M2SL, DGS10, DFII10 |
| BLS | `https://api.bls.gov/publicAPI/v2/timeseries/data/` | CPI, PPI, Payrolls |

### FRED Series IDs
```
T10Y2Y        — 2s/10s yield curve spread
BAMLH0A0HYM2  — HY credit spread (OAS)
M2SL          — US M2 money supply
DGS10         — 10-year Treasury yield
DFII10        — 10-year TIPS (real yield)
DEXJPUS       — USD/JPY exchange rate
DEXUSAL       — AUD/USD exchange rate
WALCL         — Fed balance sheet
```

### BLS Series IDs
```
CUUR0000SA0   — CPI All Urban Consumers
WPUFD49104    — PPI Final Demand
CES0000000001 — Total Nonfarm Payrolls
```

### Web Scraping Targets

| Source | URL | Data |
|---|---|---|
| Farside Bitcoin ETF Flows | `https://farside.co.uk/bitcoin-etf-flow/` | Daily ETF net flows table |
| Blockchaincenter Altcoin Index | `https://blockchaincenter.net/altcoin-season-index/` | Altcoin season score |
| LookIntoBitcoin MVRV | `https://lookintobitcoin.com/charts/mvrv-zscore/` | MVRV Z-Score chart data |

### All Reference Websites

| Category | Sites |
|---|---|
| Global macro | lynalden.com · federalreserve.gov · fred.stlouisfed.org · cboe.com/vix · tradingview.com · lme.com · cftc.gov/COT · koyfin.com · tradingeconomics.com |
| International | imf.org/en/Publications/WEO · imf.org/en/Publications/FM · spglobal.com/pmi · data.oecd.org · pages.stern.nyu.edu/~adamodar · starcapital.de/en/research · cfr.org/global-conflict-tracker |
| US markets | bls.gov · ismworld.org · finviz.com · advantage.factset.com · stockcharts.com · macrotrends.net · tikr.com · stockanalysis.com · seekingalpha.com · tipranks.com · etf.com · cmegroup.com |
| Crypto | lookintobitcoin.com · farside.co.uk/bitcoin-etf-flow · alternative.me/crypto/fear-and-greed-index · coinglass.com · defillama.com · blockchaincenter.net |
| Paid/limited | tradingview.com (raw data) · spglobal.com/pmi · lme.com · finviz.com (Elite for exports) · seekingalpha.com · tipranks.com |

---

## Glossary

| Term | Meaning |
|---|---|
| BTC.D | Bitcoin Dominance — BTC mcap as % of total crypto. Rising = early cycle. Falling = altcoin season. |
| CAPE | Cyclically Adjusted P/E (Shiller P/E) — 10yr inflation-adjusted earnings multiple |
| COT | Commitments of Traders — CFTC weekly futures positioning. Extremes = contrarian signal. |
| DXY | US Dollar Index vs basket of 6 major currencies. Strong = EM stress, commodity pressure. |
| FCF | Free Cash Flow — cash after capex. Quality indicator. |
| HY/OAS | High Yield / Option-Adjusted Spread — junk bond premium over Treasuries. Leads equities ~2 weeks. |
| MVRV | Market Value to Realized Value — Bitcoin on-chain ratio. Z-Score >7 = bubble. <0 = deep value. |
| QE/QT | Quantitative Easing/Tightening — CB balance sheet expansion vs contraction. |
| ROIC | Return on Invested Capital. ROIC > WACC = value creation. |
| TIPS/DFII10 | 10yr real yield — hurdle rate for gold and growth stocks. |
| VIX | S&P 500 implied 30-day volatility. <15 complacent, >35 panic. |
| YCC | Yield Curve Control (BoJ) — caps 10yr JGB yield via unlimited bond buying. |
