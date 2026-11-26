# Component 02: Collectors (point-in-time)

> Build spec for Phase 2. Written from Dom's answers in [`docs/interrogation/02_collectors.md`](../interrogation/02_collectors.md) (S02, 2026-11-25).
> Where this file and `docs/spec/system_spec_v2.md` disagree, this file wins. Decisions: D-021 to D-026.
> Items marked **(default)** were not asked in the questioning; Claude picked them; **Dom accepted all of them** (S02, 2026-11-25).

## 1. Purpose
This phase gets every raw input into the DB, append-only and point-in-time correct, and keeps it flowing unattended:
- backfill about 10 years of prices plus full macro history
- a scheduler that runs the news loop every 15 minutes and the daily and weekly collects
- a `released_at` on every row that the Phase 3 feature layer can trust blindly

It contains no features, no screening logic beyond storing membership, no SVM/LLM, and no trading.

## 2. Scope
| In Phase 2 | Deferred |
|---|---|
| Alpaca bars (stocks, ETFs, crypto) + Assets API | Weekly factor screen that picks the ~80 candidates → Phase 3 (Phase 2 stores the inputs and `universe_size` in config) |
| FRED/ALFRED all vintages, HY OAS archive | `BAA10Y` ↔ HY OAS switching rules → Phase 3 (D-001) |
| yfinance index-level series (pre-2016 back to ~2000) and the spec's yfinance tickers | Quota, SVM, n-grams, LLM pass → Phase 8 |
| EDGAR companyfacts + submissions (acceptance times) | NO TRADE gating on stale feeds → Phase 5 (Phase 2 *records* freshness and criticality) |
| S&P 500 membership: fja05680 load + our weekly snapshots | Crypto order-size cap volume source → Phase 6 (D-021) |
| Aliases table + `configs/aliases.yaml` | GDELT news backfill (optional, unscheduled) |
| News: 5 RSS feeds every 15 min, trafilatura, ticker tagging, exact + near dedup, raw HTML archive | |
| Fed RSS (statements, minutes, speeches) + FOMC calendar | |
| Congress.gov bill actions | |
| Earnings calendar (Nasdaq, Finnhub fallback) | |
| Crypto structure: CoinGecko volume + dominance, DefiLlama stablecoins, funding/OI (4 venues), SoSoValue (if free) | |
| APScheduler service + launchd agent (Mac) | Windows Task Scheduler registration → Phase 12 |
| Daily summary content (freshness, rows collected) | |

## 3. Point-in-time contract (the core of this phase)
Every raw row carries `released_at`: the earliest instant the system could have known it. Phase 3 features for decision time T read only rows with `released_at <= T`. The table below is the authoritative rule for each source, and each row has a test (§12).

| Source | `ts` / period | `released_at` |
|---|---|---|
| Alpaca stock/ETF daily bar | session date | XNYS `session_close(date)`, which handles early closes |
| Alpaca crypto daily bar | `YYYY-MM-DDT00:00Z` bar start (verified) | `ts + 1 day` (00:00 UTC close, D-024) |
| yfinance XNYS-listed (SPY, QQQ, sector ETFs, `^VIX`, `^VIX3M`) | session date | `session_close + 15 min` (VIX settles 16:15 ET) |
| yfinance futures/FX (`NQ=F`, `GC=F`, `DX-Y.NYB`) | trade date | **(default)** 17:00 ET on the trade date (the CME/ICE daily break). So the 16:30 decision on day D sees D−1, which is correct |
| FRED/ALFRED vintage | `period` = observation date | `vintage` date at `release_time_et` for that series (config; BLS/BEA = 08:30 ET; **(default)** 17:00 ET for any series not listed) |
| `BAMLH0A0HYM2` archive | observation date | fetch time (D-001, Q4) |
| EDGAR fact | fiscal period | the filing's `acceptanceDateTime` from the submissions API, joined on accession. If it's missing, the `filed` date + 1 day at 00:00 ET |
| News article | — | **first-seen time** (when our fetch inserted it). `published_at` is stored separately (D-022) |
| Fed statement/minutes/speech (live) | event time | first-seen time |
| Fed item (backfill) | event time | the item's published time from the feed or page (official, reliable) |
| FOMC scheduled meeting dates | meeting date | **(default)** Dec 1 of the preceding year (the Fed publishes next year's schedule by mid-year, so this is conservative) |
| Congress action (live) | action date | first-seen time |
| Congress action (backfill) | action date | 23:59:59 ET on the action date |
| Earnings calendar (live) | report date | fetch time (each observation is a row, so date changes are preserved) |
| Earnings calendar (backfill) | report date | **(default)** report date − 7 days, flagged `approx=true`, with the bias documented |
| Funding rate (history) | settlement time | settlement time |
| OI / mark snapshot | fetch time | fetch time |
| CoinGecko daily volume/dominance (history) | day | `day + 1` at 00:00 UTC |
| DefiLlama stablecoin supply (history) | day | `day + 1` at 00:00 UTC |
| SoSoValue ETF flow | trade date | live: fetch time; history: trade date + 1 day at 09:00 ET (flows publish the next morning) |
| fja05680 membership | snapshot date | snapshot date at 00:00 ET (effective date; additions are announced before then, so this is conservative) |
| Our weekly membership snapshot | snapshot time | snapshot time |

Rules:
- `released_at` is computed by the collector from the rules above, **never** defaulted to `now()` silently. A row the collector can't time is rejected and logged.
- Backfills and live collects use the same function per source, so there's one code path (Q4).
- `ingested_at` is always the DB insert time, and it's kept separately for auditing.

## 4. Schema: migration `005_collectors.sql`
Applied migrations are immutable (D-013), so every change here is new.

### 4.1 New tables
| Table | Kind | Columns (key in bold) | Notes |
|---|---|---|---|
| `index_membership` | raw, append-only | **`index_code`, `snapshot_at`, `ticker`, `source`**, `asset_id` (nullable until mapped), `released_at`, `ingested_at` | One row per member per snapshot. `index_code='SPX'`. Snapshots only, never `valid_to` updates, so it stays append-only |
| `aliases` | reference (mutable) | **`alias_id`**, `asset_id`, `alias`, `kind` (`edgar_name`/`short`/`manual`/`ticker`), `case_sensitive`, `needs_cue`, `valid_from`, `valid_to`, `source` | Built from EDGAR + `configs/aliases.yaml` (Q7). Rebuilt weekly in one transaction |
| `raw_events` | raw, append-only | **`source`, `external_id`, `content_hash`**, `kind`, `event_ts`, `title`, `url`, `payload jsonb`, `released_at`, `ingested_at` | Fed items, FOMC dates, Congress actions, earnings dates. A changed item (e.g. a moved earnings date) is a new row with a new `content_hash` |
| `raw_metrics` | raw, append-only | **`source`, `metric`, `subject`, `ts`**, `value`, `unit`, `released_at`, `ingested_at` | Generic time series: CoinGecko volume (`subject`=coin), BTC dominance, stablecoin supply, ETF flows. `subject=''` for market-wide |
| `raw_crypto_derivs` | raw, append-only | **`venue`, `symbol`, `ts`, `kind`** (`funding`/`oi`), `funding_rate`, `funding_interval_h`, `oi_contracts`, `oi_usd`, `mark_price`, `released_at`, `ingested_at` | Raw per venue; aggregates are features (Q9) |
| `collector_state` | ops (mutable) | **`collector`**, `last_ok_at`, `last_attempt_at`, `last_error`, `cursor jsonb`, `rows_last_run` | Freshness and resume cursors. Feeds `runs.feed_freshness` and the daily summary |

### 4.2 Changes to existing tables
- `articles`: add `word_count int`, `extract_status text CHECK IN ('ok','short','failed','skipped')`, `guid text`, `title_hash text`, `near_dup_of bigint REFERENCES articles`. Add `COMMENT ON COLUMN articles.released_at` describing first-seen semantics (D-022). That's a comment, not a data change.
- `assets`: no change. yfinance-only series (`^VIX`, `NQ=F`, …) are rows with `asset_class` = `index`/`future`/`fx`.
- `raw_macro`: no change. `source` = `alfred` for normal vintages and `fred_archive` for the HY OAS snapshots.
- Append-only triggers (from 004) are added to `index_membership`, `raw_events`, `raw_metrics` and `raw_crypto_derivs`.

## 5. Package layout
```
src/qt/collect/
  http.py          shared httpx client: per-host rate limits, retry w/ backoff+jitter, timeouts, generic UA
  base.py          Collector protocol, run wrapper (run_context + collector_state + freshness), insert helpers
  pit.py           released_at rules per source (§3), pure functions
  alpaca_bars.py   alpaca_assets.py   yfin.py   fred.py   edgar.py
  membership.py    aliases.py         fed.py    congress.py   earnings.py
  crypto_structure.py (coingecko, defillama, sosovalue)   crypto_derivs.py (okx, hyperliquid, deribit, kraken)
  news/
    feeds.py       RSS fetch (feedparser), GUID/URL canonicalization
    extract.py     trafilatura full-text extraction, word count, raw HTML archive
    dedup.py       url/title sha256, 64-bit SimHash over 5-word shingles, Hamming ≤ 3 within 30 days
    tag.py         alias matching → tickers[]
  backfill.py      `python -m qt.collect.backfill --source <name> [--start YYYY-MM-DD]`, idempotent, resumable
src/qt/db/pit.py   as-of readers used by tests now and by Phase 3: macro_asof(series, T), latest_rows(table, T)
src/qt/orchestrator/
  scheduler.py     APScheduler 3.x service (`python -m qt.orchestrator.scheduler`)
  jobs.py          job functions (thin: call collectors under run_context)
scripts/install_scheduler.py   writes and loads the launchd agent on macOS (Windows path stubbed for Phase 12)
```

### 5.1 Collector contract
```python
class Collector(Protocol):
    name: str                      # matches configs/collect.yaml key and collector_state.collector
    critical: bool                 # from config; Phase 5 uses it for NO TRADE
    def collect(self, ctx: RunCtx, *, since: datetime | None, until: datetime) -> CollectResult: ...
```
- Every insert is `INSERT ... ON CONFLICT DO NOTHING`, so re-runs and overlapping backfills are safe.
- `run_collector(c)` wraps `run_context(stage=f"collect_{c.name}")`. It updates `collector_state`, logs `collected` with row counts, and on failure alerts (critical by Phase 1 rules) while leaving the other collectors running.
- Retries: 3 attempts with exponential backoff. A success after a retry logs a `warn`, and exhaustion raises.

## 6. Sources
### 6.1 Alpaca prices + assets
- **Universe for bars:**
  - every ticker ever in `index_membership` for SPX from 2016 on (so delisted names have history, avoiding survivorship bias)
  - plus ETFs used by the spec (SPY, QQQ, VNQ, sector ETFs; list in config)
  - plus the crypto pool
- **Stocks/ETFs:** daily bars, `adjustment=raw` stored as source `alpaca` (adjusted series are built in Phase 3 as their own feature version: raw tables hold raw data). **(default)** Also store `adjustment=all` as source `alpaca_adj` so Phase 3 doesn't need to rebuild splits/dividends itself; both are raw vendor outputs.
- **Feed:** `feed=sip` for daily history. That's free when older than 15 min, and the daily collect runs 30 min after close.
- **Crypto:** `v1beta3/crypto/us` daily bars for the pool from Alpaca's earliest available date (confirm in build). 
- **Assets API:** refresh `assets.alpaca_tradable` for equities at each stock collect and for crypto at each crypto collect. The crypto pool is USD-quoted, tradable, minus stablecoins (`stablecoin_exclude` list in config) (D-021).
- **Minute bars** are **not** collected in Phase 2 (they're needed in Phase 6/10 for the price transformer, and backfillable then).

### 6.2 yfinance
- **Spec tickers daily:** `SPY, NQ=F, GC=F, DX-Y.NYB, ^VIX, ^VIX3M`.
- **Pre-2016 backfill to 2000-01-01:** SPY, QQQ and the 11 SPDR sector ETFs, as source `yfinance`, for the HMM only (Q3).
- Never mixed with Alpaca rows: the source differs and Phase 3 picks sources explicitly.
- Fragile: cache raw responses to `data/cache/yfinance/`, throttle (config), and if it fails, `VIXCLS` from FRED covers VIX.

### 6.3 FRED / ALFRED
- **Series:** every FRED series listed in the spec's data table, plus `BAA10Y` and `VIXCLS` (list in `collect.yaml`).
- **Backfill:** `series/observations` with `realtime_start=1776-07-04&realtime_end=9999-12-31` (all vintages) → `raw_macro`.
- **Daily:** the same call with `realtime_start = last stored vintage`; new vintages append.
- **HY OAS:** each day, fetch the current series and store rows whose `(period, value)` differs from our latest stored value for that period, as `source='fred_archive'`, `vintage=fetch date (ET)`, `released_at=fetch time`.
- `qt.db.pit.macro_asof(series, T)` = per period, the row with max `vintage` where `released_at <= T`.

### 6.4 EDGAR
- **UA:** `SEC_EDGAR_USER_AGENT` from `.env`. The collector refuses to start if it's missing or has no `@`. Throttle ≤ `edgar.max_rps` (10).
- **Weekly:** download the `companyfacts.zip` bulk file (once a week, not per company), extract the needed us-gaap/dei concepts (list in config) for all mapped CIKs → `raw_fundamentals`.
- **Acceptance times:** from `submissions/CIK##########.json` (`filings.recent.accessionNumber` ↔ `acceptanceDateTime`), cached per CIK.
- **Ticker ↔ CIK:** `company_tickers.json` weekly. It also seeds `aliases`.

### 6.5 S&P 500 membership (Q5)
- **Load:** fja05680 `S&P 500 Historical Components & Changes` CSV, from 2016-01-01 → `index_membership` (`source='fja05680'`).
- **Weekly (Sunday):** snapshot of current constituents, **(default)** from the Wikipedia "List of S&P 500 companies" table (`source='wikipedia'`, `released_at` = snapshot time). From our first snapshot on, these are authoritative.
- **One-time cross-check at load:** compare fja05680's latest vs the first Wikipedia snapshot and write the differences to `reports/membership_crosscheck.md`. Not hand-patched.
- **Ticker → asset mapping:** via CIK when available, then via aliases with validity dates (FB → META).

### 6.6 Aliases (Q7)
- **Build:** EDGAR names → short names (strip suffixes from a config list) → merge `configs/aliases.yaml` (manual and crypto aliases, ambiguous flags).
- **Ambiguous names:** `needs_cue=true` aliases require a case-sensitive match **and** a finance cue (`finance_cues` list in config) in the same article.

### 6.7 News (Q6, D-022)
- **Every 15 min, 24/7:**
  1. Fetch the 5 RSS feeds.
  2. For each entry not already in `article_seen` (canonical URL sha256):
     1. Fetch the page.
     2. Archive the raw HTML to `data/news_archive/YYYY/MM/DD/<source>.jsonl.gz` (url, fetched_at, status, html).
     3. Extract with trafilatura.
     4. Compute `word_count`; `extract_status='short'` if < 150 words. The row is **still stored**; filtering is downstream.
     5. Exact dedup on canonical URL and normalized-title sha256; near dedup by SimHash (Hamming ≤ 3 vs the last 30 days). A near-duplicate is still stored, with `near_dup_of` set to the earlier article; the quota step in Phase 8 decides.
     6. Tag tickers.
     7. Insert into `articles` and `article_seen` in one transaction.
- **Polite fetching:** per-host throttle, honour `Retry-After`, generic UA.
- **Extraction failures:** stored with `extract_status='failed'` and `body=null`. No in-place retry (`articles` is append-only); the raw HTML archive allows re-extraction later into a new derived table if ever needed.
- **Fed RSS** (`press_monetary.xml`, `press_all.xml`, speeches feed) is polled in the same 15-min loop, so its first-seen time is accurate → `raw_events` (`kind` = `fomc_statement`/`minutes`/`speech`/`press`). Speech items are tagged with speaker and role (Chair/Governor/President) from a config map.

### 6.8 Congress.gov (Q8)
- **Daily** at **(default)** 06:00 ET: list bill actions updated since the cursor, keep those in the 8 committees (config), and store every action. `payload.is_event` is true only for the event statuses (config list); introductions are `is_event=false`.
- **Tagging:** `payload.tags` gets `debt_ceiling` / `cr_shutdown` from config keyword lists.
- Needs `CONGRESS_API_KEY` (api.data.gov).

### 6.9 Earnings calendar
- **Weekly + daily** refresh of the next 4 weeks from `api.nasdaq.com` (unofficial); Finnhub free is the fallback. Rows go to `raw_events` (`kind='earnings'`).
- **Backfill** of historical report dates for the event clock, flagged approximate (§3).

### 6.10 Crypto structure (Q2, Q9, Q10)
- **CoinGecko Demo:** daily `/global` (BTC dominance) and per-pool-coin 30-day volume (`/coins/{id}/market_chart`, 365 days of backfill). These feed the D-021 ADV ≥ $50M filter in Phase 3.
  - The coin-id map lives in config.
  - Budget: about 25 calls/day, well inside 10k/month. Attribution noted in the README.
- **DefiLlama:** total stablecoin supply, daily with full history.
- **Funding:** OKX (`/api/v5/public/funding-rate-history`, ~3 months of backfill) and Hyperliquid (`/info` `fundingHistory`), for BTC, ETH, SOL.
- **OI + mark:** daily snapshots from all 4 venues (OKX, Hyperliquid, Deribit, Kraken Futures).
- **SoSoValue:** verify the free tier in the build. If it isn't free, the collector is disabled in config and a DECISIONS entry records the drop (non-critical either way).

## 7. Scheduler (APScheduler 3.x, D-009)
- **Process:** a single long-running `python -m qt.orchestrator.scheduler`, run under the `paper` profile, timezone America/New_York.
- **Job store:** in memory. Every collector is idempotent and resumes from `collector_state`, so restarts catch up by themselves.
- **Job options:** `coalesce=True`, `max_instances=1`, `misfire_grace_time` from config. A missed or misfired job logs `ERROR` → critical alert.

| Job | When | Collectors |
|---|---|---|
| `news_loop` | every 15 min, 24/7 | news (5 feeds), Fed RSS |
| `stock_daily` | XNYS sessions only, at `decision_time(session)` (close + 30 min) | Alpaca equities/ETF bars, Alpaca assets, yfinance, FRED/ALFRED, HY OAS archive |
| `crypto_daily` | 00:15 UTC daily | Alpaca crypto bars, crypto assets, CoinGecko, DefiLlama, funding/OI, SoSoValue |
| `congress_daily` | 06:00 ET daily | Congress.gov |
| `earnings_daily` | 07:00 ET on sessions | earnings calendar (next 4 weeks) |
| `weekly` | Sunday 18:00 ET | membership snapshot, EDGAR tickers + companyfacts + submissions, aliases rebuild |
| `backup` | 21:00 ET daily | `scripts/db_backup.py` logic (Phase 1) |
| `daily_summary` | 21:30 ET daily | `write_daily_summary` with freshness + rows collected filled in |
| `heartbeat` | every 5 min | writes `data/run/heartbeat.json` |

- **`stock_daily` scheduling:** it's computed per day from the calendar (it re-plans itself each morning at 06:00 ET for that session), because APScheduler cron can't express "close + 30 on early-close days".
- **macOS:** `scripts/install_scheduler.py` writes `~/Library/LaunchAgents/com.qt.scheduler.plist` (RunAtLoad, KeepAlive, logs to `logs/`) and loads it. It's cross-platform Python, with the Windows branch raising `NotImplementedError` until Phase 12.
- **Watchdog:** launchd KeepAlive restarts a crash. The daily summary reports heartbeat gaps > 15 min.
- **Mac runtime (D-018):** this scheduler *is* the "lightweight collectors" process. No features, models or trading jobs are registered in Phase 2.

## 8. Freshness and criticality
`configs/collect.yaml` declares `critical` and `max_staleness` per collector. Phase 2 only **records and alerts**; Phase 5 turns stale critical feeds into NO TRADE.

| Critical (default) | Non-critical |
|---|---|
| alpaca_bars (stocks, crypto), alpaca_assets, fred core series, news, funding (all-venue) | yfinance (has FRED fallback for VIX), EDGAR, membership, Congress, earnings, CoinGecko, DefiLlama, OI venues individually, SoSoValue/ETF flows (Q10) |

A critical feed past `max_staleness` raises a critical alert once, coalesced by Phase 1 rules.

## 9. Config: `configs/collect.yaml` (new) and others
```yaml
# collect.yaml (excerpt; every number lives here)
universe:
  index: SPX
  candidates: 80                 # Q1: weekly screen output size, used in Phase 3
  membership_start: 2016-01-01
  extra_etfs: [SPY, QQQ, VNQ, XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]
crypto:
  quote: USD
  stablecoin_exclude: [USDC, USDT, DAI, PYUSD, USDG]
  adv_min_usd: 50_000_000        # D-021, measured on CoinGecko global volume
  adv_window_days: 30
  coingecko_ids: {BTC: bitcoin, ETH: ethereum, SOL: solana}   # extended for the full pool in build
  derivs_symbols: [BTC, ETH, SOL]
  derivs_venues: {funding: [okx, hyperliquid], oi: [okx, hyperliquid, deribit, kraken]}
schedule:
  news_every_min: 15
  stock_offset_min: 30           # = calendar decision offset
  crypto_utc: "00:15"
  congress_et: "06:00"
  earnings_et: "07:00"
  weekly: {day: sun, time_et: "18:00"}
  summary_et: "21:30"
  heartbeat_min: 5
  misfire_grace_s: 900
backfill:
  prices_start: 2016-01-01
  yfinance_index_start: 2000-01-01
  yfinance_index_tickers: [SPY, QQQ, XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]
fred:
  series: [DGS3MO, DGS2, DGS10, DFII10, T10YIE, DFF, BAMLH0A0HYM2, BAA10Y, VIXCLS,
           CPIAUCSL, CPILFESL, PCEPI, PCEPILFE, UNRATE, PAYEMS, GDPC1,
           WALCL, WTREGEN, RRPONTSYD, ECBASSETSW, JPNASSETS, DEXUSEU, DEXJPUS]
  release_time_et: {default: "17:00", CPIAUCSL: "08:30", CPILFESL: "08:30", UNRATE: "08:30",
                    PAYEMS: "08:30", PCEPI: "08:30", PCEPILFE: "08:30", GDPC1: "08:30"}
  archive_series: [BAMLH0A0HYM2]
news:
  min_words: 150
  simhash: {bits: 64, shingle: 5, max_hamming: 3, window_days: 30}
  feeds: {cointelegraph: ..., decrypt: ..., cnbc_markets: ..., bbc_business: ..., npr_economy: ...}
congress:
  committees: [hswm00, hsba00, hsap00, hsbu00, ssfi00, ssbk00, ssap00, ssbu00]   # verify codes in build
  event_statuses: [reported, passed_house, passed_senate, resolving_differences, to_president, became_law, vetoed]
edgar: {max_rps: 10}
http: {timeout_s: 20, retries: 3, backoff_base_s: 2, per_host_rps: {default: 2, "data.alpaca.markets": 3}}
freshness:                       # max staleness per collector; critical flags
  alpaca_bars_equity: {critical: true, max_hours: 30}
  news: {critical: true, max_hours: 2}
  # … one entry per collector
```
Plus `configs/aliases.yaml` (manual and crypto aliases, ambiguous flags, `finance_cues`, `strip_suffixes`).

## 10. Dependencies
- **Runtime:** `alpaca-py`, `apscheduler>=3.10,<4`, `feedparser`, `trafilatura`, `yfinance`, `lxml` (for the Wikipedia table).
- **Dev:** none new. HTTP mocking uses `httpx.MockTransport` and recorded fixtures in `tests/fixtures/`.

## 11. Keys Dom must provide (`.env`)
`ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY` (paper, `PK…`), `FRED_API_KEY`, `CONGRESS_API_KEY`, `COINGECKO_DEMO_API_KEY`, `SOSOVALUE_API_KEY` (optional), `FINNHUB_API_KEY` (optional). `SEC_EDGAR_USER_AGENT` is already set. A collector whose required key is missing is skipped with a warning at startup, and it is never silently "ok".

## 12. Tests (`uv run pytest`)
Network is never hit in unit tests: recorded fixtures plus `httpx.MockTransport`. One opt-in live smoke test per source (`-m live`, not run by default).

| Test | Asserts |
|---|---|
| `test_pit_rules` | Each §3 rule: stock bar `released_at` = session close (incl. 2026-11-27 13:00 ET early close); crypto = ts + 1 day; FRED vintage + release time; futures 17:00 ET; EDGAR acceptance join + fallback |
| `test_no_lookahead_macro` | Synthetic CPI with 3 vintages: `macro_asof(T)` returns the vintage known at T, never a later revision, for T just before and after each release |
| `test_no_lookahead_bars` | `latest_rows(raw_prices, T)` at 16:29 ET excludes that day's bar; at 16:30 ET (after close) includes it |
| `test_no_lookahead_news` | An article with a backdated `published_at` is only visible from its first-seen `released_at` |
| `test_idempotent_collect` | Running each collector twice on the same fixture inserts nothing the second time |
| `test_append_only_new_tables` | UPDATE/DELETE on `index_membership`, `raw_events`, `raw_metrics`, `raw_crypto_derivs` raise |
| `test_hy_archive` | Unchanged values aren't re-stored; a changed value appends a new `fred_archive` row |
| `test_crypto_pool` | Stablecoins excluded; non-USD quotes excluded; pool follows the Assets API fixture |
| `test_derivs_normalize` | OKX 8h and Hyperliquid 1h rates annualize correctly; contracts → USD at mark |
| `test_news_dedup` | Canonical URL strips query/fragment; identical titles collide; SimHash Hamming ≤ 3 is flagged as near-dup; short bodies stored with `extract_status='short'` |
| `test_alias_tagging` | "Google" → GOOGL/GOOG; "Target" without a cue → no tag, with "Target shares" → TGT; a 2019 "Facebook" maps to the ticker valid then |
| `test_membership_mapping` | Ticker reuse resolved by CIK; FB → META across the rename |
| `test_edgar_ua_guard` | Missing or `@`-less UA → collector refuses to start |
| `test_scheduler_plan` | Jobs registered with the right triggers; `stock_daily` lands at 16:30 ET on a normal day, 13:30 on 2026-11-27, none on 2026-11-26 (Thanksgiving) or weekends |
| `test_collector_failure_isolated` | One collector raising doesn't stop the others in the same job; a critical alert fires (mocked) |
| `test_missing_key_skips` | A missing key → skipped with a warning, `collector_state` shows it, no fake "ok" |

## 13. Acceptance criteria (exit gate)
1. Migration `005` applies on the real DB and on an empty test DB, and re-running is a no-op.
2. **Backfill complete:**
   - Alpaca daily bars from 2016-01-01 for every SPX member since 2016 + the ETFs + the crypto pool (crypto from Alpaca's earliest date)
   - yfinance index-level series from 2000
   - all FRED/ALFRED vintages for every configured series
   - fja05680 membership loaded and cross-checked
   - EDGAR facts for all mapped CIKs
   - funding/CoinGecko/DefiLlama history as far as each API allows

   A backfill report (`reports/backfill_YYYYMMDD.md`) lists first/last date and row count per source.
3. Every row in every raw table has non-null `released_at` (it's a NOT NULL constraint), and the §3 rules are covered by `test_pit_rules`.
4. `uv run pytest` is green and ruff is clean.
5. **5 consecutive days unattended on the Mac (D-018):** the scheduler runs under launchd and every scheduled job succeeds (or fails *and alerts*, and is then fixed). News has no gap > 1 h between successful loops, and the daily summary shows freshness for every collector each day.
6. `.env.example` lists every key; no secret is in git.
