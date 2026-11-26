# Component 03: Feature layer + regime models

> Build spec for Phase 3. Written from Dom's answers in [`docs/interrogation/03_features.md`](../interrogation/03_features.md) (S04, 2026-11-26).
> Where this file and `docs/spec/system_spec_v2.md` disagree, this file wins. Decision: D-033.
> Items marked **(default)** were not asked in the questioning. Claude picked them; **Dom accepted all of them** (S04, 2026-11-26).

## 1. Purpose
Turn raw, point-in-time rows into the two daily matrices every model reads, plus the regime signals:
- **Market vector:** macro, cross-asset, breadth, crypto-structure and regime features (one row per decision time).
- **Asset matrix:** per-asset price features, cross-sectional ranks and five composite factors (one row per asset per decision time).
- **Regime:** growth/inflation quadrant, 3-state HMM (filtered probabilities only), stress score.

Models never see raw price levels. Every feature for decision time T reads only rows with `released_at <= T`.

Out of scope: labels (Phase 4), the weekly screen and tier eligibility (Phase 5, **(default)**: Phase 3 supplies the ranks the screen uses), models, trading.

## 2. Scope
| In Phase 3 | Deferred |
|---|---|
| Collector top-up for the Q1 additions (§4) | Weekly screen, top-80 candidates, `universe_weekly.tier_eligible` → Phase 5 |
| Split/dividend adjustment of raw bars from `raw_events alpaca_ca` (D-030), in memory, never written back to raw | Minimum-history trading rule → Phase 5 (Q8) |
| Per-asset price features (§5.1) | Feature-importance pruning → Phases 4/7 |
| Cross-sectional ranks + 5 composites (§5.2) | Text and LLM features → Phases 8/9 |
| Macro and cross-asset features, GLI, breadth (§5.3) | |
| Quadrant, HMM, stress score (§6) | |
| Normalization, staleness features, masks (§7) | |
| Feature registry + versioning (§8) | |
| Sector fill for delisted names (Q3) and survivorship gap closure (Q9) (§9) | |
| Daily `features` job in the scheduler at 18:00 ET, monthly HMM refit | |

## 3. Point-in-time contract
- **Decision time.** `asof` = the 18:00 ET feature run on each XNYS session (spec flow), stored in UTC. **(default)** Crypto features are computed at the same `asof` 7 days a week. On non-sessions only crypto and market features that changed are written.
- **Inputs.** Only `db.pit.latest_rows` / `macro_asof` (Phase 2). No feature function opens a raw table directly. A lint test (§11) enforces this.
- **Macro.** The ALFRED vintage known at T. Forward-fill until the next release, never back-fill. Each macro series gets a `days_since_release` feature.
- **Fundamentals.** The newest fact whose `released_at <= T` (EDGAR acceptance time). TTM values are built from the quarterly facts known at T only.
- **Corporate actions.** The adjustment factor at T uses only actions with `released_at <= T`. A split announced for next week does not adjust today's history.
- **Earnings proximity.** Uses the newest calendar observation with `released_at <= T`. Backfilled rows carry `approx=true` (D-031), passed through as a mask feature `earnings_date_approx`.

## 4. Collector top-up (Q1 additions)
Added to existing collectors through `configs/collect.yaml` only, with no new collector classes. Each new series is backfilled and covered by the Phase 2 PIT rules already in force.

| Collector | Add |
|---|---|
| `fred` | `DEXUSAL`, `T5YIFR`, `THREEFYTP10`, `NFCI`, `ICSA`, Japan 3-month rate `IR3TIB01JPM156N` (OECD, monthly; verify the ID in the build). `T10YIE` is already collected |
| `yfinance` | `HG=F`, `CL=F` (futures_fx rule), `^MOVE`, `^SKEW`, `^VVIX` (xnys rule, settle +15 min) |
| `alpaca` bars (`universe.extra_etfs`) | `TLT` (**missing today, and the spec's stock-bond correlation needs it**), `IEF`, `HYG`, `IWM`, `RSP`, `SOXX`, `KRE`, `EEM` |
| `coingecko` | none: ETH/BTC is computed from bars we already have |

`^MOVE` and `^SKEW` history on yfinance may be short or gappy. The backfill report records first/last date, and the features carry masks either way.

## 5. Feature catalogue
Annualization: √365 for crypto, √252 for everything else. All windows are in config (`configs/features.yaml`).

### 5.1 Per-asset price features (stocks, ETFs, coins)
As in the spec table, unchanged:
- log return at 1, 5, 21, 63, 126 and 252 days; momentum 12-1; realized vol at 21 and 63 days; Garman-Klass vol at 21 days; downside deviation at 63 days; max drawdown at 63 and 252 days
- trend distance (P/SMA50, P/SMA200, SMA50/SMA200); RSI(14); MACD histogram / P; Bollinger %B(20); ATR(14)%
- volume z-score at 21 days; Amihud illiquidity at 21 days; beta and residual vol at 63 and 252 days (OLS on SPY, coins on BTC); 63-day correlations to SPY, DXY returns and 10y yield changes
- mean-reversion z at 63 days; OU half-life at 126 days; **Hurst by variance ratio at 126 days only (Q4)**; earnings proximity (trading days to next report, last surprise %)

### 5.2 Cross-sectional factors (stocks)
- **Rank universe (default):** the point-in-time S&P 500 membership at T (`index_membership`), about 500 names. It's the pool the Phase 5 screen picks from. Each stock feature becomes a percentile rank (0–1) across that universe and again within its sector.
- **Composites,** each the mean of the available component ranks:
  - Value: FCF yield, earnings yield, EBITDA/EV
  - Quality: ROIC, gross-margin stability (inverse std over 8 quarters), inverse net debt/EBITDA
  - Momentum: 12-1, 63-day return
  - Low volatility: inverse 252-day vol, inverse beta
  - Growth: revenue YoY, EPS YoY
- **(default)** A composite needs ≥ 2 of its components for a value. Otherwise it's NaN with a mask.
- **(default)** An EDGAR concept map in `configs/features.yaml` lists the XBRL tags per input (e.g. revenue = `Revenues` → `RevenueFromContractWithCustomerExcludingAssessedTax` → `SalesRevenueNet`, first present wins). Banks and REITs get NaN where a concept is undefined (EBITDA/EV, net debt/EBITDA), not a fudged value.

### 5.3 Market vector
**Spec list, unchanged except where noted:**
- curve slope (10y−2y, 10y−3m, each with a 21-day change); real yield DFII10 (level, 21-day change); policy path (2y − fed funds); inflation momentum (core CPI and core PCE, 3-month annualized − YoY); Sahm; real GDP QoQ annualized
- **credit stress on BAA10Y only (Q2):** level, 252-day z-score, 21-day change. The ICE series is archived but not a feature
- VIX/VIX3M; VIX 252-day z-score; dollar (DXY 21-day return, 252-day z-score); 63-day SPY/TLT correlation; gold residual (gold return − 252-day OLS fit on the real-yield change, z-scored); VNQ/SPY 63-day return
- event clock: days to next FOMC, next CPI, and the next earnings-heavy week. **(default)** An earnings-heavy week has ≥ 20% of members reporting
- crypto structure: BTC dominance and its 21-day change; stablecoin supply 30-day % change; 7-day average funding (annualized, venue average); OI / market cap; 5-day ETF net flows (NaN + mask until SoSoValue has a key)
- **GLI**, weekly, in USD billions: `(WALCL − WTREGEN − RRPONTSYD) + ECBASSETSW × DEXUSEU + JPNASSETS / DEXJPUS`, with unit conversion per series in config. Features: level, 13-week % change, 252-day z-score

**Added in Q1:**
| Group | Features |
|---|---|
| Framework four | copper/gold ratio (21-day change, 252-day z); AUD/USD 21-day return; USD/JPY carry = US 3-month − Japan 3-month, plus the USD/JPY 21-day return; MOVE level, 252-day z, MOVE/VIX |
| Rates and inflation | 10y breakeven (level, 21-day change); 5y5y forward; ACM 10y term premium (level, 21-day change); NFCI level and 4-week change; initial claims 4-week average YoY % |
| Commodities and volatility | WTI 21-day return and 252-day z; SKEW z; VVIX z; VVIX/VIX |
| Risk-appetite ratios | 63-day log return of IWM/SPY, RSP/SPY, XLY/XLP, SOXX/SPY, KRE/SPY, EEM/SPY, HYG/IEF |
| Breadth (own universe) | % of PIT members above their SMA200; 52-week new highs − new lows (% of members); 21-day advance/decline line change |
| Crypto | ETH/BTC ratio 21-day change and 252-day z |

## 6. Regime models (`features/regime/`)
### 6.1 Quadrant (Q6)
- **Growth up** = Sahm ≤ its 3-month average + `sahm_flat_tol` (0.05; flat counts as falling) **and** real GDP QoQ annualized > its 4-quarter average.
- **Inflation up** = inflation momentum (core CPI) > 0. **(default)** Core PCE momentum is a separate feature, not part of the rule.
- Goldilocks (up, down), Reflation (up, up), Stagflation (down, up), Deflation (down, down). Stored as `regime.quadrant` and as 4 one-hot features.
- All inputs come from ALFRED vintages at T.

### 6.2 HMM (Q5)
- `hmmlearn` GaussianHMM, 3 states, full covariance, weekly samples (the last session of each week).
- Inputs: SPY 21-day return, SPY 21-day realized vol, BAA10Y 21-day change, VIX/VIX3M. **(default)** Inputs are standardized with mean and std from the training window only.
- 20 restarts with seeds 0–19. The best log-likelihood wins.
- After each fit, states are sorted by mean SPY vol into calm < normal < stress. The rule is locked.
- Refit monthly (first Sunday) on an expanding window, ≥ 260 weekly points before the first fit. Models are saved to `models/hmm/<fit_date>.joblib` with seed, log-likelihood and the training window.
- **Filtered probabilities only:** the forward pass up to T, never `predict_proba` on the full sequence (that's smoothed and leaks). Daily values between weekly samples are forward-filled from the latest weekly filter step. **(default)**
- **Stability alert:** if a refit changes > 20% of the prior 52 weeks' argmax states, send an ntfy alert (not critical).
- **Backtest history:** for each historical month the model is the one fitted on data up to that month. Every past probability is produced by a model that could have existed then.

### 6.3 Stress score
Mean of z(VIX), z(BAA10Y), z(−curve slope 10y−3m) and the VIX/VIX3M > 1 flag. Each z is a 252-day past-only z. Stored in `regime.stress_score` and as a feature.

## 7. Normalization, staleness and missing data
- **Time-series features:** rolling 252-day z-score from past data only, winsorized at ±4. Both the raw transform (e.g. a return) and its z are stored, as separate features.
- **Stock features:** daily cross-sectional ranks (§5.2). NaN is left out of the rank and stays NaN (Q8).
- **New listings (Q8):** in the data from day one. Each feature is computed once its own window is full and is NaN before that. Masks: `history_days` and one validity flag per window group (21, 63, 126, 252).
- **Macro:** forward-fill until the next release only; `days_since_release` per series.
- **Model-side handling** (documented here, applied in Phase 7): trees take NaN; neural nets use a neutral fill plus the mask.

## 8. Feature registry and versioning (Q7)
- `feature_registry (feature_name, version, formula, window, source_series, created_at)` already exists. Migration `008` adds `changelog text`, `status text` (`active` | `frozen`) and `group text`.
- Each feature is a Python function registered with a decorator: `@feature(name, version, window, sources, group)`. On startup the registry is synced, and a code version not in the table is inserted. An existing `(name, version)` whose formula text changed fails the run: bump the version.
- Bump on any change to formula, window, input or normalization. Refactors with identical output don't bump; the hand-calculation test proves it.
- **`features` becomes append-only (migration 008):** UPDATE and DELETE are blocked like the raw tables, and inserts use `ON CONFLICT DO NOTHING`.
- Trained models pin their `(feature, version)` set (a Phase 7 table; Phase 3 provides `pinned_versions()`, reading it once it exists).
- A non-current version is computed daily while any active model pins it, then marked `frozen` with its history kept. A new version is backfilled over the full history at once.
- `model_input_latest` is replaced by `features_asof(T, versions)`, which returns the wide market vector and asset matrix (polars or pandas) for a given T and pinned set.

## 9. Sectors and survivorship gaps
### 9.1 Sectors (Q3)
- Current members keep the Wikipedia GICS sector (`assets.gics_sector`).
- Delisted members: sector from the EDGAR submissions `sic` code via `configs/sic_to_gics.yaml` (SIC range → one of 11 GICS sectors).
- Migration 008 adds `assets.sector_source` (`wikipedia_gics` | `sic_map`). Labels are static, and the 2018/2023 reshuffle lookahead is accepted.

### 9.2 Fundamentals for delisted members (Q9a)
- Script `scripts/map_delisted_ciks.py` downloads `cik-lookup-data.txt` with the configured SEC user agent (cached under `data/cache/sec/`) and normalizes names (case, punctuation, suffixes such as Inc/Corp/Co/Ltd/Holdings/The).
- Exact match → accepted. Fuzzy match (rapidfuzz ≥ 92, **(default)**) → accepted only if the CIK's submissions show 10-K/10-Q filings across the asset's membership span. Anything else → `reports/cik_review_YYYYMMDD.md` for Dom.
- Accepted mappings are written to `configs/cik_overrides.yaml` (ticker, security_key, CIK, match type), loaded into `assets.cik`, and then `edgar_facts` backfills them.
- **Target:** ≥ 80% of the 135 names mapped **(default)**. The rest are listed as a known gap.

### 9.3 Bars for delisted names (Q9b; list widened by D-035)
- SNDK (old, to 2016-05), TE (to 2016-06), STI (to 2019-12), plus three found in S04: Paramount Global (`SPX:PARA:2022-02-17`, 2022-02-17 → 2025-08-06; its Alpaca bars were another company's and are excluded), WYND and AABA (no bars inside their membership spells). Try yfinance, then Stooq.
- Accept only an exact ticker whose last bar matches the known delisting date within 5 sessions **(default)**. Never splice in a successor company's prices.
- Stored as `raw_prices` source `yfinance` / `stooq` under the existing asset. Misses are documented.

## 10. Package layout
```
src/qt/features/
  registry.py        @feature decorator, registry sync, versions, pinned_versions()
  inputs.py          PIT loaders: adjusted bars, macro as-of, fundamentals TTM, events (all via db.pit)
  adjust.py          split/dividend adjustment factors from alpaca_ca as known at T
  price.py           §5.1
  fundamentals.py    point-in-time TTM/instants and the §5.2 ratios (added in the build, D-037)
  cross_section.py   ranks, sector ranks, composites, EDGAR concept map
  macro.py           §5.3 incl. GLI, framework additions, breadth
  crypto.py          crypto structure
  normalize.py       rolling z, winsorize, masks, days_since_release
  matrix.py          features_asof(T, versions) → market vector, asset matrix
  build.py           daily job + backfill CLI: python -m qt.features.build --since ... --until ...
  regime/quadrant.py, regime/hmm.py, regime/stress.py
```
Scheduler: job `features_daily` at 18:00 ET every day (equity parts only on sessions), job `hmm_refit` on the first Sunday of each month. Both are registered in `orchestrator/jobs.py` and gated so they won't run if a critical feed is stale. Instead the run is marked `stale_inputs` and alerts. NO TRADE itself is Phase 5.

## 11. Tests
| Test | Asserts |
|---|---|
| `test_feature_hand_calcs` | **Every** registered feature against a hand calculation on a tiny fixture (parametrized over the registry; a feature with no hand case fails the suite) |
| `test_no_lookahead_features` | For each feature, mutating any input row with `released_at > T` leaves the value at T unchanged (parametrized over the registry) |
| `test_inputs_only_via_pit` | AST lint: no module in `features/` references a raw table name or a raw SQL string |
| `test_adjustment_pit` | A split with `released_at` after T doesn't adjust history at T; after release it does |
| `test_macro_ffill` | Forward-fill stops at the next release; no back-fill; `days_since_release` counts correctly across a revision |
| `test_quadrant` | Sahm flat within 0.05 → growth leg passes; Sahm above avg + 0.05 → fails; each quadrant reached in a synthetic series |
| `test_hmm_filtered_only` | Appending future observations doesn't change past filtered probabilities; relabel order = vol order; same seed → identical fit |
| `test_hmm_stability_alert` | A forced relabel flip fires the alert (mocked ntfy) |
| `test_gli_units` | Unit conversion per series against a hand total |
| `test_ranks_nan` | NaN is excluded from ranks; ties get the average rank; sector ranks only within sector |
| `test_versioning` | Changed formula without a bump → error; a new version writes alongside the old; UPDATE/DELETE on `features` raise |
| `test_new_listing_masks` | 30-day-old asset: 21-day features set, 63/252 NaN, masks correct |
| `test_cik_matching` | Normalization cases; a fuzzy match outside the filing span is rejected |
| `test_scheduler_plan` | `features_daily` at 18:00 ET; `hmm_refit` on the first Sunday |

## 12. Acceptance criteria (exit gate)
1. Migration 008 applies on the real DB and an empty test DB, and re-running is a no-op.
2. The §4 top-up series are backfilled and appear in the backfill report.
3. The full feature history is built from 2016-01-01 (macro-only features as far back as their inputs allow), and the registry has a row for every feature.
4. Every feature passes its hand-calculation test and its no-lookahead test (MASTER_PLAN gate). `uv run pytest` is green and ruff is clean.
5. HMM: monthly expanding-window fits across history, filtered probabilities stored in `regime`, and a stability report (`reports/hmm_stability_YYYYMMDD.md`).
6. Survivorship report: CIKs mapped / reviewed / unmapped out of the 135 names, and the 3 bar-less names resolved or documented.
7. `features_daily` runs unattended for 3 consecutive sessions **(default)** and the daily summary shows feature counts and NaN rates per group.

## 13. Config: `configs/features.yaml` (new; every number lives here)
Windows per feature; annualization; z window (252) and winsor (4); composite minimum components (2); EDGAR concept map; GLI unit factors; quadrant `sahm_flat_tol` (0.05), GDP average quarters (4); HMM (states 3, covariance full, seeds 0–19, min_weeks 260, refit monthly, stability window 52 and threshold 0.20); stress components; earnings-heavy week threshold (0.20); CIK fuzzy threshold (92); bar date tolerance (5 sessions).

## 14. Dependencies
`rapidfuzz` (name matching). `hmmlearn` (pinned, D-009) and `scikit-learn` should already be in the lock; check before adding. Nothing paid.
