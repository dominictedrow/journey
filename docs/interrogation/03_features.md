# Questioning: Phase 3 · Feature layer + regime models

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/03_features.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Quant feature layer, Regime models, Normalization rules

1. Confirm the feature list as written, or tell me what to add or remove. Should we also compute the extra macro composites from your framework (copper/gold, AUD/USD, USD/JPY carry, MOVE)?
   **A (2026-11-26):** "I want more data, not less." Keep the whole spec list, remove nothing, add all four composites, and widen the macro and cross-asset set further. Proposed additions (all free; Dom can veto any):
   - Framework four: copper/gold (`HG=F` / `GC=F`), AUD/USD (`DEXUSAL`), USD/JPY carry (`DEXJPUS` plus the Japan short rate vs the US 3-month), MOVE (`^MOVE`).
   - Rates and inflation (FRED): 10y breakeven `T10YIE`, 5y5y forward `T5YIFR`, ACM 10y term premium `THREEFYTP10`, Chicago Fed NFCI, initial claims `ICSA` (4-week average, YoY).
   - Commodities and volatility: WTI `CL=F`, `^SKEW`, VVIX `^VVIX`.
   - Risk appetite ratios (yfinance ETFs): IWM/SPY, RSP/SPY (breadth), XLY/XLP (cyclical vs defensive), SOXX/SPY, KRE/SPY (banks), EEM/SPY, HYG/IEF.
   - Breadth computed from our own universe: % of members above SMA200, 52-week new highs minus new lows, advance/decline 21-day.
   - Crypto: ETH/BTC ratio and 21-day change.
   - These need collector additions (tickers and series in `configs/collect.yaml`), so they land as a small Phase 2 top-up before the Phase 3 feature build.
2. **HY OAS (D-001):** use BAA10Y for history and ICE once ≥ 3 years overlap? Or BAA10Y always, for consistency?
   **A (2026-11-26):** B. Use BAA10Y always, for every credit-stress feature, the HMM input and the stress score. Never switch or splice. The ICE series keeps being archived daily (the raw table is append-only) but isn't a feature in v1. Closes D-001.
3. **GICS sectors:** where do they come from (SIC codes from EDGAR mapped to GICS-like sectors, or yfinance `sector`)? Are approximate sectors acceptable?
   **A (2026-11-26):** A. Current members keep the real GICS sector from the Wikipedia constituents table (already in `assets.gics_sector`, 503 of 731 equities). Delisted members get a sector from their EDGAR SIC code through a SIC→GICS mapping table in config. Labels are static. The mild lookahead from the 2018 and 2023 GICS reshuffles is accepted and documented, since it only changes the peer group used for within-sector ranks. Each asset records where its sector came from (`wikipedia_gics` or `sic_map`).
4. **Hurst:** the variance-ratio method is in the spec. Do you also want R/S or DFA estimates for comparison?
   **A (2026-11-26):** A. Variance ratio only, 126-day window, as in the spec. No R/S or DFA.
5. **HMM:** weekly sampling, as in the spec. How many restarts and which seeds? Should state relabelling (sorted by vol) be locked across refits?
   **A (2026-11-26):** Defaults accepted (all in `configs/features.yaml`):
   - Weekly sampling, 3 states, full covariance. Inputs: SPY 21-day return, SPY 21-day realized vol, BAA10Y 21-day change (per Q2), VIX/VIX3M.
   - 20 restarts per refit with seeds 0–19. The best log-likelihood wins.
   - Relabel after every refit by sorting on the mean SPY vol per state: calm < normal < stress. The rule is locked.
   - Stability alert: if a refit changes more than 20% of the prior 52 weeks' filtered state assignments, send an ntfy alert.
   - Refit monthly on an expanding window, with at least 260 weekly points before the first fit. Filtered probabilities only.
6. **Quadrant:** "Sahm falling" means a falling month-over-month change, or below its 3-month average?
   **A (2026-11-26):** B with the flat rule. "Sahm not rising" means current Sahm ≤ its 3-month average + tolerance, with tolerance 0.05 (config `features.quadrant.sahm_flat_tol`). So flat counts as falling. Growth up = Sahm not rising AND real GDP QoQ annualized > its 4-quarter average. Computed from ALFRED vintages known at T.
7. **Feature versioning:** bump the version on any formula change and keep the old versions computed?
   **A (2026-11-26):** A.
   - Bump `formula_version` (an integer) on any change to formula, window, input series or normalization. Refactors with identical output, proven by the hand-calculation test, don't bump. Each bump adds a registry row with a changelog note.
   - Feature rows are never overwritten: a new version is written alongside the old one.
   - Trained models pin their exact `(feature, version)` set.
   - An old version keeps computing daily while any active model pins it. After that it's frozen and its history kept.
   - A new version is backfilled over the full history at once.
8. **Missing data policy for new listings** (fewer than 252 days of history): leave NaN plus a mask feature, or exclude the asset until it has enough history?
   **A (2026-11-26):** A. The asset is in the data from day one. Each feature is computed once its own window is full and is NaN before that. Mask features: `history_days` plus one validity flag per window group. NaN is left out of cross-sectional ranks and stays NaN. Dom chose A, not C, so Phase 3 sets no minimum-history trading threshold. Whether to block trading on short-history names is left to the Phase 5 rules engine (tier eligibility).
9. **Survivorship gaps (Dom asked to close these in Phase 3, 2026-11-26):** (a) delisted S&P members have no EDGAR facts, because SEC's `company_tickers.json` lists only current filers. Lead, unverified: `companyfacts` works for any CIK, so map old tickers to CIKs through SEC's full `cik-lookup-data.txt` name list (or EDGAR company search) using the security names from membership, then backfill through `edgar_facts`. (b) 3 delisted members have no Alpaca bars (old SNDK to 2016-05, TE to 2016-06, STI to 2019-12). Leads, unverified: Stooq, or a yfinance history under the successor ticker (STI → TFC?). Which gaps are worth closing, and is an approximate name match acceptable for (a)?
   **A (2026-11-26):** A. Close both. Measured gap: 135 of 731 equities have no CIK and no facts, all delisted. `cik-lookup-data.txt` confirmed reachable (HTTP 200).
   - (a) Normalize names (case, punctuation, corporate suffixes) and accept exact matches automatically. Accept a fuzzy match only if that CIK's filing history covers the asset's index-membership span. Everything else goes on a review list for Dom. Confirmed mappings are stored in a config table (`configs/cik_overrides.yaml`) and backfilled through `edgar_facts`.
   - (b) Try yfinance, then Stooq, for SNDK (old), TE and STI. Accept only an exact ticker with a matching date range, and never splice in a successor company's prices. Anything still missing is a documented gap.
