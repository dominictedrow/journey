# Questioning: Phase 3 · Feature layer + regime models

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/03_features.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Quant feature layer, Regime models, Normalization rules

1. Confirm the feature list as written, or tell me what to add or remove. Should we also compute the extra macro composites from your framework (copper/gold, AUD/USD, USD/JPY carry, MOVE)?
2. **HY OAS (D-001):** use BAA10Y for history and ICE once ≥ 3 years overlap? Or BAA10Y always, for consistency?
3. **GICS sectors:** where do they come from (SIC codes from EDGAR mapped to GICS-like sectors, or yfinance `sector`)? Are approximate sectors acceptable?
4. **Hurst:** the variance-ratio method is in the spec. Do you also want R/S or DFA estimates for comparison?
5. **HMM:** weekly sampling, as in the spec. How many restarts and which seeds? Should state relabelling (sorted by vol) be locked across refits?
6. **Quadrant:** "Sahm falling" means a falling month-over-month change, or below its 3-month average?
7. **Feature versioning:** bump the version on any formula change and keep the old versions computed?
8. **Missing data policy for new listings** (fewer than 252 days of history): leave NaN plus a mask feature, or exclude the asset until it has enough history?
