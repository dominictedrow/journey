# Handoff: S05 · 2026-11-26 · Phase 3 build step 5a (regime + normalization)

**Phase:** 3 (Feature layer + regime models), step (b) build
**Session goal:** Start build step 5: regime (§6), normalization + masks (§7), then `matrix.py`/`build.py` and §9 survivorship.
**Outcome:** 🟡 partial. Step 5a (regime + normalization) is done and committed. 5b (matrix/build/jobs) and 5c (survivorship) are next.

## What was done
- Step 5 split into three parts: 5a regime + normalize, 5b matrix/build/scheduler jobs, 5c survivorship.
- **`features/regime/quadrant.py`** (§6.1): Sahm not rising (≤ its 3-month mean + 0.05) AND GDP QoQ annualized > its 4-quarter mean; inflation up = core CPI momentum > 0. Registers 4 one-hots `quad_*`.
- **`features/regime/stress.py`** (§6.3): `stress_score` = mean of z252(VIX), z252(BAA10Y), z252(−(10y−3m)) and the 1[VIX/VIX3M > 1] flag.
- **`features/regime/hmm.py`** (§6.2):
  - Weekly observations at each week's last session (18:00 ET): SPY 21d return, SPY 21d realized vol, BAA10Y 21d change, VIX/VIX3M.
  - Fits: 20 seeds, best log-likelihood, standardized on the training window, states sorted by vol (stored already in that order).
  - Forward filter only. Plus `relabel_change`, joblib save/load and `HmmInputs.fit_at`.
  - Features `hmm_p_{calm,normal,stress}`; `regime.regime_row()` builds the `regime` table row.
- **`features/normalize.py`** (§7):
  - `rolling_z` / `_tsz252` features for market groups and for ETF/crypto per-asset features.
  - `asset_ts_z` ready times.
  - Masks `history_days`, `hist_ok_{21,63,126,252}`.
  - Staleness `dsr_<series>` through the new `MacroState.newest_period_release`.
- `market.sahm` refactored onto `sahm_values`, with output unchanged; added `gdp_qoq_values`.
- Config: `features.regime`, `features.normalize` (+ pydantic classes).
- Deps: hmmlearn 0.3.3 (pinned), scikit-learn, rapidfuzz.
- **Live smoke:**
  - Fits on real data for Feb 2014, Jan 2016, Jan 2020 and Nov 2026. 370 to 1,035 weeks, about 1.5 s each, with similar state means across fits.
  - The HMM reads stress on 2020-03-20 (p = 1.0) and 2022-06-17 (0.92), calm on 2024-07-12, and normal on 2026-11-25.
  - About 0.1 s per T for the probabilities.

## Files touched
| File | Change |
|---|---|
| `src/qt/features/regime/{__init__,quadrant,stress,hmm}.py` | New: regime models + registration |
| `src/qt/features/normalize.py` | New: z, masks, staleness |
| `src/qt/features/market.py` | `regime` field on MarketState; `sahm_values`, `gdp_qoq_values` |
| `src/qt/features/inputs.py` | `MacroState.newest_period_release` |
| `src/qt/common/config.py`, `configs/features.yaml` | Regime + normalize config |
| `tests/test_regime.py` (33), `tests/test_normalize.py` (8) | New |
| `pyproject.toml`, `uv.lock` | Deps |
| `DECISIONS.md` | D-040 |

## Tests
- `uv run pytest -q`: 691 passed (before lint-only edits; the new and market test files were rerun after them: 158 passed). `ruff check` clean.

## Decisions made (also in DECISIONS.md)
- D-040: the step 5a design (see the entry).

## Experiments (also in EXPERIMENTS.md)
- None.

## State of the system
- DB: migrations up to 008; nothing new written this session (no regime/features rows yet: that's 5b).
- Scheduler: running, `com.qt.scheduler` (paper), on `89f522d`/`6cbfad0` code. No restart needed: 5a adds no jobs.
- Git: `8bab645` committed, **not pushed**.

## Blockers / surprises
- **BAA10Y's first ALFRED vintage is 2014-01-27.** Strict PIT means no HMM fit before Feb 2014. That's fine for a 2016 backtest start.
- **Quadrant is NaN right now.** `infl_mom_core_cpi` needs CPI 12 months back exactly, and Oct 2025 was never published (shutdown). So the YoY leg is NaN for Oct 2026 data until Nov 2026 CPI (~Dec 10). The same NaN hit Oct 2025 → Sep 2026 for any month whose 3- or 12-month lag fell on Oct 2025. Possible fix: a v2 of `infl_mom_*` that interpolates a single missing month geometrically, like Sahm tolerates one missing month. That needs Dom's call.
- exchange_calendars emits a pandas 4 deprecation warning on import (harmless).

## Next exact task
1. 5b: `matrix.py` (`features_asof(T, versions)`) and `build.py` (one entry point that loads every input, computes asset + market + regime + normalized features, writes `features`/`regime` append-only; backfill CLI). Plus the HMM refit job (writes `hmm_fits`, joblib, stability alert + report) and the `features_daily` 18:00 ET job with the stale-input gate. Restart the scheduler after.
2. 5c: §9 survivorship (`scripts/map_delisted_ciks.py`, SIC → GICS map, delisted bars via yfinance/Stooq).
3. Phase 2 gate: Congress `backfill.done`; 5 unattended days from Fri 2026-11-27.

## Questions to ask Dom next session
- Push `8bab645`?
- The missing Oct 2025 CPI: add a `infl_mom_*` v2 that bridges one missing month (so the quadrant isn't NaN for ~a year of history)?
