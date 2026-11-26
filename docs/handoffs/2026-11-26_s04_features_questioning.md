# Handoff: S04 · 2026-11-26 · Phase 3 questioning (features + regimes)

**Phase:** 3 (Feature layer + regime models), step (a) questioning
**Session goal:** Check the Phase 2 gate status, then run the Phase 3 questioning and write the build spec.
**Outcome:** ✅ done (questioning and spec). The Phase 2 exit gate is still running in parallel.

## What was done
- Phase 2 status checked:
  - Scheduler is running (launchd, paper).
  - The Congress backfill driver is still going (cursor: 115th congress, offset 2,250; 19,312 events so far).
  - **SoSoValue:** the `.env` line holds only a placeholder comment (`SOSOVALUE_API_KEY= # …`), so the collector skips correctly. No key yet.
- All 9 Phase 3 questions answered (inline in `docs/interrogation/03_features.md`):
  1. More data: full spec list + framework four + ~20 extra free series (collector top-up).
  2. BAA10Y always (closes D-001).
  3. Sectors: Wikipedia GICS for current members, SIC map for delisted.
  4. Hurst: variance ratio only.
  5. HMM defaults accepted.
  6. Sahm: below its 3-month average, with a 0.05 flat tolerance.
  7. Versioning: option A.
  8. New listings: NaN + masks.
  9. Survivorship: close both gaps.
- Measured the gaps: 135 of 731 equities have no CIK or facts. `cik-lookup-data.txt` returns HTTP 200.
- Found **TLT not collected** (needed for stock-bond correlation) and added it to the top-up.
- Wrote the build spec `docs/components/03_features.md` and D-033.

## Files touched
| File | Change |
|---|---|
| `docs/interrogation/03_features.md` | Answers Q1–Q9 |
| `docs/components/03_features.md` | New: Phase 3 build spec |
| `DECISIONS.md` | D-033 |
| `START_HERE.md` | v9 checkpoint |

## Tests
- None run (no code changed). Last run: 170 passed.

## Decisions made (also in DECISIONS.md)
- D-033: the Phase 3 answers above. Closes D-001.

## Experiments (also in EXPERIMENTS.md)
- None.

## State of the system
- DB: migrations up to 007. Congress backfill still in progress.
- Scheduler running? Yes, `com.qt.scheduler` (paper).
- Git: `308483b` plus this session's docs. Commit not made yet; `308483b` (SoSoValue collector) not pushed.

## Blockers / surprises
- SoSoValue still has no key (see above).
- `^MOVE` / `^SKEW` history depth on yfinance is unverified.
- The Japan 3-month rate FRED ID (`IR3TIB01JPM156N`) still needs checking in the build.

## Next exact task
1. Dom accepts or changes the **(default)** items in `docs/components/03_features.md`.
2. Phase 3 build step 1: collector top-up (§4) + migration 008 + the registry/inputs/adjust skeleton.
3. Keep watching the Phase 2 gate: Congress `backfill.done` and 5 unattended days, starting Fri 2026-11-27.

## Questions to ask Dom next session
- Accept the (default) items in the Phase 3 spec?
- OK to commit this session's docs and push them with `308483b`?
- SoSoValue key: paste it into `.env` after `SOSOVALUE_API_KEY=` (no `#`).

## Addendum: build step 1 (same session, after Dom accepted the defaults and said "start the build")
- Pushed `308483b`, `de6fe67`, `cb03720`. Step 1 committed as `bed7b04` (not pushed). 188 tests pass, ruff clean.
- **Top-up backfilled:**
  - FRED: 6 series, about 1.0M vintage rows. NFCI alone is 940k, because every weekly vintage revises its whole history.
  - Alpaca: 8 ETFs, 2,741 bars each from 2016.
  - yfinance: 5 tickers from 2000–2007.
- **Migration 008 applied to the live DB.** Service restarted on `bed7b04`.
- **New modules:** `features/registry.py`, `features/inputs.py`, `features/adjust.py`, plus `db.pit.frame_asof`.
- **Bugs caught:**
  - pandas 3 microsecond timestamps made `MacroState` see every release at once (a lookahead). Caught by the test and fixed.
  - `corp_actions` never backfilled symbols added after its first run. Fixed, and 580 rows recovered, including SOXX's 2024 split.
- **Adjustment checked on live data.** After the split price check, name-change ticker history, zero-volume filter and the KDP manual dividend, the biggest remaining adjusted daily moves are real events (bankruptcies, SVB/FRC, COVID, spin-offs) apart from PARA.
- **Open for Dom: contaminated identities (D-034 item 5).** Alpaca serves a symbol's *current* holder's history.
  - Asset 505 PARA holds Bannix's bars, all of them.
  - Asset 79 BBT holds another company's bars, all of them. The real BB&T history is served under TFC, which was confirmed with a 2017 probe.
  - A dollar-volume scan of every S&P member flagged only these two. WYND and AABA have no bars during their membership spells, so the survivorship gap list grows from 3 to 5.

## Addendum 2: identity fix (Dom: "go ahead with that fix")
- D-035, committed as `1b16d47`. 190 tests pass, ruff clean. Service restarted on it.
- **Overrides:** `collect.identity_overrides` config.
  - BBT: listed_until 2019-12-06, Alpaca bars excluded, 990 real BB&T bars refetched under TFC (source `alpaca_succ`).
  - PARA: listed_until 2025-08-06, Bannix bars excluded.
  - Membership and aliases were re-run, so both assets are now unlinked from the live Alpaca assets.
- **New collector** `succ_bars` (stock_daily; runs once per override entry).
- **Daily summary** has an "Identity check" section: members with median dollar volume during membership < $10M. Without the overrides it flags exactly PARA and BBT; with them, nothing.
- **Survivorship list (spec §9.3)** is now SNDK, TE, STI, Paramount 2022–2025, WYND, AABA.
- Feature loaders must pass `exclude=inputs.bar_exclusions(conn, overrides)` to `load_bars`. The step-2 build wires this into one entry point.

## Next exact task (updated)
1. Build step 2: per-asset price features (§5.1), with hand-calc + no-lookahead tests parametrized over the registry. Bars come through `load_bars(..., exclude=bar_exclusions(...))` + `mapped_corp_actions` + `adjust`.
2. Ask Dom to push `bed7b04`, `3a951b2`, `1b16d47` (+ this handoff).
