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
