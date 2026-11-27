# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v16 / 2026-11-26 (Session 07, Phase 3 build step 5c)

| | |
|---|---|
| **Phase** | **Phase 3: steps 1–5c done** (5c survivorship, D-043, migration 010). Rebuild verified. Left: the exit gate (3 unattended `features_daily` sessions). Phase 2 gate still running in parallel |
| **Last session** | S07 ([handoff](docs/handoffs/2026-11-26_s07_survivorship_step5c.md)) |
| **Git** | Pushed through `4bc3579` |
| **Tests** | 739 passed, ruff clean |
| **Service** | `com.qt.scheduler` (paper), PID 22950 on `5008828`. First unattended `features_daily` 2026-11-26: ok |
| **Data** | 120/137 former-member CIKs (+298k EDGAR facts), 226 historical sectors. `features` rebuilt 2016-01-01 → 2026-11-26: 194.8M rows, 1.7 GB compressed. Fundamentals coverage bug (63 current + 10 former names lack price ratios), see S07 handoff |

### Next exact task
1. ~~Verify the rebuild~~: done (S07 handoff, Rebuild results).
2. Exit gate: `features_daily` unattended Fri 11-27, Mon 11-30, Tue 12-1; Phase 2 gate (Congress `backfill.done`, 5 unattended days from Fri 11-27). Then close Phase 3 and start Phase 4 questioning.

### Open questions for Dom
- [x] Pushed through `4bc3579`.
- [ ] Fix the fundamentals coverage bug (multi-class shares, missing net income) now, or after the exit gate?
- [x] 17 CIK review names: dropped by Dom, accepted gap with no CIK (D-044).
- [ ] DECISIONS has two `D-041` rows (append-only; left as is). Add a renumbering note?
- [ ] Carried: LLM size (Phase 8), whole vs fractional shares (Phase 6), same-session rule (Phase 5), starting NAV (Phase 5), crypto size-cap volume source (Phase 6), short-history trading threshold (Phase 5, from Q8)

---

## Prior checkpoints
### v15 / 2026-11-26 (Session 05, Phase 3 build step 5b)
- Build/matrix/jobs, migration 009, D-042; inflation v2 (D-041). `d8c37b4`/`70b05bf`/`153e9ad` on origin.
### v14 / 2026-11-26 (Session 05, Phase 3 build step 5a)
- Regime (quadrant, HMM, stress), normalization, masks, staleness (D-040). Pushed as `8bab645`/`97ba2f2`.
### v13 / 2026-11-26 (Session 04, Phase 3 build step 4b)
- Event clock + crypto structure (D-039), SoSoValue envelope fix. Pushed as `89f522d`/`6cbfad0`.
### v12 / 2026-11-26 (Session 04, Phase 3 build step 4a)
- 57 market features (`features/market.py`), ALFRED fixes, D-038. Pushed as `509f639`.
### v11 / 2026-11-26 (Session 04, Phase 3 build step 3)
- Fundamentals, ranks, composites, EDGAR top-up + predecessor CIKs, D-037. Pushed as `32ad4b8`.
### v10 / 2026-11-26 (Session 04, Phase 3 build step 2)
- 35 per-asset price features, per-feature readiness, D-036. Pushed as `498b7f2`.
### v9 / 2026-11-26 (Session 04, Phase 3 questioning + build step 1)
- Q1–Q9 answered, spec `docs/components/03_features.md`, D-033 to D-035: top-up, migration 008, registry/inputs/adjust, identity overrides. [Handoff](docs/handoffs/2026-11-26_s04_features_questioning.md)
### v8 / 2026-11-26 (Session 03e, Phase 2 build step 5)
- Phase 2 build complete (5 of 5): yfinance, corp actions, EDGAR facts, earnings, Congress, crypto structure, SoSoValue collector (D-031, D-032). `32c5117`/`e147c04` pushed, `308483b` not. [Handoff](docs/handoffs/2026-11-26_s03e_collectors_build_rest.md)
### v7 / 2026-11-26 (Session 03d, Phase 2 build step 4)
- Alpaca bars/assets, S&P 500 membership as securities, aliases, migration 007, D-030. Committed as `eca541a`, pushed. [Handoff](docs/handoffs/2026-11-26_s03d_collectors_build_prices.md)
### v6 / 2026-11-25 (Session 03c, Phase 2 build step 3)
- FRED/ALFRED all vintages (186k rows), HY archive, migration 006, D-029. Committed as `ba2804b`. [Handoff](docs/handoffs/2026-11-25_s03c_collectors_build_fred.md)
### v5 / 2026-11-25 (Session 03b, Phase 2 build step 2)
- News loop, Fed RSS, scheduler, launchd service (paper). Committed as `434e551`. [Handoff](docs/handoffs/2026-11-25_s03b_collectors_build_news_scheduler.md)
### v4 / 2026-11-25 (Session 03, Phase 2 build step 1)
- Migration 005, `collect/{pit,http,base}.py`, `db/pit.py`, PIT + no-lookahead tests. Committed as `34fbe52`. [Handoff](docs/handoffs/2026-11-25_s03_collectors_build_foundations.md)
### v3 / 2026-11-25 (Session 02, Phase 2 questioning)
- Phase 1 closed (pushes confirmed). Phase 2 questioning done, spec `docs/components/02_collectors.md`, D-021 to D-026. [Handoff](docs/handoffs/2026-11-25_s02_collectors_questioning.md)
### v2 / 2026-11-25 (Session 01, build)
- Phase 1 built: config, run_id, logging, ntfy alerts, calendar, migrations 001–004, backup job. 55 tests green. Committed in S02 as `9788e9c`.

### v1 / 2026-11-25 (Session 01, questioning)
- Phase 1 questioning done and the build spec written (D-012 to D-018). Commit `5e6985a`.

### v0 / 2026-11-25 (Session 00)
- Phase 0 (Framework) done: repo, docs, folder tree and branch.
- Next task was the Phase 1 questioning, which was completed in S01.
- [Handoff](docs/handoffs/2026-11-25_s00_framework.md)
