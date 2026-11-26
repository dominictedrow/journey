# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v13 / 2026-11-26 (Session 04, Phase 3 build step 4b)

| | |
|---|---|
| **Phase** | **Phase 3 (Features + regimes): build steps 1–4 done** (foundations; 36 price features; fundamentals + ranks + composites, D-037; 57 market features, D-038; event clock + crypto structure, 13 features + FOMC/CPI calendar collectors, D-039); Phase 2 exit gate still running in parallel |
| **Last session** | S04: questioning, spec, steps 1–4b, identity fix ([handoff](docs/handoffs/2026-11-26_s04_features_questioning.md)) |
| **Git** | `89f522d` pushed (4a `509f639` was already pushed) |
| **Tests** | 650 passed, ruff clean |
| **Service** | launchd `com.qt.scheduler` (paper), restarted on `89f522d` (PID 62917); weekly job runs `fomc_cal`/`cpi_cal` from Sun 18:00 ET; Congress backfill driver still running |

### Next exact task
1. Phase 3 build step 5: regime (§6), normalization + masks (§7), `matrix.py`/`build.py`, §9 survivorship.
2. Phase 2 gate: Congress `backfill.done`; 5 unattended days from Fri 2026-11-27.

### Open questions for Dom
- [ ] Carried: LLM size (Phase 8), whole vs fractional shares (Phase 6), same-session rule (Phase 5), starting NAV (Phase 5), crypto size-cap volume source (Phase 6), short-history trading threshold (Phase 5, from Q8)

---

## Prior checkpoints
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
