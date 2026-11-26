# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v9 / 2026-11-26 (Session 04, Phase 3 questioning)

| | |
|---|---|
| **Phase** | **Phase 3 (Features + regimes): build step 1 done** (top-up, migration 008, registry/inputs/adjust); Phase 2 exit gate still running in parallel |
| **Last session** | S04: Q1–Q9 answered, [`docs/components/03_features.md`](docs/components/03_features.md), D-033 (closes D-001) ([handoff](docs/handoffs/2026-11-26_s04_features_questioning.md)) |
| **Git** | S04 docs pushed (`cb03720`); step 1 `bed7b04` committed, not pushed |
| **Tests** | 188 passed, ruff clean |
| **Service** | launchd `com.qt.scheduler` (paper); Congress backfill driver still running |

### Next exact task
1. Dom decides the PARA/BBT contaminated-identity fix (D-034 item 5, handoff addendum), then implement it.
2. Phase 3 build step 2: per-asset price features (§5.1).
3. Phase 2 gate: Congress `backfill.done`; 5 unattended days from Fri 2026-11-27.

### Open questions for Dom
- [ ] Identity fix for PARA/BBT (exclude the wrong bars, refetch BB&T under TFC)? Push `bed7b04`?
- [ ] **SoSoValue key:** the `.env` line still holds only a placeholder comment. Paste the key after `SOSOVALUE_API_KEY=`
- [ ] Carried: LLM size (Phase 8), whole vs fractional shares (Phase 6), same-session rule (Phase 5), starting NAV (Phase 5), crypto size-cap volume source (Phase 6), short-history trading threshold (Phase 5, from Q8)

---

## Prior checkpoints
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
