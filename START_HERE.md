# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v4 / 2026-11-25 (Session 03, Phase 2 build step 1)

| | |
|---|---|
| **Phase** | **Phase 2 (Collectors): build in progress, step 1 of 5 done** (foundations) |
| **Last session** | S03: migration 005, `collect/{pit,http,base}.py`, `db/pit.py`, PIT + no-lookahead tests ([handoff](docs/handoffs/2026-11-25_s03_collectors_build_foundations.md)) |
| **Git** | Branch `vibe/quant_trader`, last commit `fc919dc`. S03 work uncommitted. Nothing pushed |
| **Spec** | [`docs/components/02_collectors.md`](docs/components/02_collectors.md), decisions D-021 to D-027 |
| **Tests** | 102 passed, ruff clean |
| **DB** | Migrated to 005, no market data yet |
| **Keys** | Alpaca paper, FRED, Congress, CoinGecko, SEC UA set. SoSoValue, Finnhub empty (optional) |

### Next exact task
1. Build step 2: news loop + Fed RSS + scheduler + launchd installer, so news history starts accumulating (D-018). Ask Dom before loading the launchd agent.
2. Then FRED/ALFRED + HY archive → Alpaca + membership + backfill → everything else.

Until the scheduler exists, the backup runs by hand: `QT_PROFILE=paper uv run python scripts/db_backup.py`.

### Open questions for Dom (carried forward)
- [ ] The LLM's final size (spec: 50–125M) and the KV-cache experiment design → Phase 8
- [ ] Whole shares vs fractional shares (D-003) → Phase 6
- [ ] Keep the "never sell same session" rule now that PDT is retired? (D-002) → Phase 5
- [ ] Is the SoSoValue key free? Verified during the Phase 2 build (D-007)
- [ ] The paper account's starting NAV → Phase 5
- [ ] Crypto order-size cap: which volume source (D-021) → Phase 6

### Prohibited until a later gate
- Live trading. Paper only until the Phase 11 go-live gate and Dom's explicit authorization.
- Paid APIs without Dom's approval.
- Training the text models (Transformer, CNN #2) before 6+ months of news exist.

---

## Prior checkpoints
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
