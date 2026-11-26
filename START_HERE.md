# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v8 / 2026-11-26 (Session 03e, Phase 2 build step 5)

| | |
|---|---|
| **Phase** | **Phase 2 (Collectors): build complete (5 of 5); exit gate pending** (Congress backfill finishing, 5-day unattended run) |
| **Last session** | S03e: yfinance, corporate actions, EDGAR facts (1.52M), earnings (25k), Congress (in progress), CoinGecko, DefiLlama, funding, OI ([handoff](docs/handoffs/2026-11-26_s03e_collectors_build_rest.md)) |
| **Git** | Branch `vibe/quant_trader`; steps 1–5 pushed; SoSoValue collector committed after |
| **Spec** | [`docs/components/02_collectors.md`](docs/components/02_collectors.md), decisions D-021 to D-031 |
| **Tests** | 170 passed, ruff clean |
| **DB** | Migrated to 007; every Phase 2 raw table populated; 0 null `released_at` |
| **Service** | launchd `com.qt.scheduler` (paper) under `caffeinate -is`, on step 5 code. `uv run python scripts/install_scheduler.py --status` |

### Next exact task
1. Confirm the Congress backfill finished (`collector_state` cursor `backfill.done`); the 06:00 ET job continues it otherwise.
2. Phase 2 exit gate: 5 consecutive unattended days (component doc §13 item 5). Check each daily summary.
3. Meanwhile: Phase 3 questioning (now 9 questions; Q9 = survivorship gaps, D-032).

### Open questions for Dom (carried forward)
- [ ] **Dom: get a SoSoValue key** (sosovalue.com/developer/dashboard → "Apply your Key"), put it in `.env` as `SOSOVALUE_API_KEY`, restart the service. Then check the history depth on the first run (D-032)
- [ ] The LLM's final size (spec: 50–125M) and the KV-cache experiment design → Phase 8
- [ ] Whole shares vs fractional shares (D-003) → Phase 6
- [ ] Keep the "never sell same session" rule now that PDT is retired? (D-002) → Phase 5
- [ ] The paper account's starting NAV → Phase 5
- [ ] Crypto order-size cap: which volume source (D-021) → Phase 6

### Prohibited until a later gate
- Live trading. Paper only until the Phase 11 go-live gate and Dom's explicit authorization.
- Paid APIs without Dom's approval.
- Training the text models (Transformer, CNN #2) before 6+ months of news exist.

---

## Prior checkpoints
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
