# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v6 / 2026-11-25 (Session 03c, Phase 2 build step 3)

| | |
|---|---|
| **Phase** | **Phase 2 (Collectors): build in progress, step 3 of 5 done** (FRED/ALFRED + HY archive) |
| **Last session** | S03c: ALFRED all vintages (186k rows, 23 series), HY archive, migration 006, PIT release fixes ([handoff](docs/handoffs/2026-11-25_s03c_collectors_build_fred.md)) |
| **Git** | Branch `vibe/quant_trader`; steps 1–2 `34fbe52`, `434e551`, step 3 committed after. Nothing pushed |
| **Spec** | [`docs/components/02_collectors.md`](docs/components/02_collectors.md), decisions D-021 to D-029 |
| **Tests** | 134 passed, ruff clean |
| **DB** | Migrated to 006; ALFRED loaded; articles + Fed events accumulating |
| **Service** | launchd `com.qt.scheduler` running (paper). `uv run python scripts/install_scheduler.py --status`; heartbeat `data/run/heartbeat.json` |

### Next exact task
1. Build step 4: Alpaca bars (equities/ETFs from 2016 + crypto) + assets + fja05680 membership + EDGAR ticker map → aliases + backfill.
2. Then step 5 (yfinance, EDGAR facts, Congress, earnings, crypto structure) and the backfill report.

### Open questions for Dom (carried forward)
- [ ] Fed Board roles in `configs/collect.yaml` correct? (Warsh Chair?)
- [ ] OK with D-029 release-time changes (18:00 ET default; HY/VIX next session 12:00 ET)?
- [ ] `caffeinate` the service so Mac sleep doesn't break the 5-day gate?
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
