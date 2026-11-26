# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v3 / 2026-11-25 (Session 02, Phase 2 questioning)

| | |
|---|---|
| **Phase** | Phase 1 ✅ (pushes confirmed on Dom's phone). **Phase 2 (Collectors): questioning done, build next** |
| **Last session** | S02: Phase 1 close-out + Phase 2 questioning ([handoff](docs/handoffs/2026-11-25_s02_collectors_questioning.md)) |
| **Git** | Branch `vibe/quant_trader`. Phase 1 build is committed as `9788e9c`; the S02 docs are committed. Nothing is pushed |
| **Spec** | [`docs/components/02_collectors.md`](docs/components/02_collectors.md), decisions D-021 to D-026 |
| **Tests** | 55 passed, ruff clean (no code changed in S02) |
| **DB** | Migrated to 004, no market data yet |

### Next exact task
1. Dom fills the empty `.env` keys (the spec defaults are accepted): Alpaca paper, FRED, `CONGRESS_API_KEY`, `COINGECKO_DEMO_API_KEY` (optional: SoSoValue, Finnhub).
2. Phase 2 build. Start with migration 005 and the PIT helpers, then get the **news loop + scheduler** running first so history starts accumulating (D-018).

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
### v2 / 2026-11-25 (Session 01, build)
- Phase 1 built: config, run_id, logging, ntfy alerts, calendar, migrations 001–004, backup job. 55 tests green. Committed in S02 as `9788e9c`.

### v1 / 2026-11-25 (Session 01, questioning)
- Phase 1 questioning done and the build spec written (D-012 to D-018). Commit `5e6985a`.

### v0 / 2026-11-25 (Session 00)
- Phase 0 (Framework) done: repo, docs, folder tree and branch.
- Next task was the Phase 1 questioning, which was completed in S01.
- [Handoff](docs/handoffs/2026-11-25_s00_framework.md)
