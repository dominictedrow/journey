# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v2 / 2026-11-25 (Session 01, build)

| | |
|---|---|
| **Phase** | **Phase 1 (Infrastructure): built**. One exit-gate item is left (Dom's phone check), then **Phase 2 questioning** |
| **Last session** | S01: Phase 1 questioning and build ([handoff](docs/handoffs/2026-11-25_s01_infra_questioning.md)) |
| **Git** | Branch `vibe/quant_trader`. Questioning docs are committed as `5e6985a`; the build is not yet committed. Nothing is pushed |
| **Code written** | Config, run_id, logging, ntfy alerts, calendar, migrations 001 to 004, backup job |
| **Tests** | `uv run pytest`: 55 passed. Ruff is clean |
| **DB** | `docker compose up -d` brings up `qt_db` (54320) and `qt_db_backup` (54321). Primary is migrated to 004 and has no data yet |

### Next exact task
1. Dom subscribes to `NTFY_TOPIC` (in `.env`) in the ntfy app and confirms the test pushes arrived. Then mark Phase 1 ✅.
2. Dom confirms D-020: pushes only in the `paper` profile.
3. Start the **Phase 2 questioning**, with the news collector and the `BAMLH0A0HYM2` archive first (D-001, D-018). Only the collectors will run on the Mac.

Until the Phase 2 scheduler exists, the backup runs by hand: `QT_PROFILE=paper uv run python scripts/db_backup.py`.

### Open questions for Dom (carried forward)
- [ ] The LLM's final size (spec: 50–125M) and the KV-cache experiment design → Phase 8
- [ ] Whole shares vs fractional shares. Fractional is DAY-only, so GTC stops at the broker can't protect fractional positions (D-003) → Phase 6
- [ ] Keep the "never sell same session" rule now that PDT is retired? (D-002) → Phase 5
- [ ] Is the SoSoValue key free? If not, drop ETF flows from v1 (D-007) → Phase 2
- [ ] The paper account's starting NAV. Paper defaults to $100k, so should it match the real capital you plan to use? → Phase 5

### Prohibited until a later gate
- Live trading. Paper only until the Phase 11 go-live gate and Dom's explicit authorization.
- Paid APIs without Dom's approval.
- Training the text models (Transformer, CNN #2) before 6+ months of news exist.

---

## Prior checkpoints
### v1 / 2026-11-25 (Session 01, questioning)
- Phase 1 questioning done and the build spec written (D-012 to D-018). Commit `5e6985a`.

### v0 / 2026-11-25 (Session 00)
- Phase 0 (Framework) done: repo, docs, folder tree and branch.
- Next task was the Phase 1 questioning, which was completed in S01.
- [Handoff](docs/handoffs/2026-11-25_s00_framework.md)
