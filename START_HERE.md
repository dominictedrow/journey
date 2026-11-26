# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v1 / 2026-11-25 (Session 01)

| | |
|---|---|
| **Phase** | **Phase 1 (Infrastructure), step (b): build**. Step (a) questioning is done |
| **Last session** | S01: Phase 1 questioning done, build spec written, D-012 to D-018 logged ([handoff](docs/handoffs/2026-11-25_s01_infra_questioning.md)) |
| **Git** | Branch `vibe/quant_trader` on `dominictedrow/journey`. S01 doc changes not yet committed |
| **Code written** | None yet. Package skeleton only. |
| **Tests** | None yet |

### Next exact task
**Build Phase 1** from [`docs/components/01_infrastructure.md`](docs/components/01_infrastructure.md), in this order:
1. uv project
2. docker-compose (`db` + `db_backup`)
3. migrations 001 to 004 and the runner
4. config and profiles
5. `run_id` and `run_context`
6. logging and ntfy alerts
7. `scripts/db_backup.py`
8. calendar
9. tests

Exit gate: §14 of that file, including one test push reaching Dom's phone.

Before building, check that Docker Desktop is installed and that Dom has the ntfy app; generate an `NTFY_TOPIC` for him.

**Urgent: news collection starts on day one.** Per D-018, only the collectors run on the Mac. Text models can only learn from news collected from the first collection day onward. Phase 2 should reach the news collector quickly, even before the other collectors are finished. Also archive `BAMLH0A0HYM2` daily from day one (D-001).

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
### v0 / 2026-11-25 (Session 00)
- Phase 0 (Framework) done: repo, docs, folder tree and branch.
- Next task was the Phase 1 questioning, which was completed in S01.
- [Handoff](docs/handoffs/2026-11-25_s00_framework.md)
