# START HERE

> Read this first every session, then read the newest handoff in `docs/handoffs/`.

---

## CURRENT CHECKPOINT: v0 / 2026-11-25 (Session 00)

| | |
|---|---|
| **Phase** | 0 (Framework) is done → **Phase 1 (Infrastructure), step (a): questioning** |
| **Last session** | S00: repo, docs, folder tree and branch created ([handoff](docs/handoffs/2026-11-25_s00_framework.md)) |
| **Git** | Branch `vibe/quant_trader` on `dominictedrow/journey` |
| **Code written** | None yet. Package skeleton only. |
| **Tests** | None yet |

### Next exact task
Run the **Phase 1 questioning** with Dom, using [`docs/interrogation/01_infrastructure.md`](docs/interrogation/01_infrastructure.md):
1. Ask one question at a time and record his answers inline.
2. Write `docs/components/01_infrastructure.md`.
3. Build Phase 1: Docker TimescaleDB, schema, config loader, run_id and calendar utilities.

**Urgent: news collection starts on day one.** Text models can only learn from news collected from the first collection day onward. Phase 2 should reach the news collector quickly, even before the other collectors are finished. Also archive `BAMLH0A0HYM2` daily from day one (D-001).

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
_(none)_
