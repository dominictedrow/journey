# Bloons Bot — Project Instructions

RL agent + pixel-based harness for **Bloons TD 6+ (Apple Arcade, macOS)**. Full plan: `docs/PLAN.md`. Game facts: `docs/RESEARCH.md`.

## Session protocol (mandatory)

### On "continue" (or any session start without a specific task)
1. Read `docs/START-HERE.md`.
2. Read the newest file in `docs/handoffs/` (highest `NNN`).
3. Read the `docs/PLAN.md` section that START-HERE points to.
4. Resume at START-HERE's **Next action**. Don't re-plan finished steps.

### Handoff (written automatically, no need to be asked)
Write a handoff when any of these happens:
- a PLAN.md step is completed
- Dom says "wrap up", "handoff", "done for today", or similar
- the session is ending, or the context is getting long (before it gets summarized)

Steps:
1. Create `docs/handoffs/NNN-YYYY-MM-DD.md`, where NNN is the previous number + 1, zero-padded. Use the template in `docs/handoffs/README.md`.
2. Overwrite `docs/START-HERE.md`: phase/step pointer, status, link to the new handoff, one concrete next action. Keep it to one screen.
3. Tick finished checkboxes in `docs/PLAN.md`. Add new findings to `docs/RESEARCH.md`.

Handoffs are append-only. Never edit an old one.

## Project rules
- The harness runs in Python on the **Mac**. Windows is only for editing, git, and offline tests (`tests/test_offline.py`, which runs on saved screenshots).
- `bloons/platform_mac.py` is the only OS-specific file. Everything else stays portable.
- UI coordinates, costs and ranges live in `bloons/data/*.json`, never in code. Data files carry a game-version stamp.
- Harness actions return `ActionResult(ok, reason)` and don't raise on game-side failures.
- Use the hotkey when `RESEARCH.md` marks it as confirmed, otherwise click the UI coordinate.
- No game modding or memory reading. Pixels and synthetic input only.
