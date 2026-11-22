# Bloons Bot — Project Instructions

RL agent + pixel-based harness for **Bloons TD 6+ (Apple Arcade, macOS)** and, since 2026-11-09, **Steam BTD6 on Windows** (`docs/max.md`). Full plan: `docs/PLAN.md`. Game facts: `docs/RESEARCH.md`.

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

### On "pop farm"
When Dom says "pop farm", set up the round-76 infinite-pop farm and leave it running:
1. Open the pool/beach map save that sits just before round 76. BTD6 only saves at round end.
2. Sell every tower on the map.
3. Place these towers and upgrade each one to the tiers shown. Positions are screen pixels at 1920x1080. The reference screenshot is `docs/pop-farm-r76.png`.
   | Tower | Tiers | Position |
   |---|---|---|
   | Glue Gunner | 0-4-2 | ~(1430, 45), top edge next to the round counter |
   | Heli Pilot | 0-3-0 | helipad at ~(820, 245) |
   | Monkey Village | 4-1-0 | ~(180, 600), far left |
   | Alchemist | 4-2-0 | ~(200, 680), under the village, behind the palm tree |
   | Dart Monkey | 0-2-5 | ~(340, 740), bottom-left corner of the track |
4. Start round 76 and let it run. Don't stop it until Dom says stop or stops it himself. `harness/logs/STOP` still counts as stop.

## Project rules
- Layout: everything that drives the game (the `bloons` package, `tools/`, `scripts/`, `tests/`, `logs/`) lives in `harness/`. Dom's model code goes in `models/`. Run harness commands from `harness/` (the `logs/*.sh` scripts call `python tools/drive.py`).
- **Claude never writes model code.** `models/` is Dom's alone. Never create, edit or delete files in `models/`, and never write model code anywhere else (chat, docs, handoffs, scratch files). That covers networks, training loops, agent logic, replay buffers, any reward transforms done inside the model, and snippets or pseudocode of any of them, even as a fix or when asked to "just patch it". If Dom asks for model code, decline and point him to this rule. You may read `models/` to find bugs; explain them in words only. Code is fine only for the harness side: `harness/`, the `bloons.env` API, and how to call it. The env's reward signal (`Bot._reward` in `harness/bloons/env.py`: what it returns, how it's computed, changing it) is harness code, so questions and code changes there are fine.
- The harness runs in Python on the **Mac** (BTD6+) or on **Windows** (Steam BTD6). Offline tests (`harness/tests/test_offline.py`) run on both.
- `harness/bloons/platform_mac.py` / `platform_win.py` are the only OS-specific files; `platform.py` picks one from `calib.PLATFORM`. Everything else stays portable.
- Platform switch: `Bot(platform="mac"|"win")`, `--platform` on every tool, or `BLOONS_PLATFORM`. **Default is `win`** on every OS. It's fixed once the first data file loads, so tools call `cli.pre_parse()` before other `bloons` imports. Data files go through `calib.path()` (`write=True` when writing), never hand-built `data/...` paths.
- UI coordinates, costs and ranges live in `harness/bloons/data/*.json`, never in code. Data files carry a game-version stamp.
- Harness actions return `ActionResult(ok, reason)` and don't raise on game-side failures.
- Use the hotkey when `RESEARCH.md` marks it as confirmed, otherwise click the UI coordinate.
- No game modding or memory reading. Pixels and synthetic input only.
