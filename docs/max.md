# Plan: move the harness to Windows (Steam BTD6), then XP-farm every monkey to tier 5 on all 3 paths

## Context
The harness was built for **BTD6+ on the Mac (Apple Arcade)**. Dom now plays **BTD6 on Steam on this Windows PC**, which is a new account with no progress. The work has two parts:
1. Make the harness drive Steam BTD6 on Windows smoothly.
2. Use it to farm XP, unattended through day and night, until **every upgrade on all 3 paths of every monkey is unlocked up to tier 5**. Paragons are out of scope. The goal is the least wall-clock time.

Dom's decisions: the PC is dedicated while farming. Background input is tried first, and the bot falls back to the real mouse.

What the exploration found:
- Only `bloons/platform_mac.py` is OS-specific. Portable code calls `game.grab / click / move / key / scroll / ocr / read_hud / reattach` and `frame.content_bgr()` (`bloons/harness.py:3`, grep confirmed). So a Windows backend with the same interface is the whole port.
- `bloons/data/ui.json` coordinates and `data/templates/*.png` were calibrated on a 1470×830 Mac window. The Steam build runs at a different aspect ratio and has a slightly different UI (home screen, events, rank unlocks), so both need recalibrating.
- BTD6 is installed at `C:\Program Files (x86)\Steam\steamapps\common\BloonsTD6`. This PC has Python 3.14, numpy and pywin32.
- XP facts from research (to re-verify live). Round XP is 20r for r ≤ 20, 40(r−20)+420 for r 21–50, and 90(r−50)+1620 for r > 50. Map difficulty adds +10%, +20% or +30% XP (Intermediate, Advanced, Expert). Freeplay XP is heavily reduced. **Round XP is split between towers in proportion to the cash invested in them.** The usual fast farm is **Deflation** (rounds 31–60, fixed starting cash) on an Expert map, about 65k XP per game. Community AFK farms report ~45.9k XP in 6–9 minutes.

## Part A — Windows port

### A1. `bloons/platform_win.py` (new; the second and last OS-specific file)
It gives the same `Game` and `Frame` interface as `platform_mac.py`, and `harness.py` stays unchanged.
- **Window:** `win32gui.FindWindow` for the game window. Read the client rect with `GetClientRect` + `ClientToScreen`, so there is no title bar and `title_bar_pt = 0`. Make the process DPI-aware (`SetProcessDpiAwareness(2)`) so coordinates are physical pixels.
- **Capture:** `PrintWindow(hwnd, PW_RENDERFULLCONTENT)` into a DIB, which yields a BGR numpy array and works while the window is covered. Fall back to `mss`/BitBlt of the screen rect if it returns black (some Unity/DX11 games do). `Frame` holds the array: `content_bgr()` returns it and `crop(box)` slices it.
- **Input:** try `PostMessage` (WM_LBUTTONDOWN/UP, WM_KEYDOWN/UP, WM_MOUSEWHEEL) first so the PC stays usable. If the game ignores it, use `SendInput` with the real cursor and foreground focus (`SetForegroundWindow` with the Alt-tap workaround). Pick the backend with one flag set after the feasibility probe. Key map: Windows virtual-key codes.
- **OCR:** use the built-in Windows OCR (`winrt-Windows.Media.Ocr`), which matches Apple Vision with no model download. Fall back to `rapidocr-onnxruntime` only if digits come back unreliable on the HUD fixtures.
- **Keep awake:** `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)`, plus a `screen_locked()` equivalent so a locked desktop isn't misread as a white screen.
- **Relaunch:** `relaunch()` uses `steam://rungameid/960090` for crash recovery.

### A2. Platform selection
Add a 5-line `bloons/platform.py`: `from .platform_win import *` on win32, otherwise `platform_mac`. Point `tools/play.py`, `tools/calibrate.py`, `tools/bloon_bg.py`, `scripts/*` and `tests/test_offline.py` at it. Update CLAUDE.md's rule to "platform_mac.py / platform_win.py are the only OS-specific files", and "Windows runs the harness (Steam)".

### A3. Feasibility gate (same idea as Phase 0)
Write a throwaway probe. It finds the window, captures a frame, clicks, presses `Q`, and confirms the game reacts, first with PostMessage and then with SendInput. **Gate: if neither input reaches the game, stop and re-plan.** Fix one windowed resolution, 1600×900 client at 16:9, and record it in START-HERE.

### A4. Recalibrate Steam BTD6
Keep the Mac file as `data/ui.mac.json`. Re-measure `ui.json` with the existing `tools/calibrate.py` and REPL `cursor`: HUD boxes, shop cells, upgrade panel, hero panel, ability bar, pause, map select, difficulty/mode pages and prompts. Re-capture the templates and fixtures (`btn_play_idle`, FF on/off, victory, defeat, home, difficulty_select, overwrite_save, collection, level-up/rank-up popups, which are new on Steam). Stamp the Steam game version. Fix the offline tests against new Windows fixtures (keep the Mac fixtures in a `mac/` subdir so their tests still run on the Mac).
Monkey Knowledge in `data/account.json` gets reset to empty (new account).

### A5. Live check (Phase 4 gate, on Windows)
In `tools/play.py`: `new_game` Monkey Meadow Easy, `place_tower`, `upgrade`, `start_round` (FF), `observe()`, `sell`, `restart`, `go_home`. Each must match the screen.

## Part B — XP farm

### B1. Research (first, ~1 session)
Confirm these, with sources in `RESEARCH.md`:
- the XP split rule and whether invested cash or something else drives it
- Deflation's unlock rule and starting cash
- the map difficulty bonus, and which Expert maps are unlocked on a fresh account
- the tower unlock ranks for a new account
- the XP cost of every upgrade for every tower, and the total. This goes in `data/xp_costs.json` with a version stamp.
- Monkey Knowledge that boosts XP, so MK points are spent on it first
- any XP caps, and whether Steam's "unspent XP / Monkey Teams / events (Races)" give faster XP

Then measure XP per game live: read a tower's XP bar before and after. This picks the actual farm.

### B2. `bloons/xp.py` + `data/xp_costs.json`
- **Progress reader:** OCR each tower's XP/upgrade screen in the Monkeys menu → XP banked + upgrades still locked.
- **Planner:** pick the next target tower(s). Cheapest remaining XP-to-unlock per game, with locked towers counted first as they unlock by rank.

### B3. `tools/farm.py` (the unattended loop)
1. `new_game(farm_map, "easy", "deflation")`.
2. Place a fixed, data-driven defense (`data/farm_builds.json`) that reliably clears 31–60. Put the remaining money into the **target towers**, so they take most of each round's XP.
3. `start_round` once (Auto Start on) and watch `observe()` until victory. Then collect and handle prompts, go home, and loop.
4. Every N games, re-read progress and re-plan targets.

Early on (low rank, Deflation or Expert maps locked), it plays Easy Standard on beginner maps with whatever towers are unlocked, until the farm setup is unlocked.

**Recovery:**
- If a game is lost, log it, restart, and lower build risk.
- On an unknown screen, `handle_prompts`, then Escape, then `go_home`.
- If the game crashes or the window is gone, run `relaunch()` and reattach.
- After 3 failed recoveries in a row, take a screenshot, log it, pause 5 min and retry.

Each game appends to `logs/farm.jsonl` (map, targets, XP gained, duration, outcome).
- **Running all night:** start `farm.py` as a detached background process (and optionally from Task Scheduler at logon) so it doesn't depend on a Claude session. In session I check its log and screen on a `/loop` wakeup cadence and fix issues that come up.

### B4. Done condition
The progress reader shows every tower × 3 paths × tier 5 unlocked. Then the farm stops and writes a final report.

## Files
- **New:** `bloons/platform_win.py`, `bloons/platform.py`, `bloons/xp.py`, `bloons/data/xp_costs.json`, `bloons/data/farm_builds.json`, `tools/farm.py`
- **Edited:** `bloons/data/ui.json` (+ `ui.mac.json`), templates and fixtures, `account.json`, `tests/test_offline.py`, imports in `tools/*` / `scripts/*`, `requirements.txt` (pywin32, winrt OCR, opencv if missing)
- **Docs:** CLAUDE.md, PLAN.md (new Phase W and Phase X), RESEARCH.md, START-HERE.md, handoffs

## Verification
- `pytest tests/test_offline.py` passes on Windows with the new fixtures.
- The A5 live gate in the REPL.
- One supervised Deflation game. A logged win and a tower XP increase matching the formula, within ±10%.
- A 2-hour unattended run with zero manual touches, including one forced recovery (kill the game mid-run, and it relaunches and continues).
- Then the long run until B4.

## Risks / notes
- **Time:** maxing all ~25 towers is likely millions of XP, which is tens of hours of farming. The exact estimate comes after B1 measures XP per game.
- Ninja Kiwi's terms discourage automation tools. The bot uses only pixels and synthetic input, with no modding or memory reading (project rule), but the account risk isn't zero.
- Unity can ignore PostMessage. If so, the real mouse is used (agreed with Dom).
