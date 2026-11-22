# Plan: a `--platform mac|win` switch for the harness

> **Revised 2026-11-22 (Dom):** the main switch is the constructor arg `Bot(platform="mac")`, set in `run.py`, and the default is **win** on every OS. Steps 1-5 are done (handoffs 062-063). Sections 1-4 below describe what was built.

## Context
Dom wants one flag that puts the harness in Mac mode (BTD6+, Apple Arcade) or Windows mode (Steam BTD6). Since the Windows port (Phase W, 2026-11-09) Windows is the mode that gets built out. Mac still works offline (121/121 `tests/test_offline.py` pass on the Mac, 2026-11-22), but nobody has run it live since the port, and there's no explicit way to choose it.

**How it's chosen today (two hidden switches):**
- `bloons/platform.py` picks the input/capture backend from `sys.platform` (`platform_win` on Windows, `platform_mac` otherwise).
- `bloons/calib.py` picks the data set: `NAME = BLOONS_CALIB or ("steam" if win32 else "mac")`. `calib.path()` maps `ui.json` to `ui.mac.json`, `templates/` to `templates/mac/`, `maps/x.json` to `maps/x.mac.json`.
- Both are read **at import time**. `ui.py`, `vision.py`, `platform_*.py` and `harness.py` load their JSON at module level, so a flag has to be set before the first `import bloons.*`.

**Hard constraint:** each backend only runs on its own OS. `platform_mac` needs Quartz/Vision (pyobjc) and `platform_win` needs win32 APIs. So `--platform mac` on a Windows PC can't drive a game. It can only run offline (tests, fixtures). The flag selects a mode and checks it against the OS. It can't emulate the other OS.

## Design

### 1. One resolver: `bloons/calib.py`
- Add `PLATFORM`, resolved in this order:
  1. `Bot(platform=...)` / `--platform` (via `calib.use()`, before any data loads)
  2. `BLOONS_PLATFORM` env var (`mac` | `win`; `steam` also means `win`)
  3. the default, `win`
- `NAME` (the calibration) follows from `PLATFORM`: `win` gives `steam`, `mac` gives `mac`. `BLOONS_CALIB` stays as a calibration-only override, so the existing tests (`test_offline.py` pins `mac`, `test_steam.py` pins `steam`) keep working unchanged.
- Add `calib.describe()`, which returns one line like `platform=mac calib=mac data=ui.mac.json templates=templates/mac`. Every tool prints it at startup, so a run's log shows which mode it was in.

### 2. Backend pick: `bloons/platform.py`
- Import `platform_mac` or `platform_win` from `calib.PLATFORM`, not from `sys.platform`.
- If the mode doesn't match the OS, raise a clear error at import: `--platform mac needs macOS (BTD6+); this is win32`. Offline code that never imports `bloons.platform` (tests, `env.ACTIONS`, obsvec) is unaffected.

### 3. The CLI flag: `bloons/cli.py` (new, tiny, imports nothing else from `bloons`)
- `pre_parse(argv=sys.argv)`: removes `--platform X` / `--platform=X` from argv, sets `os.environ["BLOONS_PLATFORM"]`, and returns the value. It must run **before** any `from bloons... import`, so it is the first two lines of each tool:
  ```python
  from bloons import cli; cli.pre_parse()
  from bloons.harness import BloonsHarness  # now loads the right calibration
  ```
  (`bloons/__init__.py` is empty, so importing `bloons.cli` loads no data.)
- Tools that get it: `tools/drive.py`, `play.py`, `farm.py`, `calibrate.py`, `bloon_bg.py`, `bananas.py`, `scripts/watch_state.py`. Each tool's argparse also lists `--platform` (choices `mac`, `win`) so `--help` shows it. Pre-parse already did the real work.
- Guard: if `bloons.ui` or `bloons.vision` was already imported with another calibration when `pre_parse` runs, raise. This catches tools that were wired in the wrong order.

### 4. The env API: `bloons/env.py` (built)
- `Bot(..., platform=None)`: `'win'` (Steam BTD6) or `'mac'` (BTD6+). `None` keeps `BLOONS_PLATFORM`, else `win`. `Bot` calls `calib.use(platform)` **before** it imports `bloons.harness`, and importing `bloons.env` reads no data files, so `Bot(platform="mac")` really switches.
- After a process has loaded data for one platform, a `Bot` with the other platform raises `RuntimeError`. It never switches silently.
- `bot.platform` shows the platform in effect.
- `calib.path(rel, write=True)` always names the `.mac.json` twin in Mac mode. That way tools that write map files (`calibrate.py`, and the readers in `bloon_bg.py` / `train_bloons.py`) never touch Steam files in Mac mode.

### 5. Mac parity: what Mac mode needs to be "usable", not only selectable
The Windows side gained features the Mac never got. Each gap gets closed, or Mac mode refuses it clearly:

| Gap | Where | Fix |
|---|---|---|
| `Game.relaunch()` missing on Mac | `tools/farm.py` recover() raises AttributeError | Add it to `platform_mac.Game`: `open -a "Bloons TD 6+"`, then `reattach()`. Exact app name goes in `data/ui.mac.json`, not code. |
| `Game.read_number()` missing | Windows reads prices/XP with glyph templates | On Mac, fall back to Vision OCR (`self.ocr`) plus the existing HUD number parsing |
| `Game.width/height` missing | used by Windows-era callers | Add properties from the window bounds |
| `keep_awake()` missing | Windows `SetThreadExecutionState` | Mac: hold a `caffeinate -dims` subprocess, stop it on `keep_awake(False)` |
| UI keys only in `ui.json`: `menus.victory_home/freeplay`, `popup_ok`, `level_up_continue`, `unlock_monkey`, `mode_deflation/magic_only`, `monkeys_menu`, `xp_screen`, `pause.store`, `prompts.*` | post-game menus, XP farm | **Calibrate on BTD6+** (`tools/calibrate.py` + screenshots) for the in-game ones: victory_home, popup_ok, level_up. Mark the Phase X farm keys (monkeys_menu, xp_screen, unlock_monkey) as Steam-only. |
| Templates only on Steam: `level_up`, `popup_ok`, `unlock_monkey`, `victory_rewards`, `xp_pill`, `xp_screen`, `monkeys_menu`; probes `letterbox`, `reward_bg` | `state.py` skips them on Mac already | Cut Mac templates for `level_up`, `popup_ok`, `victory_rewards` (a player level-up popup mid-game would stall a Mac run). Leave the rest Steam-only. |
| Kill switch is Windows-only (`tools/stop.cmd`) | | Add `tools/stop.sh` (writes `harness/logs/STOP`) |
| XP farm (`tools/farm.py`, `bloons/farm.py`, `bloons/xp.py`) | Steam account (Gift Box) | On Mac, exit at start with "farm is Steam-only (docs/max.md Part B)" until Dom wants it on BTD6+ |

**Parity test** (offline, runs on both OSes): parse `platform_mac.py` and `platform_win.py` with `ast` and check that `Game`, `Frame` and the module-level functions have the same public names. That stops the two from drifting again. Also check that every ui/templates key the shared code reads is in both data sets, or is on an explicit Steam-only list.

### 6. Tests (`tests/test_offline.py`, all offline)
- Resolver: env var → OS default; `steam` alias; bad value raises; `BLOONS_CALIB` still overrides the calibration only.
- `cli.pre_parse` removes the flag from argv in both spellings and sets the env.
- `platform.py` mismatch raises the clear error (run in a subprocess with `BLOONS_PLATFORM=win` on the Mac).
- `Bot(platform=...)` mismatch raises (fake game).
- The parity tests from section 5.
- `test_steam.py` keeps passing on Windows.

## Steps
1. [x] `calib.PLATFORM` resolver + `describe()`; `platform.py` picks by it with the mismatch error. Tests.
2. [x] `bloons/cli.py` `pre_parse` + wire `--platform` into every tool and `watch_state.py`. Tests.
3. [x] `Bot(platform=)` switch (lazy harness import) + `calib.path(write=True)`. Tests with a fake game.
4. [x] Mac `Game` parity: `relaunch` (`open -a`, app name in `ui.mac.json` window.app), `read_number` (Vision OCR), `keep_awake` (`caffeinate -w <pid>`); `tools/stop.sh`; AST parity test. (`width/height` turned out to be `Frame` attributes on both, not a gap.)
5. [x] Data parity: test resolves every `section.key` UI point the code names on both sets. Only 3 are Steam-only (`menus.victory_home`, `monkeys_menu.open`, `xp_screen.back`), listed in `ui.mac.json` `_meta.steam_only`. `tools/farm.py` refuses non-win.
6. [ ] **Live on the Mac (needs Dom with BTD6+ open):** `scripts/smoke_platform.py`, `scripts/watch_state.py --platform mac`, then `tools/drive.py serve --platform mac --difficulty easy` and one full Monkey Meadow game. Re-calibrate anything a BTD6+ update has moved (`ui.mac.json` is stamped 57.4).
7. [ ] Calibrate the missing in-game Mac UI/templates (victory_home, popup_ok, level_up) from step 6 screenshots.
8. [ ] Docs: CLAUDE.md "Project rules" (the flag), START-HERE run commands for both modes, RESEARCH.md for any BTD6+ changes found, handoff.

Steps 1-5 are offline and can be done on either machine. Steps 6-7 need the Mac with the game open.

## Decisions for Dom
- **Flag name:** `--platform mac|win`, env `BLOONS_PLATFORM`. (`--game btd6plus|steam` would also work. The data is really split by *game build*, not OS.)
- **Steam BTD6 on macOS** would be a third combination (Mac backend + Steam calibration at the Mac's resolution). It's out of scope here, but the two-axis resolver (backend from `PLATFORM`, data from `NAME`) leaves room for it.
- **XP farm on Mac:** refuse for now (recommended), or calibrate the monkeys menu / XP screens on BTD6+ as a later step.
