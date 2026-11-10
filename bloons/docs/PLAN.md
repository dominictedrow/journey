# PLAN — Bloons TD 6+ RL Bot

Progress is tracked by the checkboxes here. `START-HERE.md` points at the active step. Rationale and sources are in `RESEARCH.md`.

## Goal
An RL agent that plays **Bloons TD 6+ (Apple Arcade) on macOS** through a pixel-based harness. Its actions and observations are plain Python methods that a person can drive by hand and wire into an RL loop without trouble. The first targets are 4 beginner maps on Easy, Standard mode.

## Architecture
```
Bloons bot/
  CLAUDE.md                 # session protocol + project rules
  docs/START-HERE.md  PLAN.md  RESEARCH.md  handoffs/
  bloons/
    platform_mac.py         # find window, capture, click/drag/key (Quartz). Only OS-specific file.
    vision.py               # OCR, template match, bloon detector, popup detector
    data/towers.json heroes.json rounds.json ui.json
    data/maps/<map>.json    # UI coords, placeable masks, path polyline, obstacles
    data/templates/*.png    # buttons, popups, victory/defeat, upgrade-state crops
    data/fixtures/*.png     # saved screenshots for offline tests
    harness.py              # BloonsHarness: actions + observe() -> GameState
    env.py                  # Gymnasium env + action mask
  tools/calibrate.py        # subcommands: ui | mask <map> | path <map> | templates
  tools/play.py             # REPL to drive the harness by hand
  train.py                  # MaskablePPO
  tests/test_offline.py     # vision/data tests on fixtures (no game needed)
```

## Harness API (`bloons/harness.py`)
Every action returns `ActionResult(ok: bool, reason: str)` and doesn't raise on game-side failures. Coordinates are normalized `0–1` over the game window. Towers get a stable integer `tid` at placement. The harness keeps a **tower registry** (kind, xy, tiers, targeting, money spent).

### Actions
| Requirement | Method |
|---|---|
| Buy, select and place a monkey | `place_tower(kind, x, y) -> tid` (checks the mask and money first) |
| Buy an upgrade | `upgrade(tid, path)` with `path ∈ {0,1,2}` |
| Place a hero / upgrade a hero | `place_hero(x, y)`; `buy_hero_level(tid)` |
| Sell | `sell(tid)` |
| Use abilities | `use_ability(slot)`; `use_ability_of(tid)`; `place_ability_target(slot, x, y)` |
| Configure abilities and targeting | `set_targeting(tid, mode)`; `set_target_point(tid, x, y)`; `set_flight_path(tid, pattern)`; `set_patrol_points(tid, points)` |
| Remove terrain | `remove_obstacle(obstacle_id)` |
| Start next round | `start_round()` (also switches fast forward on; FF and Auto Start are not actions, removed 2026-11-09 for episode speed) |
| Handle prompts | `handle_prompts() -> list[str]` (runs automatically before every action) |
| Episode control | `new_game(map, difficulty, mode, hero)`; `restart()`; `go_home()` |

### Observation: `observe() -> GameState` (+ `.to_vector()`)
| Requirement | Field | Source |
|---|---|---|
| Money / health / round | `money`, `lives`, `round`, `max_round` | HUD OCR + sanity filter |
| Round ended | `round_active`, `round_ended` (edge) | start/FF button state |
| Won / lost | `outcome ∈ {None,"win","loss"}` | victory/defeat templates |
| Where monkeys can go | `placeable_mask[footprint]` | map calibration minus placed towers |
| Path | `path` polyline, `path_grid` | map calibration |
| Bloons + types | `bloons: [(x, y, type_guess)]`, `round_composition` | vision detector + `rounds.json` |
| Towers + ranges | `towers: [TowerState(tid, kind, xy, tiers, range_px, targeting, sell_value)]` | registry + `towers.json` (+ circle measurement) |
| Available upgrades + costs | `upgrade_options[tid][path] = (cost, affordable, locked)` | table × difficulty, crosspath rule, panel OCR |
| Monkey / tower abilities | `abilities: [(slot, tid, ready, cooldown_frac)]` | ability bar icons + cooldown overlay |
| Hero | `hero: (tid, level, xp_frac)` | panel OCR |

## RL layer (`bloons/env.py`)
**Owned by Dom, who builds the RL model himself.** Claude doesn't write the env, rewards, model or training code; the notes below are a starting sketch for Dom, not a spec. Claude's job ends at a harness that's clean to wrap (actions + `observe()`).
- **Step:** one macro action at round start, plus every N seconds during a round. After each action the env runs `handle_prompts()` and returns `observe()`.
- **Actions:** a flat `Discrete` table: `NOOP | START_ROUND | PLACE(kind, cell) | UPGRADE(slot, path) | SELL(slot) | ABILITY(slot) | TARGETING(slot, mode)`. Cells come from a coarse grid (e.g. 24×16), with a fixed number of tower slots (~20). `action_masks()` removes unaffordable, unplaceable and locked actions.
- **Observation:** a vector (money, lives, round, upcoming composition, slot table, map one-hot) + a small grid image (path, towers, bloon density).
- **Reward:** + per round survived, − per life lost, − small for a failed action, ± large at the end.
- **Training:** a scripted baseline first, then MaskablePPO. Optional behavior-cloning warm start from scripted or manual logs.

---

## Phase 0 — Scaffold + Mac feasibility
- [x] 0.1 Write CLAUDE.md, START-HERE.md, PLAN.md, RESEARCH.md, handoffs/
- [x] 0.2 Mac setup: Python 3.11+, deps, Screen Recording + Accessibility permissions. Throwaway script: find the window, capture a frame, send a synthetic click + `Q` key. **Gate: if input is ignored, stop and re-plan (HID fallback).**
- [x] 0.3 Test every hotkey in RESEARCH.md and fill in the status column
- [x] 0.4 Measure capture FPS and OCR latency. Fix the window size used for all calibration and record it in START-HERE
- [x] 0.5 `git init` the repo structure, plus `requirements.txt`

## Phase 1 — Platform + vision basics
- [x] 1.1 `platform_mac.py`: `find_window()`, `grab()`, `click(x,y)`, `drag(a,b)`, `key(k)` (normalized coords)
- [x] 1.2 Capture fixtures: HUD at various values, every popup type, victory, defeat, the upgrade panel → `data/fixtures/`
- [x] 1.3 HUD OCR: money, lives, round (digit whitelist + sanity filter) + offline tests
- [x] 1.4 Template library + `match(name)`: start / FF / FF-active buttons, popups, victory/defeat
- [x] 1.5 Round-state + outcome detection + offline tests (live-checked 2026-11-04: one Easy game, loss at round 30)

## Phase 2 — Static game data
- [x] 2.1 `towers.json`: every tower's base cost, upgrade costs per path/tier, base range + range modifiers, abilities, hotkey, footprint class, land/water — *all from the wiki via `scripts/fetch_wiki_data.py`. Gaps to measure in game: Banana Farm and Skywarden stats, Monkey Village tiers other than 100/500 and its abilities; crosspath ranges are estimates (5.5)*
- [x] 2.2 `heroes.json`: cost, level-up costs, abilities by level — *XP table + $1/XP cash rule and level descriptions for all 18; abilities/range/footprint only for the 6 heroes with a wiki dump (rest: read the ability bar, 5.4)*
- [x] 2.3 `rounds.json`: Standard rounds 1–140 (types, counts, properties) — *plus spawn times, RBE, layers, pop cash, end bonus; RBE/cash/duration match the wiki's rendered table for all 140*
- [x] 2.4 Cost helper `cost(base, difficulty)` + crosspath rule `upgrade_allowed(tiers, path)` + tests
- [x] 2.5 Stamp the game version. Spot-check 10 costs against the in-game panel — *33 prices checked; fixed `cost()` rounding; Monkey Knowledge effects in `account.json`; Dart/Tack ranges confirmed (7.95 px/unit)*

## Phase 3 — Map calibration (Monkey Meadow, Tree Stump, Town Center, Logs)
- [x] 3.1 `ui.json`: shared UI coords (tower shop, upgrade panel buttons, sell, targeting, ability bar, settings/auto-start, FF) — *plus hero panel, pause, menus, defeat/restart buttons; `bloons/ui.py` accessors; shop scrolls (HID wheel), two scroll layouts*
- [x] 3.2 `calibrate.py mask <map>`: grid sweep while holding a tower, reading the red tint → placeable masks per footprint (land/water) — *Monkey Meadow: 6 land classes (r6, r7, r8, r11, heli, ace), outline-then-ring classifier; no water on this map; `placement.placeable()` + `calibrate.py show`. At ~50 min per map the sweep is now optional: it seeds the learned-spots memory (`calibrate.py spots`), and placement runs the layered check in `docs/PLACEMENT-PLAN.md`*
- [x] 3.3 `calibrate.py path <map>`: frame-difference bloons on round 1 → polyline, then a hand-correct click tool — *new-red-pixel detection (cloud shadows/leaves/bat break plain differencing), lead-bloon centre line, sprite offset 4 units, half-width 0.25 fitted to the sweep; it's placement layer 1 (`placement.PathKeepout`) and explains 44–56 % of blocked cells. `path-edit` written, not yet used by hand. Monkey Meadow done*
- [x] 3.4 Obstacles: click points + IDs per map — *none of the 4 maps has a removable obstacle (every candidate clicked, no panel); `obstacles: []` in each map file. 4.7 `remove_obstacle` has nothing to act on for now*
- [x] 3.5 Calibrate all 4 maps and save `data/maps/<map>.json` — *each has `scale`, `path` (lead-bloon line; Tree Stump / Town Center / Logs from Standard round 1, since their sandbox is locked; half-width 0.5 default) and `obstacles`; map-select pages in `ui.json`. Placeable masks only on Meadow (the sweep is optional; memory + tint do the rest)*

## Phase 4 — Harness actions
- [x] 4.1 `BloonsHarness` skeleton, `ActionResult`, tower registry, hotkey-or-click helper — *`bloons/harness.py`: data + map keep-out + spots memory, registry (`register` / `record_upgrade` / `unregister`, hero tid 'hero'), `guard_state` / `place_context` / `available_actions`, `press` (hotkey from `ui.json` `hotkeys` / towers.json, else click), `select_shop` (scrolls for Desperado / Mermonkey / Skywarden; `Game.scroll`). Checked live on Meadow. `ActionResult` (`bloons/result.py`) and the guards (`bloons/guards.py`: `can_*` + `available_actions`) exist; every action calls its guard first and returns the refusal without touching the game*
- [x] 4.2 `place_tower`, `place_hero`, `sell` — *done in `bloons/harness.py`, checked live (Meadow Standard + Sandbox): Dart placed, tree-corner spot refused by tint then by memory, hero placed, both sold at 70 %. Plan was: guard (`can_buy_tower` / `can_place_hero` / `can_sell`) → `placement.can_place` → hold, one `held_tint` read; red = cancel + `record_placement(False)` + `ActionResult(False, "blocked")`; else click and confirm money dropped + hold ended (if not, record blocked)*
- [x] 4.3 `upgrade`, `buy_hero_level` — *done: select → panel side (`state.read_screen`) → path key / level-up button → close the panel → HUD money confirms. Live in Meadow Sandbox: 4 upgrades to 3-0-1 + 4 hero levels, net money exactly as priced. Plan was: guard first (`can_upgrade`: crosspath, tier ≤ 5, one tier 5 per path per type, money; `can_buy_hero_level`)*
- [x] 4.4 `start_round`, `set_fast_forward`, `set_auto_start` — *2026-11-09: `set_fast_forward` / `set_auto_start` removed (handoff 048); `start_round` keeps FF on, Auto Start stays whatever the game setting is*
- [x] 4.5 `use_ability`, `use_ability_of`, `place_ability_target` — *guard first (`can_use_ability`: ready flag, round-only abilities)*
- [x] 4.6 `set_targeting`, `set_target_point`, `set_flight_path`, `set_patrol_points`
- [ ] 4.7 `remove_obstacle`
- [x] 4.8 `handle_prompts` (template → default click, overridable) — *4.7 skipped for now: no beginner map has an obstacle*
- [x] 4.9 `new_game`, `restart`, `go_home`
- [x] 4.10 `tools/play.py` REPL. Drive every action by hand on Monkey Meadow — *done in Sandbox (handoff 038); `observe()` from the Phase 4 gate is 5.1, add it to the REPL then*

## Phase 5 — Full observation
- [x] 5.1 `GameState` dataclass + `observe()` (HUD, round state, outcome, registry, masks, path) — *`bloons/observe.py` + `BloonsHarness.observe()`; also upgrade options (price table), ability estimates, `rounds_done` / `next_round` / `round_composition`*
- [x] 5.2 Bloon detector (HSV classes first; small YOLO only if HSV falls short) + `round_composition` — *`bloons/detect.py`, HSV was enough; stored background for Monkey Meadow only (the other 3 maps need one: Sandbox is locked there)*
- [x] 5.3 `upgrade_options` (table + rules + panel OCR verification) — *`bloons/panel.py` reads row state / price / tier pips; `h.check_upgrades(tid)`, `upgrade()` reads the row before buying; `UpgradeOption.checked`. Live sweep: 17 tower kinds match the table on every tier; Sniper / Spike / Super / Tack / Wizard / Skywarden not swept yet*
- [x] 5.4 Ability bar reading (ready + cooldown fraction) — *`bloons/abilitybar.py`: the red-orange cooldown ring's arc (it shrinks clockwise to 12 o'clock) → remaining fraction, ±0.02 vs a fitted line; `BarFilter` holds rings that vanish mid-cooldown; `h.sync_abilities` resets the estimates from every read (observe, use_ability, use_ability_of); uses confirmed by the full ring. Keys 1–3 confirmed*
- [x] 5.5 Range: table value + circle-measure verification — *`bloons/rangecircle.py` + `h.measure_range(tid)` (→ `TowerState.range_measured`, `h.range_mismatches`); 29 live reads within 0.07 units of the table except Super Monkey path 2 (+3). Range circles are ellipses (ground seen at a tilt, y × 0.866): `range_norm` fixed; placement's y unit is still the old one*
- [x] 5.6 `GameState.to_vector()` + grid image — *`bloons/obsvec.py`, layout in `data/obs.json`: 1,200-float vector (global, map/difficulty/mode one-hots, next round's composition, bloons seen + lead progress, 20 tower slots × 53, hero, 8 ability slots) + 8 × 16 × 24 grid (path, 4 placeable classes, towers, ranges, bloons); towers keep a stable slot (`h.slots`); `obsvec.cell_centre` for grid actions*

## Phase 6 — Scripted baseline
- [ ] 6.1 Hand-written strategy using only the harness API — *Dom writes it (2026-11-06)*
- [ ] 6.2 Beat Easy on all 4 maps with outcome == "win" detected. Log (obs, action) pairs for BC

## Phase 7 — RL (Dom's)
- [ ] 7.1 `env.py` (Gymnasium) + `action_masks()`. `check_env` passes
- [ ] 7.2 `train.py` with MaskablePPO, checkpoints, TensorBoard logging, multi-map sampling
- [ ] 7.3 Optional BC warm start from the Phase 6 logs
- [ ] 7.4 Eval script: win rate + mean round reached per map

## Phase 8 — Expand
- [ ] 8.1 More maps, Medium/Hard, other modes
- [ ] 8.2 Hero choice and late-game strategy (rounds 40+, MOAB-class)

---

## Verification
- `pytest tests/test_offline.py`: OCR, templates and data rules on fixtures (Windows or Mac).
- Phase 4 gate, in `tools/play.py`: `place_tower("dart", .4, .5)`, `upgrade(0, 0)`, `start_round()`, `observe()`. Each must match the screen.
- Phase 6 gate: the scripted baseline wins Easy on every calibrated map.
- Phase 7 gate: `gymnasium.utils.env_checker.check_env` passes, and a short PPO run steps without harness errors.

## Risks
- BTD6+ ignores synthetic input → Phase 0 gate. Fallback: HID-level injection.
- Game updates move UI or change costs → everything lives in JSON, data is version-stamped, OCR cross-checks flag drift.
- Training is bound by real time → macro actions, masking, a scripted warm start, Easy first.
