# Plan: placement that works on every map + action guardrails

## Context
Step 3.2 built placeable masks by sweeping a held tower over about 4,400 grid points. That's about 8.5 min per tower size, so about 50 min per map. Dom wants something that scales to every map:
1. Limit placement using the bloon path, found by a per-map path diagnostic.
2. Use the game's own indicator of a valid spot.
3. Keep a fallback memory: when a placement fails, record the spot, block off the area around it, and don't retry it. Save valid and invalid spots to a file.
4. Guardrails: the harness refuses actions it knows are illegal, such as an upgrade the player can't afford, and says which actions are available.

Scope rule (memory `rl-model-is-doms`): all of this is harness. The env's action-mask vector, rewards and policy stay Dom's. The harness exposes plain `can_*` checks and an availability list, and Dom builds `action_masks()` from them.

**Can the path alone be enough?** No, not on its own. The path explains most blocked area, but not obstacles (rocks, buildings, water on other maps), the map edges, or sharp edges like the Meadow tree corner. So the path is the cheap first filter, and the game's held-tower tint is the exact check at the moment of placing.

## Design: 3 layers, checked in order by `can_place`

1. **Path keep-out (prior, any map, about 1 round to build).** Frame-difference round 1 to get a heat map of where bloons move. Threshold it to get the track band. Skeletonize the band to get the path polyline, which the observation needs anyway (PLAN 3.3). A spot is blocked if it's within band + the tower's footprint radius. Tunnels show no motion, which is correct, since towers can go over tunnels.
2. **Learned spots (memory file).** Every checked spot is stored per map and per footprint class as `ok` or `blocked`. A blocked result blocks a small disc around the spot (default 1 cell, about 3.7 units) so nearby retries are skipped. A later `ok` reading at an exact point overrides the disc. Spots blocked by one of our own towers are **not** saved to the file, because the harness can tell from its tower registry that the block is temporary.
3. **Held-tower tint (indicator, exact).** In `place_tower`, the harness holds the tower and moves to the spot. It reads `placement.held_tint()` once, which already exists (about 70 ms). If the tint is red, it cancels, records the spot as blocked and returns `ActionResult(False, "blocked")`. If the tint is ok, it clicks, then confirms that money dropped and the hold ended. If that confirmation fails, the spot is recorded as blocked anyway.

The full sweep (3.2) stays as an optional tool to pre-fill the learned-spots file. The Monkey Meadow masks already swept are imported as its starting data.

## Files
- **`bloons/placement.py`** (extend):
  - `PathKeepout` (band mask + polyline, `blocked(x, y, radius_units)`).
  - `SpotMemory` (load/save, `record(cls, x, y, ok)`, `lookup(cls, x, y) -> ok | blocked | unknown`, disc blocking).
  - `can_place(kind, x, y, ctx) -> ActionResult`, which runs the layers in order: registry overlap → keep-out → memory. Reuse `mask_classes`, `Grid`, `held_tint` and `rules.tower_shape`.
- **`bloons/data/maps/<map>.spots.json`** (new, written at runtime): `_meta.game_version` stamp, grid, and per class a list of `[x, y, ok]` points. It's kept separate from the hand-calibrated `<map>.json`, so runtime writes never touch calibration data.
- **`tools/calibrate.py path <map>`** (PLAN 3.3): capture round 1, build the heat map, then band + polyline → `<map>.json` under `path`. Plus `show` support for the band.
- **`bloons/guards.py`** (new, pure, portable). Each check returns `ActionResult(ok, reason)`:
  - `can_buy_tower(kind, money, ...)`: price from `rules.shop_price`.
  - `can_upgrade(tower, path, money, ...)`: `rules.upgrade_allowed` (crosspath, tier ≤ 5) + `rules.upgrade_price` vs money + one tier 5 per path per tower type.
  - `can_buy_hero_level`, `can_sell`, `can_use_ability` (ready flag).
  - `available_actions(state) -> list` of the currently legal actions with their costs, for Dom's mask.
- **Harness (Phase 4):** every action method calls its guard first and returns the refusal without touching the game. PLAN rows 4.2 / 4.3 / 5.3 get this added.
- **`docs/PLAN.md`:** rewrite 3.2 / 3.3 notes to describe the layered approach, add a guards line to 4.1, and add a RESEARCH note on why the path alone isn't enough.

## Order
1. Portable, now (no game needed): `SpotMemory`, `can_place` layers 1–2, `guards.py` + offline tests. Import the Meadow sweep into `monkey_meadow.spots.json`.
2. Game needed: `calibrate.py path monkey_meadow`. Compare the keep-out band against the swept `land_r6` mask (report the agreement %, to show how much the path alone explains).
3. Phase 4: wire guards + the tint check into `place_tower` / `upgrade`.

## Verification
- `pytest tests/test_offline.py`. New tests:
  - `SpotMemory` round trip + disc blocking + ok override.
  - The keep-out on a synthetic polyline.
  - `can_upgrade` refuses for low money / crosspath / tier 6 and allows the exact-money case.
  - `available_actions` on a fake state.
- Live (Mac, Monkey Meadow sandbox): run `calibrate.py path`. Then a short script tries 20 random spots via the tint check and confirms that blocked spots end up in `spots.json` and aren't probed again. Check against the `show` overlay.

## Status (2026-11-05)
- Order 1 done: `bloons/result.py`, `bloons/guards.py`, `bloons/pathmap.py`, `placement.PathKeepout` / `SpotMemory` / `can_place` / `record_placement`, `monkey_meadow.spots.json` seeded from the sweep.
- Order 2 done on Monkey Meadow, with changes from the design. Plain differencing is replaced by new-red-pixel detection, and skeletonizing by lead-bloon tracking. The keep-out is the centre line + a fitted half-width (0.25 units), not the band. Measured: the path alone explains 44–56 % of blocked cells, with ≤ 1 % false blocks (RESEARCH, 2026-11-05). The live tint loop worked: 16/16 probes matched the sweep, and pass 2 needed no probes.
- Order 3 waits for Phase 4 (PLAN 4.1–4.3 / 4.5 say how).
- Paths recorded on Tree Stump, Town Center and Logs too (Standard round 1, `--start space`). Red leaves and animated chevrons needed two `PathRecorder` fixes (RESEARCH, steps 3.4 / 3.5). These maps use the default half-width 0.5: no sweep to fit against.
