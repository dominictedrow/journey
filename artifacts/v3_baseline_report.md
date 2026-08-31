# v3 baseline and build report

Date: 2026-08-21

## Baseline identity

The editable control is `scripts/strategy_follower.py`. The submission is
generated from it by `scripts/build_submission.py`; experimental edits must
not be made directly in `scripts/main.py`.

| Setting | v3 control |
|---|---|
| Animal mix | `COW=11`, `SHEEP=6` |
| Livestock hands | `6` |
| Melon hands/tiles | `2` / `19` |
| Wheat hands/tiles | `0` / `0` |
| Carrot hands/tiles | `0` / `0` |
| Melon exit | disabled (`999`) |
| Land target | `2` quadrants |
| Sell threshold | `0.4` of base price |
| Deadline planner | enabled |
| Adaptive targets | enabled |
| Town watcher rules | enabled |
| Opponent watcher rules | enabled |
| Watcher backend | `rules` |
| Embedded watcher model | none |

## Expanded-artifact differences

The archived expanded artifact had three melon hands, 45 melon tiles, four
wheat tiles and one wheat hand, four carrot tiles and one carrot hand, a
day-19 melon exit, and a three-quadrant land target. These are classified as
unvalidated experiments and are absent from the v3 control.

The following behavior in the editable control is retained as intentional
promoted behavior: deadline planning, adaptive targets, town watcher rules,
opponent watcher rules, and the constants in the table above.

The following are retained as correctness-only behavior: owned-cell routing
and locked-spawn recovery, feed purchasing before discretionary investment,
market-order-cap enforcement, end-season liquidation/action safety, and
per-signal deterministic watcher fallback.

Differences from the checked-in historical editable source at `HEAD` are
correctness-only: the already-present dirty-worktree watcher classifier-name
correction uses `land_expansion` and `labor_expansion` rather than the obsolete
`competitive_expansion` name; phase 1 also adds an explicit ten-order market
cap and makes day-29 sales bypass the price gate and feed reserve. The watcher
name correction does not change rules-backend actions.

The generated artifact removes the demo entry point and optional development
imports. Those are artifact-generation differences and do not change rules
backend actions.

## Verification

- The build is byte-deterministic across repeated runs.
- `agent(obs)` is the last top-level callable.
- No scientific runtime import remains.
- The build-time action-contract check passes a checked-in replay observation.
- Exact source/artifact action parity passes on 5,000 early-, mid-, and
  late-season leaderboard replay observations, including their naturally
  occurring low-cash, locked-land, crop-maturity, and liquidation states.
- Every checked action has the required three fields and at most ten market
  orders.

Local short/full simulation remains unavailable in the current Python 3.11
environment because it does not contain `kaggle_environments`; the system
Python installation is an older build that does not register Kaggriculture.
Consequently no new paired-performance claim is made by phases 0–2.
