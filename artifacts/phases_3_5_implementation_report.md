# Phases 3–5 implementation report

## Phase 3 — instrumentation

Implemented evaluation-only telemetry in `scripts/evaluation_metrics.py` and
integrated it through `harness.run_matches_instrumented`. Reports now include
cash, order requests/execution-cap truncation, daily hires and active hands,
per-slot work/move/pass share, planting/harvest actions, sale quantities and
observed prices, land requests, asset losses, risk tile-turns, unused land,
ending assets/inventory, opponent exposure, status, and bankruptcy indicators.

`harness.compare_paired` now also reports the paired range, approximate 95%
confidence interval, and win/tie/loss counts.

`scripts/replay_baselines.py` produced
`artifacts/replay_operational_baselines.json` from 2,532 leaderboard
player-episodes. This corpus is explicitly marked observational/diagnostic.

## Phase 4 — gradual hiring ramp

Added independently selectable `fixed`, `calendar`, `workload`, and
`cash_workload` target functions, conservative ceilings, feed/cash reserve
protection, marginal Fibonacci cost, remaining-season payback, and actual-hand
assignment behavior. The source-of-truth default remains `workload` pending
promotion evidence.

The screen is configured in
`ab_rounds/round30_phase4_hiring_screen.json`. It includes the current workload
control, a replay-shaped calendar ramp, workload ceiling 7, and cash/workload
ceilings 8 and 9.

## Phase 5 — livestock specialists

Added stable, contiguous daily livestock zones for one- and two-specialist
candidates. Normal tasks stay with their owner; irreversible-loss emergencies
may cross zone boundaries. Actual hand counts determine whether specialist
slots exist, so low-hire states fall back safely. Animal counts and land layout
are unchanged.

The candidate sequence is configured in
`ab_rounds/round31_phase5_specialists_screen.json`.

## Verification status

Focused unit, build, watcher-contract, syntax, action-contract, and
last-callable tests pass. `scripts/main.py` was regenerated from the editable
source without a watcher model.

Gameplay screening is not complete. The public `kaggle-environments==1.18.0`
package does not include the competition's custom `kaggriculture` environment
(the checked-in replay identifies its private module version as `1.32.7`). The
round runner is ready in `scripts/run_phase_experiments.py`, but promotion or
rejection would be unsupported until that runtime is restored.
