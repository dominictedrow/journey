# Phases 7–9 implementation report

## Phase 7 — product-specific sale scheduling

The editable strategy now contains a gated `product_specific` sale mode while
the promoted default remains `legacy`.

- Per-product price thresholds separate staples from premium goods.
- Wheat and fertilizer support small routine batches; every product has an
  independently configurable batch cap.
- Inventory at or above 90 units bypasses normal hold thresholds.
- Hold thresholds decay from day 22 to liquidation.
- Day 29 sells every eligible unit regardless of threshold or batch cap.
- Routine sales have a three-order budget and remain behind critical purchases
  in the market-order priority; final liquidation remains first.

Rounds 35 and 36 isolate batch size and thresholds. They are compiled and
ready, but remain unpromoted because the custom Kaggriculture runtime is not
installed, so paired revenue and safety gates cannot run.

## Phase 8 — conditional third land

The strategy now supports a gated `conditional` land mode. A third quadrant
requires all of: configured utilization, planner capacity, operating reserve,
enough repayment days, affordability, and a concrete layout extending beyond
the currently unlocked cells.

The current promoted footprint contains 17 animal and 19 crop slots: 36 of 50
cells, at most 72% utilization. It supplies no third-quadrant layout. Phase 8
therefore stops at its prerequisite gate and retains two quadrants. Round 37 is
future-ready for the point when a promoted layout actually needs those cells.

## Phase 9 — replay-derived challengers

`scripts/replay_challengers.py` creates behavioral fingerprints from hire,
land, animal, crop, movement, sale-cadence, and liquidation observations. On
the current top-10 corpus it finds three families:

1. an eight-name family represented by `カワシギ`;
2. `MiMi`;
3. `Subramanya N`.

Each representative has an evaluation-only nearest-state action challenger.
Held-out replay validation is recorded in
`artifacts/replay_challenger_report.json`. Exact action rates are 0.427, 0.326,
and 0.247 respectively. These reconstructions are not yet in the promotion
opponent registry: action similarity is too weak to call them validated, and
state-trajectory/gameplay validation needs the unavailable custom runtime.
Replay actions remain observational and are not used as causal labels.

## Verification

- The editable and generated strategy compile under Python 3.11.
- All phase-7/8 variants generate and compile.
- Focused strategy/build/watcher tests pass except suites requiring missing
  optional dependencies (`kaggle_environments` or NumPy).
- The built `scripts/main.py` keeps `agent(obs)` as its last callable and does
  not enable either unpromoted phase-7 or phase-8 behavior.
