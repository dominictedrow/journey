# Phase 6 implementation report

## Implemented behavior

The editable strategy now supports an explicitly gated fixed crop-transition
schedule. Permanent melon remains the default.

- Melon seed purchasing and planting have independent cutoff days.
- Existing melon is retained until the engine makes the tile vacant; the
  transition code never digs a healthy crop.
- Vacated slots can remain empty or receive a bounded strawberry allocation.
- Strawberry planting is prohibited after day 19, ensuring its first scheduled
  yield can occur before the season ends.
- A bounded late-wheat stage can claim vacant slots. Wheat planting is
  prohibited after day 25, ensuring peak maturity before day-29 liquidation.
- The schedule operates on the existing 19-slot crop footprint and does not
  change land, herd, or crew settings.

Phase-3 telemetry now records planting, harvesting, and crop-exit days by crop,
and distinguishes observed weed formation from ordinary harvested crop exits.

## Candidate sequence

1. `round32_phase6a_melon_exit.json` isolates melon seed-purchase exits at days
   9, 11, and 13 against permanent melon.
2. `round33_phase6b_strawberry.json` provisionally tests 3, 6, and 9 strawberry
   slots after a day-11 exit.
3. `round34_phase6c_late_wheat.json` provisionally tests late wheat timing and
   scale after the six-strawberry candidate.

The Stage-B/C predecessor selections are deliberately provisional. They are
not enabled in `strategy_follower.py` or `main.py` unless prior stages pass the
required paired promotion gates.

## Verification and evaluation status

All 12 generated candidates compile. Eleven focused source/build/contract tests
pass, the generated submission passes syntax and action-contract validation,
and `agent(obs)` remains its last callable.

Seeded smoke, paired, and confirmation games remain unavailable because the
public Kaggle package does not ship the competition's custom
`kaggriculture` module 1.32.7. Accordingly, all transition defaults remain off
and the result records make no performance claim.
