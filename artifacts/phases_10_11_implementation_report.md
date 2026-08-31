# Phases 10–11 implementation report

## Phase 10 — causal watcher data

The generator now maintains family-qualified seed groups, rejects reuse on
append, accepts replay-guided checkpoint days, and enforces the 2,000-attempt
cap for a targeted output. Interventions continue to require identical prior
state/actions, legality, affordability, application, next-state execution
evidence, and safety-aware final-money labels.

The readiness audit enforces the complete per-output gate: 150 accepted rows,
40 positive and 40 negative labels, five of each signed class in every split,
two focal families that each contain both signed classes, exact schema and
feature order, finite values, execution evidence, and seed isolation across
both splits and families.

The existing prepared dataset is historical `counterfactual-v1` data and uses
the removed combined `competitive_expansion` output. It is not valid input for
the v2 watcher. No current output is quota-complete, so all signals remain on
deterministic rules. Fresh paired generation cannot run in this environment
because the competition's custom Kaggriculture runtime is unavailable.

## Phase 11 — offline training and acceptance

The trainer fits scaling on training rows only, excludes neutral rows from
binary fitting/evaluation, renormalizes signed training weights, chooses C by
validation balanced accuracy with smaller-C tie breaking, calibrates margin
from signed training decisions, runs leave-one-family-out diagnostics without
test rows, and applies the untouched-test 0.60 / +0.10 acceptance gate.

Seed groups crossing focal families are now rejected, and the family quota
requires at least two usable families containing both signed classes. Because
phase 10 produced no eligible output, phase 11 correctly skipped fitting and
did not create or enable `scripts/watcher_model.py`.

## Result

`artifacts/watcher_readiness_report.json` records zero eligible outputs. The
submission remains `WATCHER_BACKEND = "rules"`; missing/rejected output fallback
is unchanged.
