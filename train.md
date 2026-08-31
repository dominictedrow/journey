# Getting the watcher SVM fully online

## Current state

The phase-10/11 infrastructure is ready, but no SVM output currently qualifies
for training or export. `artifacts/watcher_readiness_report.json` is the
authoritative quota audit.

The checked-in/local prepared corpus is obsolete `counterfactual-v1` data. The
runtime feature schema is `counterfactual-v2`, expansion has since been split
into `land_expansion` and `labor_expansion`, and phase 10 now requires multiple
usable focal families. Old rows are retained as historical evidence but are
rejected rather than migrated or relabeled.

- The prepared corpus contains 540 attempted pairs and 444 accepted rows.
- Every prepared row has an explicit `strategy_family`, deterministic
  seed-group-safe `split`, and positive finite `sample_weight`.
- Seed groups use a deterministic 70/15/15 train/validation/test allocation.
- The generator supports targeted output generation and actionable checkpoint
  scheduling.
- The trainer validates schema, 129-feature order, finite values and weights,
  accepted rows, labels, quotas, and seed isolation.
- All 36 watcher-related tests pass.
- No `scripts/watcher_model.py` has been promoted, so runtime remains on rules.
- Seed groups now include the focal-family identity and the trainer rejects a
  group crossing either splits or families.
- Replay-guided checkpoint days can be supplied with `--checkpoint-days`.
- Generation enforces the 2,000-attempt cap per targeted output.
- Margin scale is calculated from signed training decisions only.

`competitive_expansion` is the only output that meets the overall data quota.
It failed the model gate:

- validation balanced accuracy: `0.420`;
- untouched-test balanced accuracy: `0.474`;
- untouched-test majority baseline: `0.684`.

The other 15 observed outputs lack the required accepted and signed-label
coverage. Generated datasets and reports are ignored local artifacts and must
not be committed.

## Remaining work

### 1. Make interventions causally valid

Strengthen acceptance beyond checking that a submitted action changed. Confirm
that each intervention is legal, affordable, executes in the environment, and
changes the intended state.

- Crop additions need an affordable purchase and applicable future crop slot.
- Crop removals and redirects need a matching future seed purchase.
- Animal additions need an affordable purchase and available structure.
- Animal deferrals need a matching future animal purchase.
- Hand changes need a valid hire or removal opportunity.
- Land changes need an affordable, actionable purchase checkpoint.
- Premium exits or retention need current or planned product exposure.

Reject a pair when the action is merely emitted but cannot execute or affect
the intended decision. Record execution evidence in `intervention_audit`.

### 2. Rework the competitive-expansion target

More examples alone are unlikely to repair its current performance. The target
combines land and hand interventions that may represent different decisions.
Evaluate splitting it into coherent outputs such as `land_expansion` and
`labor_expansion`. Retain a combined output only if evidence shows a shared,
learnable decision boundary. Regenerate its data after semantics are finalized.

### 3. Add multiple focal strategy families

The current corpus uses only `strategy_follower`. Generate compatible states
from multiple genuine focal strategy variants so leave-one-family-out results
are meaningful. `strategy_family` identifies the focal strategy that produced
the state, not its opponent, and remains separate from the seed-group split.

### 4. Generate targeted class coverage

For every modeled output, generate deterministic targeted batches until it has:

- at least 150 accepted rows;
- at least 40 positive labels;
- at least 40 negative labels.

Stop after 2,000 attempted pairs for an output if these requirements remain
unavailable. Leave that output on deterministic rules.

After every batch report attempts, acceptance, rejection reasons, all three
label counts, signed counts per split, and counts by focal strategy family.
Size later batches from measured acceptance and signed-label rates rather than
continuing a uniform mixture.

```bash
PYTHONPATH=scripts /tmp/agric-watcher-venv/bin/python \
  scripts/counterfactual_dataset.py \
  --output data/counterfactual_dataset.prepared.jsonl \
  --append \
  --pairs <batch-size> \
  --seed-base <unused-first-seed> \
  --workers <worker-count> \
  --target-output <output-id> \
  --strategy-family <focal-family>
```

Never reuse a seed group with a different split or strategy family. Do not run
multiple generator commands concurrently; each already uses multiprocessing.

### 5. Verify final corpus readiness

Before fitting, require:

- expected schema version and hash;
- exact feature order and 129 finite feature values;
- explicit positive finite sample weights;
- labels restricted to `-1`, `0`, and `1`;
- accepted, execution-verified causal pairs only;
- every seed group isolated to one split;
- the overall 150/40/40 quota for each eligible output;
- both signed classes in train, validation, and test;
- at least five examples of each signed class in every split;
- multiple usable focal strategy families.

Increase the per-split threshold if the final corpus supports it. The trainer's
default of one is only suitable for smoke tests.

### 6. Train without leakage

For each eligible output:

1. Fit `StandardScaler` on the train u val pool. Neutral rows may be included in
   scaler fitting; they are excluded from binary fitting and evaluation.
2. Renormalize fit-pool weights to mean `1.0` when using the documented mode.
3. Select `C` by mean balanced accuracy under grouped stratified k-fold CV
   (group = seed group) over train u val, breaking ties with smaller `C`.
4. Refit one `LinearSVC` on the full signed train u val pool at the chosen `C`;
   calculate margin scale from that model's training decision margins only.
5. Run the pooled leave-one-family-out diagnostic with train and validation rows
   only.
6. Evaluate exactly once on the untouched test split (with a bootstrap CI), then
   exactly once on the untouched `confirm` split.

Never derive scaling, weights, hyperparameters, thresholds, or revisions from
test or confirm outcomes.

All quotas and thresholds come from `scripts/watcher_config.py`. `--min-per-class`
**defaults to 5**; `--min-per-class 1` is a smoke-test-only setting and prints a
loud warning -- a model trained that way must never be promoted.

```bash
PYTHONPATH=scripts /tmp/agric-watcher-venv/bin/python \
  scripts/train_watcher.py \
  --rows data/counterfactual_dataset.prepared.jsonl \
  --output /tmp/watcher_model_candidate.py \
  --report watcher_training_report
  # --min-per-class / --min-accepted / --min-signed / --min-families all
  # default to watcher_config; pass them only to tighten, never to loosen.
```

Use a temporary model output while evaluating so a rejected candidate cannot
replace the active runtime model.

### 7. Apply the offline acceptance gate

`train_watcher.py` exports an output only when all four gates pass (see
`MD files/tune.md` for the rationale and `scripts/watcher_config.py` for values):

- grouped-CV mean balanced accuracy `>= ACCEPT_MIN_CV_MEAN`;
- bootstrap test balanced-accuracy CI lower bound `>= ACCEPT_MIN_BALANCED_ACC_CI_LOWER`;
- pooled leave-one-family-out balanced accuracy `>= ACCEPT_MIN_LOFO_POOLED`;
- untouched `confirm` split within the test CI width of the test point estimate.

Review the JSON and Markdown reports before export. Rejected and unavailable
outputs remain absent so runtime inference falls back to deterministic rules.

### 8. Export and validate the pure-Python model

When at least one output passes, export only accepted records to
`scripts/watcher_model.py`. Verify:

- pure-Python scores match scikit-learn decision scores;
- scores are finite and clipped to `[-1, 1]`;
- feature order and schema hash match runtime extraction;
- malformed features, missing outputs, schema mismatch, and non-finite values
  fall back safely;
- runtime imports neither NumPy nor scikit-learn.

```bash
PYTHONPATH=scripts /tmp/agric-watcher-venv/bin/python \
  -m unittest discover -s scripts -p 'test_*.py'

python3.11 -m py_compile \
  scripts/shared_features.py \
  scripts/watcher_model.py \
  scripts/strategy_follower.py \
  scripts/main.py
```

### 9. Enable the backend and build the submission

Only after export validation:

- set `WATCHER_BACKEND = "linear_svm"` in the editable strategy;
- preserve deterministic fallback for missing or rejected outputs;
- update the final artifact through the established source-to-submission flow;
- keep `agent(obs)` as the last top-level callable;
- verify the action shape and absence of NumPy/scikit-learn runtime imports.

Do not enable the backend merely because an export exists. At least one output
must have passed the complete offline gate.

### 10. Run paired gameplay promotion testing

Compare the SVM watcher with the preceding rules watcher using identical seeds
and alternating player order.

- Use at least 20 episodes per side initially.
- Use at least 30 per opponent for a promotion-quality conclusion.
- Test representative competitive opponents.
- Report paired money difference, sample size, variance or range, safety,
  regressions, and operational failures.

Promote only if the paired-money, safety, regression, and operational gates
pass. If no output passes offline acceptance, skip gameplay testing because
there is no valid SVM candidate.

## Environment

Use Python 3.11 with development-only `numpy`, `scikit-learn`, and
`kaggle-environments`.

AppleDouble `._*` metadata on this external volume can break pip environments
created inside the repository. A disposable environment under `/tmp` works.
Never package a development environment or scientific dependencies with the
Kaggle submission.

## Definition of fully online

The watcher SVM is fully online only when:

1. At least one coherent output has execution-verified, quota-complete,
   seed-isolated data across multiple focal strategy families.
2. It passes the untouched-test acceptance gate.
3. Its pure-Python export passes score-parity and fallback tests.
4. The editable strategy and final submission enable it safely.
5. Paired gameplay testing passes all promotion gates.

Until then, deterministic rules remain active for every rejected, unavailable,
or missing output.
