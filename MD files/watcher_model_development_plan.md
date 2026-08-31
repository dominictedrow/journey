# Watcher model development plan

## Outcome

Produce a nonempty, accepted `scripts/watcher_model.py` containing pure-Python
Linear SVM records, embed it into a standalone `scripts/main.py`, and promote it
only if it improves paired gameplay without violating the deterministic safety
gates in `main_v3_implementation_plan.md`.

The current repository is not ready to train a legitimate model. The prepared
dataset contains 540 `counterfactual-v1` rows, while the active feature contract
is `counterfactual-v2`; the readiness report consequently has zero eligible
outputs. The private Kaggriculture runtime needed to regenerate execution-
verified counterfactuals is also unavailable in the current environment.

## Stage 1 — restore the data-generation runtime

1. Install or provide the exact private `kaggriculture` environment and record
   its package/environment versions.
2. Run a one-episode smoke test through `scripts/harness.py`.
3. Verify that submitted market actions and intervention execution can be
   observed reliably. Do not label an intervention from requested actions alone.

Exit gate: a deterministic smoke episode completes and its execution audit is
populated without schema, action-contract, or environment errors.

## Stage 2 — regenerate causal v2 training rows

1. Select only promoted deterministic strategy families and the approved replay
   challengers; keep family identity in every row.
2. Generate paired interventions with `scripts/counterfactual_dataset.py` using
   raw simulation seeds shared within pairs.
3. Prepare rows through `scripts/prepare_watcher_data.py` so every row contains:
   `counterfactual-v2`, the current schema hash, all 129 ordered features, an
   execution-verified intervention audit, strategy family, split, and seed.
4. Keep seeds confined to one split and one strategy family.
5. Generate data incrementally, rerunning readiness after each batch rather than
   exhausting the 2,000-attempt cap blindly.

Exit gate for an output: at least 150 accepted rows, at least 40 positive and 40
negative labels, both signed classes in every split, and at least two usable
strategy families, as enforced by `scripts/watcher_readiness.py`.

Start with one bounded output—prefer `labor_expansion` if it reaches quota first.
Do not begin with a combined product/land/labor model.

## Stage 3 — train and gate one output

1. Freeze the regenerated JSONL files and record their SHA-256 hashes.
2. Run `scripts/train_watcher.py` only for readiness-eligible output IDs.
3. Select `C` on validation data; never tune against the untouched test split.
4. Require the ordinary test gate and every leave-one-family-out gate to pass.
5. Export only accepted outputs to `scripts/watcher_model.py`; rejected outputs
   must be absent from `MODELS`.
6. Verify the exported `score()` against scikit-learn's decision function on
   train, validation, test, and adversarial finite feature vectors.

Exit gate: `watcher_model.py` is nonempty, has the exact v2 schema/version/order,
contains only accepted IDs, uses no NumPy/scikit-learn imports, and achieves
numerical score parity within the existing test tolerance.

## Stage 4 — integrate into the standalone submission

1. Change `WATCHER_BACKEND` in `scripts/strategy_follower.py` to `linear_svm`.
2. Build with both required artifacts:

   ```sh
   python scripts/build_submission.py \
     --watcher-model scripts/watcher_model.py \
     --watcher-report artifacts/watcher_training_report.json \
     --manifest artifacts/model_submission_manifest.json
   ```

3. Confirm `scripts/main.py` contains embedded `_wf_` and `_wm_` namespaces,
   including `SCHEMA_VERSION`, and contains no runtime scientific imports.
4. Load the built artifact and require `watcher_signals(..., "linear_svm")` to
   report `backend == "linear_svm"` for every accepted output fixture.
5. Run source/artifact action parity with the same embedded model configuration.

The builder must fail, rather than silently falling back, when the source asks
for `linear_svm` without both model and acceptance report, or when model inputs
are supplied while the source still selects `rules`.

## Stage 5 — gameplay promotion

1. Compare deterministic rules against one accepted SVM output using paired
   seeds, both player orders, and every promoted opponent family.
2. Record signal changes, resulting action changes, final cash, minimum cash,
   service failures, land/labor spending, sale outcomes, and bankruptcy.
3. Confirm the model changes only its bounded output; missing/rejected outputs
   must retain independent deterministic-rule behavior.
4. Require a positive paired result with no family-specific safety regression.
5. Add further accepted outputs one at a time, repeating the same test before
   evaluating combinations.

Exit gate: gameplay evidence supports promotion, the downloaded submission hash
matches the manifest, and a forced missing/invalid-model test still fails safe to
deterministic rules.

## Required test additions

- A real generated `watcher_model.py` round-trip test through the builder.
- A built-`main.py` test proving an accepted output activates `linear_svm`.
- Schema version/hash/order mismatch rejection tests for both model and report.
- Empty and rejected export refusal tests.
- Nonfinite feature/score fallback tests per output.
- Source-to-built action parity with an embedded accepted model.
- Paired gameplay tests that attribute action and economic changes to each
  accepted output independently.
