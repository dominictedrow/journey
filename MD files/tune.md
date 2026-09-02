# Watcher SVM tuning guide

`0 accepted` is a promotion result, not an indication that no training occurred.
Each eligible output is fitted, evaluated, and then accepted only if it clears
the project's offline gates. The levers below affect either predictive quality,
the export decision, or only the mechanics of training.

## Levers that can affect predictive accuracy

| Lever | What it changes | Expected impact |
|---|---|---|
| Data balance | More positive examples, especially in validation/test and each strategy family | High |
| Intervention and label definition | Make each output represent one clear, causal decision | High |
| Training examples | More accepted, execution-verified pairs and more state diversity | High |
| Strategy-family coverage | Balance examples across focal families rather than mostly one policy | High |
| Features | Add, remove, or repair state signals that predict the decision | High |
| `--c-grid` | SVM regularization strength; validation selects the best `C` | Medium |
| `--reweight-mode` | Use original sample weights or renormalize train weights | Low--medium |
| Class weighting | The model currently uses `class_weight="balanced"`; changing it shifts recall and precision | Medium |
| Model family | Compare the linear SVM against calibrated linear/logistic or nonlinear models offline | Medium--high |
| Decision threshold | Tune the SVM's zero-score decision boundary using validation data only | Medium |

## Levers that do not normally improve accuracy

| Lever | Effect |
|---|---|
| `--max-iter` | Optimizer convergence only. It should be increased only when a convergence warning occurs; once the warning is gone, further increases normally do not improve accuracy. The default is now 20,000. |
| `--margin-percentile` | Rescales exported confidence scores. It does not change binary classification accuracy. |
| `--min-per-class`, `--min-accepted`, `--min-signed`, and `--min-families` | Training eligibility and safety checks only. They do not change a fitted model's accuracy. |

## Recommended experiment order

1. Inspect class counts by output, split, and focal strategy family. Several
   current test partitions contain only 6--12 positive cases, so their metrics
   can be volatile.
2. Generate targeted, execution-verified data to improve positive-label and
   strategy-family coverage for each coherent output.
3. Validate the intervention semantics and labels. A model cannot learn a
   useful boundary from a target that combines unrelated decisions.
4. Test a finer regularization sweep around the current best values. For
   example:

   ```bash
   --c-grid 0.003 0.01 0.03 0.1 0.3 1 3 10
   ```

5. Evaluate threshold and class-weight alternatives on validation data only.
   Keep the test split untouched until the candidate is finalized.
6. Compare alternative model families offline only after the data and target
   quality are sound.

## The acceptance gate (post-repair, Phase 1)

The old "balanced accuracy vs. raw majority prevalence" description is gone --
that clause had already been changed to compare against a fixed `0.5`, and the
two clauses (`bacc >= 0.60` and `(bacc - 0.5) >= 0.10`) were algebraically the
same single `0.60` threshold on one tiny partition.

`train_watcher.py` now gates on four things, all sourced from
`scripts/watcher_config.py`:

1. **Grouped-CV C selection.** `C` is picked by mean balanced accuracy under
   `StratifiedGroupKFold` (group = seed group) over train u val; the test split
   is never touched during tuning. Export floor: CV mean `>= ACCEPT_MIN_CV_MEAN`
   (0.62).
2. **Bootstrap test CI.** The headline test metric is the 5th percentile of a
   2000-sample bootstrap of balanced accuracy, not the point estimate. Gate:
   `ci_lower >= ACCEPT_MIN_BALANCED_ACC_CI_LOWER` (0.58).
3. **Pooled leave-one-family-out.** Refit per held-out family, pool every
   held-out prediction, compute one balanced accuracy over the pool. Families
   that individually lack both signs still contribute their rows. Gate:
   `pooled >= ACCEPT_MIN_LOFO_POOLED` (0.55).
4. **Untouched confirm split.** Seed groups now split 60/15/15/10
   train/val/test/**confirm**. `confirm` is scored once, after the candidate is
   otherwise finalized, and must land within the test CI width of the test
   point estimate or the output is not exported.

`--min-per-class` now defaults to `5` (`watcher_config.MIN_PER_CLASS`); passing
`1` prints a `SMOKE TEST ONLY` warning and must not be used for a promotable
run.

## Generation-run reliability (2026-08-27)

A targeted batch run in the foreground repeatedly appeared to "fail" with no
data-side error. Root cause: `scripts/counterfactual_dataset.py` computed the
entire batch in memory and only wrote the corpus once, at the very end. A
foreground shell has a timeout (as low as 120s, capped at 10 minutes); a batch
of even a few hundred pairs (each pair replays a 720-step episode twice) can
exceed that, so the process was killed before its single write ever ran --
losing 100% of the batch's work every time, with the next attempt hitting the
same wall and dying the same way.

Fixes applied:

- `scripts/counterfactual_dataset.py` now writes and flushes each row to disk
  as soon as it is generated, instead of batching everything until the end. A
  killed or timed-out run keeps every pair it already paid for; a follow-up
  `--append` run resumes rather than repeating the same work. Seed-reuse and
  the 2,000-attempt-cap checks were also moved before generation starts, so a
  doomed run fails in milliseconds instead of after minutes of wasted
  simulation.
- Always launch batches with `run_in_background` (or `nohup ... &`) rather
  than in the foreground, so a long batch is not bound by the shell's timeout
  in the first place. The write fix above only makes a cutoff non-catastrophic
  -- it does not remove the reason for the cutoff.

## Superseded corpus (2026-08-31, repair-plan Phase 3)

Everything below this line that cites `data/counterfactual_dataset.v3.jsonl` or
per-output row counts is a snapshot of a **retired** corpus. Two contract
changes invalidated it:

- Phase 2 replaced the season-end money-delta label with a bounded-horizon
  blend and split safety into hard rejects vs. recorded soft flags
  (`schema_version` `counterfactual-v3`).
- Phase 3.3 dropped the mutually-collinear `turn` / `hour` / `day` features
  (every checkpoint is at hour 0), leaving `remaining_fraction` + the `phase_*`
  one-hot. Feature count 129 -> 126, `schema_version` `counterfactual-v4`.

`product:*:{exit,retain}` and the `premium_exit` / `premium_retain` intervention
family were also removed in Phase 3.1 -- nothing in `watcher_signals` ever
consumed an `:exit` / `:retain` score, so those rows trained models no runtime
read. The modeled outputs are now exactly `product:{8 products}:{increase,avoid}`
plus `land_expansion` / `labor_expansion`.

Re-run `scripts/watcher_readiness.py` against the regenerated
`counterfactual-v4` corpus for live numbers. The tables below are kept only as a
record of the *failure modes* to expect, not their counts.

## Dead (output, strategy-family) combinations

The same failure mode recurred across many outputs on the retired corpus: a
focal-strategy family's scripted policy either never exposes the targeted
product/decision in the intervention window, or re-executes through the
deviation so the tracked metric never changes -- so `execution_not_verified`
(or `unapplied_intervention` / `intervention_ineligible`) fires on every
attempt for that combination, not intermittently. Generating more pairs under
a combination already showing 0% acceptance at meaningful volume does not
improve it; the harness is correctly rejecting an intervention that provably
never took effect for that focal policy.

Fully exhausted -- hit the full 2,000-attempt cap with zero accepted rows and
zero usable families, permanently on deterministic rules until the
intervention/checkpoint design itself changes (see "Recommended experiment
order" item 3 and the intervention-feasibility work in repo-root `train.md`):

| Output | Attempts | Accepted |
|---|---|---|
| `product:WOOL:avoid` | 2000/2000 | 0 |
| `product:EGG:avoid` | 2000/2000 | 0 |

Down to one usable family (the other two are already proven dead for this
specific output, so it cannot reach the two-usable-family requirement without
a design change, regardless of how much more data is generated):

| Output | Live family | Dead families |
|---|---|---|
| `product:MELON:avoid` | `strategy_follower` | `strategies_v2.ring.agent` (400 attempts, 0 accepted), `strategies_v2.leader_clone.agent` (100 attempts, 0 accepted) |

Fix applied: `scripts/generate_until_quota.py` now tracks per-`(output,
family)` acceptance. A family is treated as dead for an output once it has
`--dead-min-attempts` (default 15) attempts with zero acceptances, and the
least-attempted *live* family is chosen instead of blindly picking the
least-used one overall. If every family is dead for an output it is marked
blocked (`all-families-dead`, distinct from `attempt-cap` in the final
report) and skipped, rather than burning further attempts on combinations
already proven not to work. See `scripts/test_generate_until_quota.py`.

## Outputs that may be structurally unlearnable: low-value "avoid"

`product:CARROT:avoid`, `product:TOMATO:avoid`, `product:STRAWBERRY:avoid`,
and `product:WHEAT:avoid` each have substantial accepted volume (84-110 rows)
but **every accepted row is labeled neutral (`0`)** -- none has ever crossed
the +-$250 signed-label threshold, regardless of family or attempt count.
Contrast with `product:MELON:avoid` and `product:MILK:avoid` (higher-value
products), which produce plenty of signed rows at similar or lower volumes.

This looks economic, not statistical: avoiding a cheap crop may simply never
move money by enough within the label window to register as a decision,
however many more pairs are generated. Before spending more attempts on these
four, either confirm that hypothesis (e.g. inspect the actual money deltas on
their neutral rows) or revisit the label window/threshold for low-value
products. This is a data/label-design decision, not a data-volume one.

## Current per-output snapshot

The `counterfactual-v3` snapshot here is retired (see "Superseded corpus"
above); so are the interim `counterfactual-v4` / `-v5` corpora. The current
corpus is `data/counterfactual_dataset.v6.jsonl` (`counterfactual-v5` schema:
the repair.md Phase B/C divest contract + Phase E feature normalization).
Read live numbers from `scripts/watcher_readiness.py`:

```bash
PYTHONPATH=scripts /Users/dominictedrow/miniconda3/envs/agric/bin/python \
  scripts/watcher_readiness.py --rows data/counterfactual_dataset.v6.jsonl
```

The corpus quota is >= 150 accepted / >= `watcher_config.MIN_SIGNED` (75) per
sign / >= 2 usable families for `land_expansion`, `labor_expansion`,
`product:MELON:{increase,avoid}`, and `product:MILK:{increase,avoid}`.

## Training command

Run a safe candidate without changing the runtime model:

```bash
PYTHONPATH=scripts /Users/dominictedrow/miniconda3/envs/agric/bin/python \
  scripts/train_watcher.py \
  --rows data/counterfactual_dataset.v6.jsonl \
  --output /tmp/watcher_model_candidate.py \
  --report /tmp/watcher_training_report \
  --min-per-class 5 \
  --min-accepted 150 \
  --min-signed 75 \
  --max-iter 20000
```

Review the output at `/tmp/watcher_training_report.md`.
