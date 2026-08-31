# Phase 15 execution report

Execution date: 2026-08-22

## Outcome

Phase 15 was executed as far as the available workspace and authorization
permit. The final artifact remains the deterministic rules submission and no
experimental mechanism or watcher output was promoted. Phase 15 is not
complete: the private Kaggriculture runtime is unavailable, fresh causal data
cannot be generated without it, and no Kaggle upload was authorized.

## Runtime gate

- The host's default `python3` is Python 3.9.6 with
  `kaggle-environments==1.18.0` and NumPy 2.0.2.
- Python 3.11.14 is available at
  `/Users/dominictedrow/miniconda3/bin/python3.11`, but that interpreter does
  not have the evaluation dependencies installed.
- Calling `make("kaggriculture", ...)` with the installed public package
  returns `InvalidArgument: Unknown Environment Specification`.
- Checked-in replay metadata identifies environment `kaggriculture`,
  environment version `0.1.0`, 720 episode steps, and private module version
  `1.32.7` as recorded by the earlier phase reports.
- The repository contains replay observations, but not the executable custom
  environment source.
- Kaggle CLI 2.2.4 was installed into `/tmp` only and the documented read-only
  competition download succeeded. The 16.1 KiB archive contains only
  `AGENTS.md` and `README.md`; it does not contain the environment source. The
  downloaded archive SHA-256 is
  `91298b47c5dd34500177e6cf6c41a9c2ca254804f893d135a7c06c3d920ee739`.

Therefore smoke episodes, full seasons, both player positions, repeated-seed
money determinism, executed-order telemetry, paired promotion screens, causal
simulation, gameplay trajectory validation, and runtime benchmarking remain
blocked. Replay-state action parity is not treated as a substitute.

## Deterministic experiments

Rounds 30 through 37 remain unpromoted. Their checked-in result files record
the following decisions:

| Round | Mechanism | Decision |
|---|---|---|
| 30 | hiring ramps | retain control; gameplay not run |
| 31 | livestock specialists | retain control; gameplay not run |
| 32 | melon exit | retain permanent-melon control; gameplay not run |
| 33 | strawberry replacement | retain control; blocked by predecessor |
| 34 | late wheat replacement | retain control; blocked by predecessor |
| 35 | sale batches | retain legacy sales; runtime unavailable |
| 36 | sale thresholds | retain legacy sales; blocked by predecessor |
| 37 | conditional third land | retain two-land control; currently ineligible |

These are conservative retention decisions, not evidence-based performance
rejections. The required 20-episode screens and 30/60-episode confirmations
have not run.

## Replay challengers

The reconstructed challengers remain excluded from the representative
opponent registry. Held-out exact-action rates range from 0.247 to 0.427, and
gameplay trajectory validation is unavailable. They remain observational
diagnostics and are not described as leaderboard equivalents.

## Watcher gates

The readiness report contains 540 obsolete `counterfactual-v1` rows. It has no
eligible outputs and no accepted execution-verified v2 rows. Every intended
output is below quota and still has attempts remaining; consequently the
2,000-attempt decision gate has not been reached for any output. Training was
correctly skipped, no `watcher_model.py` was exported, watcher gameplay is not
applicable, and `WATCHER_BACKEND` remains `"rules"`.

## Verification performed

- The post-review correctness fixes stop treating submitted market orders as
  executed fills, isolate dataset splits by raw simulation seed across focal
  families, require every leave-one-family-out diagnostic to pass before
  watcher export, include all collected operational failures in causal safety
  labeling, and validate action grammar across sampled opening, midgame, and
  endgame replay observations.
- Python 3.11 compilation passed for the editable source, generated artifact,
  build path, evaluation path, phase runner, replay challenger, causal dataset,
  trainer, and readiness checker.
- The focused build, instrumentation, shared-feature, causal-dataset, watcher,
  and trainer suites passed: 51 tests total.
- A post-review build reproduced `scripts/main.py` byte-for-byte; the fixes do
  not change deterministic strategy behavior.
- The previously frozen 5,000-observation exact source/artifact parity check
  remains represented by the unchanged hashes below.
- `scripts/strategy_follower.py` SHA-256:
  `250ee696a8bdd5e73f76117139fc54b27b60cd57c0190abefa2ab25ead37e1e4`.
- `scripts/main.py` SHA-256:
  `0e21bff58231afc7449868f923cf468e73386a809ada2f80cc4d55bc35a65d6e`.

No upload, submission, or other Kaggle mutation was performed. A read-only
competition-data download was performed solely to look for the missing
runtime. The external submission gate requires explicit authorization after
the gameplay and watcher gates are closed.
