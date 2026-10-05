# Bugs from the 2026-09-30 review passes

Four parallel read-only reviews: model, training loop, data pipeline, MAE + inference.

**All fixed except the two items under "Not fixed"**, each with a regression test (`tests/test_review_fixes.py` for
the third batch). Logs: `docs/sessions_log/P05_2026-09-30.md` and `P06_2026-09-30.md`.

## Fixed
- **First pass, 6 bugs** (commits `96211c0`, `04e601c`): RankBank fp16 NaN, gradients leaking across epochs, silent
  no-training when batches < accum, shared fold run dirs, re-running finished folds, silent NaN eval.
- **Second pass, 5 bugs** (commit `fbad4c7`): `soft_bce` cancelled the confidence weights, the bundle rendered at the
  derived cache's `out_px`, the MAE stem norm saw zeroed images, cache-build resume could drop studies, AttnMIL NaN on
  an empty bag.
- **Third batch (P06), the rest of the second pass:**
  - **#6 Track A BatchNorm**: `freeze_bn` (default on): BN stays in eval mode with a frozen affine, so the running
    statistics stay the pretrained ones. Top-k averaging is then exact, and the grad-checkpoint double update can't
    happen. `train_sup` also logs the best single epoch next to the average (`best_epoch_val_macro_auc`,
    `avg_val_macro_auc`) and uses that epoch if the average is worse (`used_epochs`).
  - **#7 Inference windows**: arms with the same data config share one sample per study, with one collate and one
    host→device copy (`predict.sample_key`).
  - **#8 Non-finite prediction**: a non-finite arm output gets the fallback row and an `error` entry, and
    `write_submission` replaces non-finite rows instead of asserting.
  - **#9 cv2 threads**: `dataset.worker_init` (`cv2.setNumThreads(1)`) in every DataLoader: training, eval, MAE,
    inference and the loader bench. UNVERIFIED speed-up: measure with
    `scripts/perf/mae_loader_bench.py --cv2-threads 1 default`.
  - **#10 Worker respawn**: the train and eval loaders are built once with persistent workers. The epoch travels
    inside the sampler indices (`encode_epoch`). Measured: the index pickle is ~22 MB at corpus scale (26k series),
    not the hundreds of MB estimated, so the respawn time was the real cost. No compaction was needed.
  - **Low:**
    - Top-k file pruning now happens after `checkpoint()`, and missing epoch files are skipped when averaging.
    - STOP is polled after the epoch-end eval.
    - `check_run_identity` accepts a re-mounted folds file (same bytes, or the old path gone with the same name).
    - The OOM retry runs outside `except`, one study at a time.
    - No wasted micro-batches: the sampler stops at `steps_per_epoch * accum`.
    - Loaders are no longer rebuilt every epoch.
    - The MAE recon grid takes `[::kr]`.
    - MAE resume refuses a schedule/data-order change (`--allow-config-change` overrides) and zeroes grads at each
      epoch start.
    - `mae.norm_pix_eps` is configurable (default unchanged; an ablation for the pilot).
    - The shard join takes a deadline (`infer.py --deadline-hours 8`); a hung shard is killed and its studies get
      the fallback row.
    - Track A inference decodes only slotted series (`preprocess_study(render="slots")`).
    - Series with no decoded slice are dropped (`dataset.usable`).
    - Parity check: the bundle requires the full params plus `preproc_code_hash`. `infer.py` refuses a mismatch
      (`--allow-preproc-drift` overrides), and the cache build refuses to resume across a code change.
    - `load_paths` needs only `cache_root`.
    - Memory peaks: `series_window` subsamples per slice (same pixels), and the rescale runs in float32.
    - `slice_pos.proj.weight` has no weight decay.
    - Bags are cast to float per chunk.
  - **Nits:**
    - RankBank is checkpointed.
    - The rank loss thresholds are inclusive, like `binarize`.
    - Per-epoch gold bootstrap uses 200 replicates (1000 at the end).
    - MONOCHROME1 is negated per slice (one mapping for the whole series).
    - Duplicate-position ties break by InstanceNumber, then file path.
    - Dedup uses clusters instead of fixed buckets.
    - `derive_shard` handles 0-byte shards.
    - `max_series` eval keeps the slotted series first.

## Model pass 2026-10-01 (all fixed; tests in `tests/test_perf_pass3.py`)
- **Double compile of the first shape** (`PosEmbed2D`): the sincos table lived in a dict filled inside `forward`.
  Dynamo guarded on the dict, so the first chunk shape compiled once with the cache empty and again with it filled:
  one wasted SliceViT compile per process, in training and in the scored runtime. Now a non-persistent buffer for
  the base grid (not in `state_dict`, so old checkpoints load); other grids (MAE at a lower resolution) are computed
  inline. Measured with Dynamo counters: 3 graphs for 2 shapes before, 2 after.
- **Second GPU→CPU sync per forward** (`KneeViT.forward`, `TrackAModel.forward`): `x[mask]` and `tokens[mask] = feats`
  each run `nonzero`. The second one came after the whole encoder was queued, so the CPU waited for the encoder before
  it could queue the head and backward. Now `real_index` once, then `index_select` / `index_copy`. Same forward and
  gradients bit for bit (test). Speed-up UNVERIFIED on a GPU (expected small: one pipeline bubble per step).
- **DropPath bf16 keep factor with LayerScale off**: the earlier fix covered only the γ path. The factor is now
  always drawn in at least fp32 (with LayerScale off this makes the residual stream fp32, as γ already does).

Noted, not changed (small): `to_input` makes 4 elementwise passes with 4 allocations per chunk, and it runs outside
the compiled encoder (< 1% of a chunk's encode time by estimate); `AttnMILHead.norm` uses eps 1e-5, while every other
LayerNorm uses 1e-6 (harmless, but changing it changes trained checkpoints).

## Not fixed (on purpose)
- **MAE LR "probably too low"** (blr 1.5e-4): this is a tuning question, not a bug. It needs a GPU sweep (blr ×3–4)
  with the linear probe as the signal: an item for the MAE pilot.
- **Weight decay on `head.cls_w`**: kept decayed. It is the classifier weight (a Linear in all but form), which ViT
  recipes decay. Only `slice_pos.proj.weight` (a position embedding) moved to no-decay.

## Notes (not bugs)
- **Pure-fp16 Track A**: covered by `test_half_trackA_runs_and_matches_fp32` (P05). T4 behaviour: UNVERIFIED.
- **Silent 0.5 labels in `labels_v0`**: many entries are exactly 0.5 where every reader was silent (33% of Fracture,
  43% of Baker's). Prevalence summaries that cut at ≥ 0.5 overstate prevalence. These entries train at their low
  weight (0.1). They never form rank pairs: 0.5 is neither ≥ 0.6 nor ≤ 0.4.
- **Loss scale**: the `soft_bce` fix lowers the BCE term to about the mean label weight times the old value. Re-check
  `rank_w` if it was tuned against the old scale.
- **Overfit test**: the per-pixel LN stem memorizes noise images more slowly than GN; the test runs 250 epochs.
  Compare LN vs GN with the linear probe after the MAE pilot.
