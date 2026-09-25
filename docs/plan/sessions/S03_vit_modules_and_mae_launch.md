# S03 — Dom's ViT from scratch (modules + tests) and launching MAE pretraining

**Track:** B · **Machine:** Windows · **GPU:** yes. Ends by **launching a multi-day MAE job**.
**Depends on:** S02 cache · **Unblocks:** S07 (fine-tuning from MAE)

## Cache facts from S02 (read before building the MAE data path)
- Cache: `<cache_root>/cache_v1_train/`, read it with `knee.data.dataset.CacheReader` (lazy memmaps, spawn-safe).
  Every slice of every series, uint8, 384 × 384, 130 mm FOV (0.34 mm/px), canonical right-knee frame. `docs/cache.md`.
- 2.5D windows: `dataset.window_indices` / `gather_windows` (with `out_px=` for a lower MAE resolution, INTER_AREA).
- MAE needs a flat index of (series row, centre slice). Build it from `series_index.parquet` (`n_slices`, `valid`).
  **Weight sampling per series, not per slice**: one stub 3D GRE series has 160 slices vs ~30 for a typical series.
- Series-type ids for `SeriesTypeEmbed`: `dataset.type_ids(row)` (plane 3, fluid 2, contrast 6, field 3, slot 6).

## Goal
Hand-written ViT modules (no `timm` model classes in Track B), each unit-tested, assembled into `SliceViT`/`KneeViT`;
an MAE wrapper; a timed pilot; then the long pretraining run started in the background with checkpoint/resume.

## Read first
`01_vit_barebones.md` (shapes and modules) · `02_vit_detailed_plan.md` §C and §D · `docs/research/vit_fundamentals.md`
§1–§9 · `docs/research/ssl_and_labels.md` §1–§5. **Optional (open item 4):** read forum thread 737696 about how the
public "Raptor" SSL ViT was trained (`kaggle competitions topics list` / `topic-messages`).

## Build (`src/knee/model/`), in this order, each with its test in `tests/test_model.py`
1. `patch_embed.py`: `PatchEmbed` (patchify Conv2d) **and** `ConvStem` (default). Test: patchify ≡ unfold+Linear (allclose).
2. `pos_embed.py`: 2D sin-cos (any grid), `SlicePosEmbed` (continuous z → sincos → Linear), `SeriesTypeEmbed`
   (plane, fluid, contrast_class, field_strength). Test: shapes; sincos regenerates for a new grid.
3. `attention.py`: `MultiHeadSelfAttention` with `impl="manual"|"sdpa"`, optional QK-norm, key-padding mask;
   `CrossAttention` (queries ← tokens). Tests: manual ≡ sdpa ≡ `nn.MultiheadAttention` with copied weights; masked keys
   get zero weight.
4. `mlp.py`, `block.py`: `Mlp`, `DropPath`, `LayerScale`, pre-norm `Block`, `CrossBlock`. Test: DropPath is identity in eval;
   residual shape.
5. `slice_vit.py`: `SliceViT` (stem → +pos2d → blocks → norm → GAP (default) or CLS). Test: forward on [N,3,H,W].
6. `heads.py`: `AttnMILHead` (12 gated-attention softmaxes over windows, the proven public head) and `FindingQueryHead`
   (12 learned queries, cross-attention over window tokens + a shallow self-attention aggregator). Zero-init logit layers.
7. `knee_vit.py`: `KneeViT` (window bag → SliceViT → + slice-pos + series-type → head). Configurable via `configs/`.
8. **Sanity suite:** loss at init ≈ 0.693/label; overfit 8 studies to ~0 loss (fp32, the decision allows fp32 for exact
   tests); padded windows don't change the output; bf16 forward/backward runs; peak-VRAM table for (H, K, E, depth) combos.

## MAE (`src/knee/ssl/mae.py`, `train/pretrain_mae.py`)
- Random patch masking (ratio HP, default 0.75); encoder = `SliceViT` blocks over visible tokens only; a lightweight decoder;
  per-patch normalized-pixel MSE on masked patches.
- Data: 2.5D windows sampled from **all** series in the S02 cache (train only), random crop/scale jitter.
- bf16 autocast, AdamW, warmup + cosine, grad clipping; **checkpoint every ≤30 min + resume**; logs to `runs/mae_*/`.
- **Probe:** every N epochs, a frozen-feature linear probe on report labels, if S04 labels exist by then; otherwise
  reconstruction loss plus saved reconstruction images, and the probe starts once labels land.
- **Pilot:** 1 epoch (or a fixed number of steps) → throughput (windows/s), VRAM, and an extrapolated wall-clock for the
  planned epochs. **Ask Dom** to confirm the total duration before launching (it occupies the GPU for days).
- Launch detached (e.g. `start /b` or a scheduled task: pick one and document how to check/stop it in HANDOFF);
  add it to the GPU calendar.

## Done when
- [ ] All model tests pass on CPU and GPU; the sanity suite passes; the VRAM table is in the session log.
- [ ] The MAE pilot numbers are logged in `docs/experiments.md`; Dom approved the duration; the job is running; the GPU calendar is updated.
- [ ] HANDOFF §4 says how to check MAE progress and how to pause/resume it.

## Review guide for Dom
Walk through `attention.py` line by line (shapes at each step); learning notes: (1) why pre-norm; (2) why a conv stem helps
small-data ViTs; (3) how MAE's asymmetric encoder makes pretraining cheap; (4) the finding-query head as "12 learned
questions" asked of the study.
