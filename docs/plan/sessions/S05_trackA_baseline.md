# S05 — Track A baseline (pretrained 2.5D + per-finding attention MIL)

**Track:** A · **Machine:** Windows · **GPU:** yes (**ask Dom before pausing MAE**; resume it after)
**Depends on:** S02 cache, S04 labels/folds · **Unblocks:** S06 (first submission)

## Goal
A trained, validated pretrained-backbone model reproducing the public single-model recipe on *our* labels/folds, with
OOF predictions saved for later blending.

## Read first
`02_vit_detailed_plan.md` §E · `docs/kaggle/top_solutions_from_notebooks.md` (the recipe, numbers, pitfalls) ·
`docs/kaggle/data_and_dicom_from_notebooks.md` (the window layout).

## Steps
1. `src/knee/model/trackA.py`: a timm backbone (default `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, `num_classes=0`) + the
   shared `AttnMILHead` from S03 (reuse, don't duplicate).
2. `src/knee/train/train_sup.py`: **shared by Track A and B** (the model is chosen by config). Confidence-weighted soft BCE + rank
   loss; AdamW with separate backbone/head LRs; warmup + cosine; bf16; grad checkpointing; accumulation to the target
   effective batch (HP); K random windows in training, dense windows at eval; top-k epoch weight averaging; saves OOF preds.
3. `src/knee/utils/metrics.py`: macro/per-finding AUC, bootstrap CI, OOF evaluation on report labels + gold-58.
4. Smoke run (2 folds × 1 epoch, small H) → the full fold-0 run → if sane, the remaining folds (or a single full-train model if
   time presses; log the choice).
5. Log each run in `docs/experiments.md`, with the per-finding table.

## Done when
- [ ] Fold-0 (at least) OOF report-label macro AUC + gold-58 OOF AUC with CI logged; the checkpoint is saved outside the repo.
- [ ] MAE resumed; the GPU calendar is updated.

## Review guide for Dom
Learning notes: fine-tuning a pretrained backbone (why the backbone LR ≪ the head LR); the attention-MIL head; what the
per-finding table says.
