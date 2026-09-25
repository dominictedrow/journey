# S07 — Track B: fine-tune the MAE-pretrained ViT (+ random-init control)

**Track:** B · **Machine:** Windows · **GPU:** yes (after MAE finishes, or at a good checkpoint; ask Dom)
**Depends on:** S03 (MAE checkpoint), S04 labels/folds, S05 (`train_sup.py`, metrics) · **Unblocks:** the S08 gate

## Goal
A quantified answer to "how good is Dom's from-scratch ViT?": MAE-pretrained vs random-init vs Track A, on the same
folds and labels.

## Read first
`02_vit_detailed_plan.md` §C, §D (transfer), and §C15 (LLRD) · the S03 session log (the MAE curve and probe results).

## Steps
1. Load the MAE encoder weights into `SliceViT` (strict key check; print missing/unexpected).
2. Fine-tune `KneeViT` through `train_sup.py` with LLRD (HP), the `AttnMILHead` first (proven), then `FindingQueryHead`.
3. **Random-init control:** identical config, no MAE weights (for the write-up).
4. Compare on fold 0 (then all folds for the winner): OOF report-label AUC, gold-58 OOF AUC with CI, per-finding table vs Track A.
5. First ablations if time allows: `pos2d` sincos vs learned; CLS vs GAP; stem vs patchify; C=3 vs C=1 (quick fold-0 runs).
6. **OOF rank-blend of A+B**: does B add diversity even if it's weaker?

## Done when
- [ ] The comparison table is in `docs/experiments.md` and summarized in the session log for the S08 gate.

## Review guide for Dom
Learning notes: LLRD; what the random-init control tells us; reading attention maps (plot the finding-query attention over
windows for 3 studies → `docs/figures/`).
