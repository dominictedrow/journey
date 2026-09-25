# S08–S12 — outlines (the session before each one expands it into its own brief file)

## S08 — Oct 8 decision gate + improvement round 1 (Oct 5–8)
- Present to Dom: the Track A vs B table (OOF + gold with CI + LB), runtime per model, and the MAE effect vs random init.
  **Dom decides** the gate outcome (placeholder threshold in `00_MASTER_PLAN.md` §5).
- Improvements ranked by public evidence [KT]:
  1. Denser/wider window sampling (up to +0.018 LB publicly).
  2. Resolution/crop.
  3. The finding-query head.
  4. The rank loss weight.
  5. Specialists for the weakest findings (Synovitis, Lateral OA, Lateral Meniscus, PF OA).
- Re-check the forum for host clarifications (LB split, test syntaxes) and the Overview for changes.

## S09 — Labels v2: noisy-student (Dom's pseudo-label idea) (Oct 9–12)
- The best model's OOF predictions on all studies → blend with the report labels (image weight higher where reports are silent:
  Synovitis, Fracture) → labels_v2 → retrain with noise (augmentation + stochastic depth).
- Guards: OOF only, soft targets, and gold OOF must not drop; the API budget remaining after S04 caps any re-reading.

## S10 — Ensemble/TTA + efficiency engineering (Oct 13–17)
- Accuracy submission: per-finding rank-mean blend of A (+ B if it earns its place OOF). **Flat weights preferred**; keep one
  flat-weight hedge (public per-finding LB-tuned weights overfit).
- Efficiency submission: a single fastest-good model; fp16; fewer windows; a header-parse fast path (the header ordering was
  12/32 min publicly); the DataLoader overlap; ToMe for ViT; ONNX/TensorRT if it's a real win.
  Use the trade-off from 02 §F: **0.001 AUC ≈ 70 s**.
- Profile on Kaggle with stage timings; iterate.

## S11 — Final selection + freeze (Oct 18–22)
- Pick the 2 finals: (1) best OOF-validated accuracy ensemble, (2) the efficiency model. Confirm both notebooks run on the full
  hidden test within limits (the submission must succeed; leave margin).
- Freeze the code; tag the commit; write the reproducibility notes (environment, commands, data/label versions).
- Before **Oct 22 23:59 UTC**: selections made in the Kaggle UI (Dom does this; Claude checks it).

## S12 — Winner obligations (only if top-10 LB or top-3 efficiency; Oct 23–Nov 5)
- Training + inference code, weights as a **public** Kaggle dataset, the environment (requirements + Kaggle image or Dockerfile),
  **CC-BY-NC 4.0**, a method description, a short video, and a forum post. Verify each dependency's license (OSI + commercial
  use). This is also the portfolio write-up of the from-scratch ViT.
