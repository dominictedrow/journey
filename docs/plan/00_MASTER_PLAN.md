# 00 — Master Plan (multi-session)

Written 2026-09-24 (S00). Owner: Dom. Executor: Claude sessions on the Windows box (Claude writes the code, Dom reviews).
Deadline: **Oct 22, 2026 23:59 UTC** (final submission). Entry/merge deadline Oct 15. Winner obligations due Nov 5.

## 1. Goal and the bar
- **Goal:** 1st place (private LB), efficiency prize as a stretch, and Dom learns to build a ViT.
- **The bar on 2026-09-24:** LB top **0.959**, top-20 ≥ 0.955. Public notebooks 0.941–0.947 (one shared forked ensemble:
  DINOv2-S / CoAtNet / RadImageNet arms + public LLM labels). A single fine-tuned CoAtNet on LLM soft labels: 0.914–0.924 LB.
- **Where the edge is (consensus, forum + notebooks):** label quality > model capacity. Everyone shares the same public
  label sets; better labels and honest validation are the differentiators.

## 2. Two tracks, one pipeline

```
                        ┌──────────── Track A: pretrained backbone (safety net, early submission)
 DICOM → cache (S02) ───┤
 reports → labels (S04)─┤
 folds (S04) ───────────└──────────── Track B: Dom's ViT from scratch → MAE self-pretraining → fine-tune
                                                      ↓
                                     Oct 8 gate → ensemble / pick → efficiency → final 2 submissions
```
- Both tracks read **the same cache, labels, folds, metric code, and inference preprocessing**. Only the model differs.
- Track B never imports pretrained weights (Dom's "from scratch" goal; see gap G1 in `02_vit_detailed_plan.md`).

## 3. Session map

Each session has a brief in `docs/plan/sessions/`. Briefs for later sessions are **outlines** that the session before them
refines with what it learned (the rule: whoever finishes S<N> updates brief S<N+1> if reality changed).

| # | Session | Track | Target dates | Needs GPU | Status |
|---|---|---|---|---|---|
| S00 | Research + plan (Mac) | — | Sep 24 | no | ✅ done |
| S01 | Windows env + data audit | D | Sep 25 | smoke | ☐ |
| S02 | Preprocessing library + cache v1 | D | Sep 25–26 | no (CPU decode, overnight) | ☐ |
| S03 | ViT modules from scratch + unit tests + MAE pipeline → **launch MAE pretraining** | B | Sep 26–27 | **yes, multi-day** | ☐ |
| S04 | Labels v1: Claude API reader + public sets + blend + gold calibration + folds | L | Sep 27–28 | no | ☐ |
| S05 | Track A baseline training (pretrained 2.5D + per-finding attention MIL) | A | Sep 28–30 | yes (pause MAE if still running) | ☐ |
| S06 | Kaggle inference notebook v1 + offline wheels + **first submission** | A | Sep 30–Oct 1 | Kaggle | ☐ |
| S07 | Track B fine-tune from MAE (LLRD) + scratch-vs-MAE-vs-A comparison | B | Oct 1–5 | yes | ☐ |
| S08 | **Oct 8 decision gate** + improvement round 1 (per-label queries, resolution/slices, specialists) | A+B | Oct 5–8 | yes | ☐ |
| S09 | Labels v2: noisy-student relabel (Dom's pseudo-label idea) + blend | L+A+B | Oct 9–12 | yes | ☐ |
| S10 | Ensemble/TTA + efficiency engineering (fp16, slice budget, ToMe, runtime profile) | A+B | Oct 13–17 | yes + Kaggle | ☐ |
| S11 | Final selection (2 subs), freeze, reproducibility + write-up skeleton | — | Oct 18–22 | Kaggle | ☐ |
| S12 | Winner obligations (if top-10/efficiency): code, weights dataset, CC-BY-NC, video, write-up | — | Oct 23–Nov 5 | no | ☐ |

**Why MAE launches in S03, before labels exist:** self-supervised pretraining needs no labels, takes days, and the GPU
would otherwise sit idle while labels are built through the API (S04). Everything that needs labels comes after.

## 4. GPU calendar (the single 12 GB GPU is the bottleneck; keep this current)

| Job | Session | Start | Expected end | Resumable? | Notes |
|---|---|---|---|---|---|
| _(none yet)_ | | | | | |

Rules: long jobs checkpoint at least every 30 min and resume from the last checkpoint; any session may pause a
*resumable* job for a short, higher-priority job (Track A baseline, submission debugging) **only after asking Dom**.
Log pauses here.

## 5. Decision gates
- **G-Oct 1 (end S06):** a valid Kaggle submission exists (Track A). If not, all effort goes to getting one.
- **G-Oct 8 (S08):** is Track B (MAE-pretrained scratch ViT) within ~0.01 of Track A on the out-of-fold report-label AUC,
  with the gold-58 OOF agreeing in direction? Yes → keep investing, and B joins the ensemble. No → B becomes an ensemble
  member only if it adds diversity (a rank-blend gain OOF); otherwise B continues as the learning project and portfolio
  write-up, and the compute goes to A. *(Placeholder threshold: Dom confirms at the gate.)*
- **G-Oct 15:** team-merge/entry deadline. Confirm Dom accepted the rules; decide on teammates (open question).
- **G-Oct 20:** freeze the models; only the selection of 2 final submissions after this.

## 6. Validation protocol (applies to every session; details in 02 §B)
1. **Primary:** 5-fold out-of-fold macro AUC on all ~4.3k report-labeled studies, grouped by **normalized report hash**
   (46 report texts are shared across 177 studies) **and** a site/scanner proxy.
2. **Secondary:** gold-58 macro AUC, **out-of-fold only**, reported with a 1,000–2,000× bootstrap 95% CI. Never tune
   per-finding knobs on it. A change counts only if (1) and (2) agree in direction.
3. **Tertiary:** public LB, at most a few probes/day; it's only ~a fraction of ~1,300 test studies, so don't chase it.
4. Per-finding AUC table for every run. Macro AUC hides where the problems are.

## 7. Risks (and the mitigation already in the plan)
| Risk | Mitigation |
|---|---|
| Scratch ViT underperforms (every RSNA MRI/CT winner used pretrained CNN/hybrids; data-hungry ViTs) | Track A in parallel; MAE self-pretraining; conv stem; Oct 8 gate |
| Label noise caps AUC (report ≠ gold readers; Synovitis/Fracture often unmentioned) | Multi-reader LLM blend, soft labels, per-finding policy, gold calibration, noisy-student round |
| Overfitting the 58 / the public LB | Validation protocol above; limited LB probes |
| Kaggle runtime/OOM on the hidden test (~1,300 studies) | Sequential arms, per-study try/except fallback, a runtime budget per arm measured in S06 |
| 16 GB RAM / 12 GB VRAM | Memory-mapped uint8 cache, gradient checkpointing, bf16, small batches + accumulation |
| Windows quirks (DataLoader spawn, SDPA kernels, torch.compile) | Checked in S01; nothing depends on torch.compile |
| Test DICOMs use compressed syntaxes the train sample didn't show | Bundle pylibjpeg/gdcm wheels in the inference notebook anyway |

## 8. Where things live
- Plan: `docs/plan/` (00 master, 01 barebones ViT, 02 detailed plan + gap log, `sessions/` briefs)
- Evidence: `docs/research/`, `docs/kaggle/`
- Records: `docs/decisions.md`, `docs/experiments.md`, `docs/sessions_log/`
- Code (from S01): `src/knee/{audit,data,labels,model,train,infer}/`, `notebooks/`, `configs/`, `tests/`
