# 02 — Detailed Plan (research-backed), section by section

Written 2026-09-24 (S00) on top of `01_vit_barebones.md`. For each section: **Default** (what we build first),
**Ablations** (what we test later), **Evidence** (where the reasoning comes from), **Tests / done-criteria**, and
**Gaps** (open, or answered by Dom → `docs/decisions.md`).
**Hyperparameters are placeholders (`HP`) to tune later.** Numbers quoted from sources are there to explain choices, not to fix settings.

Evidence keys:
- [F] `docs/research/vit_fundamentals.md`
- [3D] `docs/research/vit_3d_medical.md`
- [SSL] `docs/research/ssl_and_labels.md`
- [W] `docs/research/rsna_winners_and_efficiency.md`
- [KL] `docs/kaggle/labels_from_notebooks.md`
- [KT] `docs/kaggle/top_solutions_from_notebooks.md`
- [KD] `docs/kaggle/data_and_dicom_from_notebooks.md`
- [KF] `docs/kaggle/forum_and_official_pages.md`

---

## A. Data + preprocessing (Dom's DICOM checklist, annotated with what we now know)

Facts from public notebooks and the forum; **all must be re-verified in S01** against our copy of the data:
- 4,407 train studies; 24,371 series; ~819k slice files; 3–14 series per study (median 5).
- Every study has all 3 planes. Slices per series: 11–320, median 30.
- Pixel spacing 0.137–1.172 mm (median 0.3125); slice thickness 0.6–5 mm (median 3.0).
- Siemens, Philips, GE and others; 1.5 T and 3 T. [KD]

| # | Checklist item | Default | Evidence / notes |
|---|---|---|---|
| 1 | Decode by transfer syntax | pydicom + pylibjpeg (+openjpeg, libjpeg) + gdcm fallback. **Verify all files; list the corrupt ones** | Every sampled train file is Explicit VR LE (uncompressed); 0 failures in 207,801 slices [KD]. The Data page still lists 4 syntaxes, so ship the decoders in the inference notebook as insurance. 2 corrupt train files reported [KF] |
| 2 | float32 + slope/intercept | `apply_modality_lut` (Philips has non-unit slopes, RescaleType "normalized") | [KD], [KT] |
| 3 | Per-series normalization | **Percentile window, 1st–99th or 2nd–98th, per series**, → [0,1], stored as uint8 | All top arms [KT]. z-score is an ablation (HP) |
| 4 | Orientation + slice order | **Sort by IPP projected on the IOP normal** (never by filename: that matches physical order for only ~5% of slices; InstanceNumber runs backwards in >⅓ of series) | [KD], [KT] |
| 4b | **Laterality canonicalization** (new) | Detect left/right from IOP/IPP geometry (the Laterality tag is missing on ~50%; the geometry rule agrees with the tag 92.9% of the time); **mirror so every knee looks like a right knee** (mirror in-plane LR for cor/ax; reverse the slice order for sagittal) | [KD], [3D §7]; public stacks note that normalizing L/R matters for the 4 medial/lateral findings [KT]. S01 verifies the rule |
| 5 | Resample to a common grid | **Fixed physical crop (HP ≈ 130–140 mm, centred) using PixelSpacing**, then resize to `H` px (HP; public 336–384 px ≈ ≤0.4 mm/px). No through-plane resampling: we **sample slices** instead | [KT]. Dense, wide slice sampling was worth up to +0.018 LB [KT] |
| 6 | Overlays | None observed (UNVERIFIED); ignore if present | [KD]. S01 checks 60xx groups |
| 7 | Private tags | Stripped by the host (69 public keywords observed) | S01 writes `docs/dicom_tags.md` |
| 8 | Organize study→series→instance | `pd` index from `train_series.csv` + header scan (cached to parquet once; header ordering is the inference-time bottleneck [KT]) | |
| 9 | Sequence type as conditioning | **`Fluid_Sensitive` ≡ `Fat_Suppression` on every row** (1 bit, not 2). Real contrast from TR/TE (+ InversionTime → STIR). Default conditioning tokens: plane (3) × fluid (2) × **TR/TE contrast class** (PD/T1/T2/GRE/STIR, derived in S01/S02) | [KD]. CLAUDE.md's "correlated but not equivalent" is wrong for train (check test in S06) |
| 10 | Grayscale inversion | **Off.** No MONOCHROME1 in the corpus; Dom's rule (only within the same scan type) holds as an ablation | [KD] |
| 11 | Multi-view input | Default **token streams** (each series its own slice tokens, tagged by type). Channels = **3 adjacent slices of one series** (decision 2026-09-24). Plane stacking stays an ablation | [3D §3], [KT] |
| 12 | Metadata | plane / fluid / contrast class / field strength (1.5 vs 3 T) as embeddings. **PatientSex exists (94.6% of studies); PatientAge doesn't** (UNVERIFIED). Sex: ablation only. Scanner metadata alone gives ~0.60 AUC grouped (site memorization) → don't feed Manufacturer directly | [KD], [KF] |
| 13 | Splits | **5-fold GroupKFold by normalized report hash** (46–49 texts shared by 177–183 studies), also grouped/stratified by a scanner-fingerprint site proxy; gold-58 spread across folds | [KL], [KT], [KD] |
| 14 | Cache | Memory-mapped **uint8** arrays on the fastest internal drive: per series, all slices after crop/resize at `H` (HP) plus a presence mask. Build is CPU-decode bound (Kaggle: ~50–64 slices/s; public caches 11–94 GiB) | [KD], [KT]. S02 measures ours |

**Series slots (for Track A and as the default bag layout for Track B):** up to 5 slots:
- sag-fluid
- sag-other
- cor-fluid
- cor-other
- axial (fluid preferred)

Each slot has a presence mask. Slices are sampled densely across **2–98%** of each series (HP: total windows per study, public 64–96). Track B's bag is variable-length, so it can take *all* series. The slots are a default, not a limit. **Series dropout** during training handles missing slots [3D §3].

Plane relevance by finding (for the attention-head design and later specialists; soft priors only):
- ACL → sagittal
- MCL → coronal
- Menisci → sagittal + coronal. **Axial was empirically strong for meniscal tears** in one 2025 study.
- PF OA → axial + sagittal
- Effusion and Baker's cyst → fluid-sensitive sagittal/axial
- Contusion → fat-sat fluid-sensitive

[3D §4]. **Don't hard-code plane priors; let per-finding attention learn them.**

**Tests:**
- A golden test comparing our preprocessed output with a hand-checked study (plotted).
- Train/infer parity: bit-identical cache from the same function.
- A slice-order monotonicity check.
- A laterality-rule accuracy check against the tag where present.

## B. Labels + validation (decisions of 2026-09-24 apply)

**Default pipeline (S04):**
1. **Our reader:** Claude API, JSON schema per finding:
   - `state ∈ {present, absent, uncertain, not_mentioned}`
   - `severity 0–3`
   - `evidence_quote` (for audit)
   - `confidence`

   Multilingual input, English schema. Pilot on 100 reports → measure cost (budget ≤ $50 total) and agreement on the gold-58 → then the full run.
   Prompt conventions (from what measurably helped [KL]):
   - Normal counts as absent.
   - Not-mentioned stays its own state (low weight, not negative).
   - Effusion counts as positive at moderate or above (HP: threshold fitted on gold).
   - Degenerative bone oedema is not a contusion.
   - OA per compartment.
2. **Public sets:** pilkwang (Claude Opus 5, YES/NO/UNK + severity), stevenleehans v2/v4, and others after a **license check**.
   **Exclude any set containing gold answers** (yunusgmsoy v5, barun2104), and run a leak check (AUC=1.0 on gold means leaked).
3. **Blend** readers in probability space → soft label `p ∈ [0,1]` + per-label weight `w = 2|p−0.5|` or agreement×certainty.
   **Synovitis fill from Effusion** when Synovitis is not mentioned (label agreement 0.678 → 0.790 [KL]).
4. **Gold calibration:** only a tiny per-finding mapping (a threshold or monotone map from reader states to target), fitted **out-of-fold** on the 58.
   No per-finding blend weights tuned on gold (that hurt the LB: 0.887 vs a 0.906 baseline [KL]).
5. **Label QA report:**
   - per-finding reader agreement
   - gold AUC per reader and per blend, with bootstrap CIs
   - the disagreement list (report says present, gold says absent: e.g. Effusion 16/23, Contusion 20/39)

   Known ceiling: report labels vs image-read gold ≈ 0.88–0.92 [KL].
6. **Later (S09): Dom's pseudo-label idea as noisy-student:**
   - The best image model (trained on report labels) predicts OOF on all studies.
   - Blend with the report labels, most useful where reports are silent (Synovitis 84% silent, Fracture 57% [KL]).
   - Retrain with noise (augmentation, stochastic depth).

   Guard against confirmation bias: use only OOF predictions, keep soft targets, and cap the image weight on findings with good report coverage.

**Validation:** see `00_MASTER_PLAN.md` §6.
- Bootstrap CIs on gold. The per-finding CI can span 0.3; the macro CI is about ±0.03–0.04 [KL], [SSL §9].
- Keep a `metrics.py` shared by all tracks: macro AUC, per-finding AUC, bootstrap, and a rank-blend helper.

**Gaps:**
- (open) whether the test labels come from the same image readers as the gold 58 (likely; UNVERIFIED)
- (open) public label-set licenses: S04 checks them

## C. Model: Track B, Dom's ViT (every component from CLAUDE.md, detailed)

Overall: the **2.5D factorized encoder** (ViViT "Model 2" pattern) from `01`:
- a per-window `SliceViT`
- a study-level aggregation over all window tokens
- 12 outputs

Joint attention over a whole study (~23k tokens) doesn't fit in 12 GB (~1 GB of attention scores per head per layer in bf16 [3D §2]). The factorized version costs ~257²/slice + ~100² at study level.

### C1. Patch embedding / tokenizer
- **Default:** a **convolutional stem** (a few 3×3 stride-2 convs + a final 1×1/stride projection to `E`) instead of one patchify conv. It makes AdamW optimization robust and helps with small data [F §1, §7] (Xiao et al. 2021: closes the AdamW/SGD gap, +1–2% top-1).
- **Keep** the plain `Conv2d(k=P, s=P)` patchify as a baseline and as a unit-test reference (equals unfold + Linear).
- **Ablations:** plain patchify; SPT (shifted patch tokenization) [F §7]; patch size `P` (HP; 16 default).
- **Tests:** stem output shape; the patchify ≡ unfold+Linear allclose test.

### C2. Positional encodings (in-plane)
- **Default:** fixed **2D sin-cos** (no params). It beats learned 1D when training from scratch on a budget (Beyer et al.) [F §2]; type barely matters, having one does.
- **Ablations:** learned; RoPE-2D (better resolution extrapolation) [F §2].
- **Must support resolution change** (MAE may pretrain at a lower `H` than fine-tuning): sincos regenerates for any grid.

### C3. Slice-position embedding (through-plane)
- **Default:** the continuous normalized position within the series, `z ∈ [0,1]`, through sin-cos features → Linear to `E`. Continuous, because series have 11–320 slices and we sample sparse windows. Optionally add a physical-mm feature (IPP distance from the stack centre).
- **Ablation:** learned bucket embedding.

### C4. View / sequence embedding
- **Default:** a sum of learned embeddings: `plane(3) + fluid(2) + contrast_class(~5) + field_strength(2)`, added to every window token of that series.
- Also a per-series "series index" embedding so two series of the same type stay distinguishable (ablation).

### C5. [CLS] vs pooling
- **SliceViT level:** each 2.5D window → one vector. **Default: GAP over patch tokens** (GAP > CLS when training from scratch [F §3], and MAE-friendly). CLS is an ablation (Dom's list has a CLS token; it stays implemented and switchable).
- **Study level:** **12 finding query tokens** cross-attending over all window tokens (Query2Label/ML-Decoder style [F §3/§10]). This is the ViT form of the per-finding attention-MIL head that public authors call "the part that matters most" [KT].
  - The barebones study `[CLS]` + transformer stays as ablation `study_pool=cls`.
  - The simplest baseline, `attn_mil` (12 gated-attention softmaxes over windows, exactly the public head [KT]), is built first because it's tiny and proven.

### C6. Multi-head self-attention
- **Build order:**
  1. A hand-written version (qkv Linear → heads → `softmax(QKᵀ/√d)` → V → proj), for learning and unit tests.
  2. Then `F.scaled_dot_product_attention` (FlashAttention / mem-efficient kernels), with an allclose test against (1).
- **Options:** QK-norm (a LayerNorm on q, k; cheap stability under bf16) [F §4], on by default (HP); key-padding masks for padded windows; LSA's learned temperature and diagonal mask as an ablation [F §7].

### C7. Transformer block
- **Default:** pre-norm, `LayerNorm(eps=1e-6)`, MLP ratio 4 + GELU, **LayerScale** (init HP ≈ 0.1 at our depth [F §4]), stochastic depth with a linear schedule (max HP, low e.g. 0.1), dropout 0 by default (augmentation carries regularization [F §6]).
- The study-level aggregator uses the same `Block` (a shallow stack, depth HP) + the cross-attention block for the finding queries.

### C8. Initialization
- `trunc_normal_(std=0.02)` for Linear/pos weights, zero bias, LayerNorm weight=1/bias=0.
- **Zero-init the head** so loss at init ≈ 0.693/label.
- Query/CLS tokens get near-zero init [F §5].

### C9. Head + loss
- **Head:** per-finding logits from the finding queries: `logit_k = ⟨q_k_out, w_k⟩ + b_k`, i.e. 12 independent sigmoids.
- **Loss (default):** **confidence-weighted soft BCE** on blended soft labels (weights from B.3) + a small **pairwise rank loss** (weight HP ≈ 0.05–0.12; public stacks use it [KT]), since the metric is AUC.
- **Ablations:** `pos_weight` (clip 1–10); asymmetric loss [F §10]; label smoothing.

### C10. Model size
- Start at **ViT-Tiny/Small scale** (E 192–384, depth 12, heads 3–6: HP) [F §8]. VRAM: micro-batch 1–2 studies × K windows + gradient checkpointing + accumulation.
- `[derived, UNVERIFIED — measure in S03]`: tokens per window at H=256, P=16 = 256 (+0). K=12 training windows per study → ~3k tokens through SliceViT per study, in 12 independent attentions of 256 tokens each.

### C11. Encoder only
Yes. The **MAE decoder** (section D) is a separate, throwaway pretraining module.

### C12. Augmentation (per finding safety)
- **Default:**
  - Small affine: rotate, scale, shift (HP).
  - Mild shared intensity jitter (±~10%), gamma, noise.
  - **Window/slice sampling jitter** (random window centres = the "temporal" augmentation).
  - **Window dropout** and **series dropout**.
- **No LR flips by default:** after laterality canonicalization (A.4b), medial is always on the same image side, which makes medial/lateral easy to learn. A flip would force the model to learn side from anatomy alone.
  - LR flip is anatomically *label-safe* (medial stays medial) [3D §7], so it's kept as an ablation.
  - AP/SI flips and slice reversal of non-LR axes are **never** used.
- **Intensity-gain TTA hurt** (−0.018 LB [KT]); TTA only after OOF proof.

### C13. Efficient attention
- Factorization (C overview) + SDPA are the main tools.
- **ToMe** (token merging, training-free, ~2× throughput for a ~0.2–0.3% drop) is an **inference-time** efficiency lever in S10 [3D §8].
- Windowed/linear attention only if `H` grows (e.g. 384+) or a within-series 3D ablation is tried (cap: one series, ~7.7k tokens, windowed) [3D §8].

### C14. Quantization / precision
- **Training:** bf16 autocast (decision 2026-09-24).
- **Inference:** fp16 on the T4 (no bf16 convs/tensor cores on Turing [W], [KT]), with an fp32 retry per study.
- Then evaluate int8 PTQ, ONNX/TensorRT (TensorRT has an RSNA precedent [W]) *after* accuracy is set, measuring AUC and seconds together.

### C15. Optimizer + schedule
- **Pretraining (MAE):** AdamW, warmup + cosine.
- **Fine-tuning:** AdamW + **layer-wise LR decay** (HP ≈ 0.65–0.75 [SSL §5]), higher LR on the head/aggregator, warmup + cosine, grad clipping (HP ≈ 1.0), weight decay off for norm/bias/pos/tokens.
- **EMA of weights** or top-k epoch averaging (public: top-3 epoch mean/SWA [KT]).

### Track B build order (S03)
1. `PatchEmbed` (patchify + conv stem)
2. `pos_embed`
3. `attention` (manual → SDPA)
4. `Mlp`
5. `Block`
6. `SliceViT`
7. heads (`attn_mil`, finding-queries)
8. `KneeViT`
9. unit tests
10. overfit-one-batch
11. MAE wrapper

## D. Self-supervised pretraining (MAE): the key to a competitive scratch ViT

- **Why:** ViTs lack the inductive bias for our data size [F §7]. Self pre-training on the target corpus beats scratch and often ImageNet transfer in medical imaging [SSL §3]. The best MAE-vs-scratch number found was +16.9% AUC (brain MRI, single GPU) [SSL §3]. MAE is cheap: the encoder sees only ~25% of tokens.
- **Data:** every 2.5D window from every series of all 4,407 studies (plus test? **No**: only train). That's ~800k slices ≫ the "225k" the SSL report assumed. Uses the S02 cache, so there's no DICOM decoding during pretraining.
- **Default recipe:**
  - 2D MAE on 2.5D windows (C=3), mask ratio 0.75 (HP)
  - per-patch normalized-pixel loss
  - lightweight decoder (depth/width HP)
  - AdamW, warmup + cosine
  - bf16, checkpoint every ≤30 min, resumable
- **Stage 2 (optional):** tube masking across adjacent slices at 0.9 (VideoMAE [SSL §2]).
- **Budget:** pilot 1 epoch → extrapolate. The planning estimate is 1–3 days on the 5070 [SSL §5, derived]. GPU calendar in the master plan.
- **Monitoring:** reconstruction loss plus a **linear/kNN probe every N epochs** on report labels (fold 0), so we know whether the features are getting better, not just the loss.
- **Transfer:** load the encoder into `SliceViT`; the random-init aggregator + head is fine-tuned with LLRD.
- **Must keep a random-init control** (same fine-tuning, no MAE) so the write-up can quantify what MAE bought.
- No one has published a knee self-supervised or random-init result [KT], so Track B is genuinely novel.

## E. Track A: pretrained backbone (safety net)

- **Default:** the proven recipe [KT]. `coatnet_rmlp_2_rw_384` (ImageNet-12k→1k, timm) on 2.5D windows at 384 px, per-finding attention-MIL head.
  - Training: soft BCE (+ rank loss), AdamW with backbone LR ≪ head LR, K≈12 random windows per study, all windows at eval, top-3 epoch averaging.
  - Public compute bar: ~3 h on a 4090 at bs 8 with grad checkpointing (23.7 GB). On the 12 GB 5070 → bs 1–2 + accumulation; expect roughly 2–3× longer (UNVERIFIED).
- Our edge over the public copy:
  - our own blended labels
  - denser/wider sampling
  - laterality canonicalization
  - honest grouped CV
- **Diversity arm later (S08):** a second family (e.g. ConvNeXt/EfficientNetV2), only if the OOF rank-blend gain is real; the public result was ~+0.001 LB from backbone diversity.
- **License note:** RadImageNet weights are CC-BY-NC-SA-4.0; DINOv3 has its own license. Pretrained-weight licenses are exempt from relicensing, but prefer ImageNet timm weights (clean).

## F. Inference + efficiency

- **Notebook (S06):**
  - Offline wheels dataset (pylibjpeg, gdcm, timm if Track A).
  - Weights as a private Kaggle dataset; made public only if we're a winner (obligation).
  - The **same preprocessing module** as training, run with a per-study try/except and a prior fallback.
  - Arms run sequentially, no DataParallel.
  - Never walk the competition dir to find weights.
  - Header parsing overlapped with GPU compute (DataLoader workers).
  - Write `submission.csv` in the exact column order; assert on row identity.
- **Runtime budget** (to measure): public single CoAtNet ≈ 1.1 s/study at 62 windows [KT]; header reading and slice ordering ≈ 12 of 32 minutes [KT]. **Optimize the DICOM header pass first.**
- **Efficiency submission (decision: 1 of the 2 finals):** a single fast model, fp16, reduced windows. The efficiency score is `AUC/(0.5−maxAUC) + t/32400` (verified [KF]): with maxAUC ≈ 0.96 → Δscore ≈ −2.17·ΔAUC + ΔT/32400. `[derived]` **So 0.001 AUC ≈ 70 s of runtime.**
  - A 10-minute saving is worth ~0.009 AUC. Both matter; profile both.

## G. Engineering conventions (every session)

- **Layout:** `src/knee/{audit,data,labels,model,train,infer,utils}`, `configs/*.yaml` (one per run), `tests/` (pytest; CPU-runnable, small).
- **Reproducibility:** seed everything, log the git sha + config + data/label versions in each run's output dir and in `docs/experiments.md`.
- **Windows:** `if __name__ == "__main__":` in all entry points, spawn-safe datasets, `pathlib` paths, and no hard-coded drive letters (use `configs/paths.yaml`, gitignored, with a committed `paths.example.yaml`).
- **Secrets:** `.env` (ANTHROPIC_API_KEY), gitignored. Kaggle creds stay in the user profile.
- **Licenses (winner obligations):** our code must be under CC-BY-NC 4.0 at the end. Only use OSI-licensed code allowing commercial use (timm Apache-2.0, pydicom MIT: verify each in S12).

## H. Gap log

### Answered by Dom in S00 → see `docs/decisions.md`
| Gap | Answer |
|---|---|
| Who writes the code | Claude writes, Dom reviews |
| Pretrained fallback timing | Parallel Track A from the start |
| Where sessions run / sync | Windows box; private GitHub repo |
| GPU availability | 24/7 until Oct 22 |
| Use of the 58 gold studies | Reports first; calibration + OOF only; the 58-teacher idea becomes the noisy-student round (S09) |
| Label sources | Own LLM reader + vetted public sets |
| LLM reader | Claude API, ≤ $50 |
| "From scratch" definition | Own code + own weights (random init + MAE on competition data) |
| Final 2 submissions | 1 accuracy + 1 efficiency |
| Teammates | Solo |
| Training precision | bf16 from day one (supersedes "start in fp32" for training) |
| Slice input | C=3 adjacent slices default |

### CLAUDE.md facts corrected by evidence (S00 updated CLAUDE.md; S01 must verify on the real files)
- Train studies **4,407** (not 7,504). There are **22** credited sites (19+3), not "16".
- `Fluid_Sensitive` ≡ `Fat_Suppression` on all train rows.
- Transfer syntaxes: all sampled files are uncompressed; the Data page's 4-syntax list is unconfirmed in practice.
- Orientation/position/spacing tags **exist**; PatientSex exists; PatientAge not seen.
- The host allows commercial LLM APIs for labeling. DUA-gated external datasets (OAI, MRNet, fastMRI, KneeCoT) are effectively not allowed.

### Still open (owner → when)
1. Test-set DICOMs: any compressed syntaxes? Is Fluid ≡ FatSup on test too? → S06 (inspect inside a Kaggle notebook).
2. Public/private LB split %: not stated (forum thread unanswered) → re-check at S08/S11.
3. Whether test labels come from the same image readers as the gold 58 → UNVERIFIED; affects how much to trust gold-calibration.
4. How the public "Raptor" SSL ViT was trained (forum 737696): worth reading in S03 before launching MAE.
5. Exact Kaggle scoring GPU/CPU/RAM (T4×2, ~29 GB RAM, 4 CPUs per secondary sources [W]) → S06 prints `nvidia-smi`/`nproc`/`free` in the notebook.
6. Oct 8 gate threshold: placeholder "within ~0.01 of Track A OOF" → Dom confirms at S08.
7. Public label-set licenses → S04.
8. Effusion/graded-finding threshold → fitted on gold OOF in S04 (a calibration, not a hand rule), reviewed by Dom.
