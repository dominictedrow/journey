# RSNA Knee Abnormality Detection — Project Guide

Kaggle code competition: predict 12 knee-MRI findings per study. Owner: Dom.
Goal: **1st place**, plus the efficiency prize as a stretch. Secondary goal: learn by building a ViT from scratch.

**New session? Read `HANDOFF.md` first.** It has the reading order, current state, next session, and the session protocol. The plan lives in `docs/plan/`, decisions in `docs/decisions.md`.

## Working rules for Claude in this repo
- **Never guess.** If a fact about the competition, data, DICOM behavior, or a library is uncertain, either research it (cite the source) or ask Dom. Mark anything unverified as `UNVERIFIED` in code comments/docs.
- Facts below are tagged **[Kaggle]** (read directly from the official Overview/Data/Rules pages on 2026-09-23), **[3rd-party]** (news/GitHub, not confirmed), or **[Dom]** (Dom's decision/opinion).
- Re-check the Kaggle Overview/Rules pages before any decision that depends on deadlines, scoring, or eligibility; organizers may update them. Official URL: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection (the Rules page lists a slightly different slug, `...abnormalities-detection`; the one above is the one that loads).
- Git/GitHub operations: follow Dom's global checklist (specific files only, no credentials, no force-push).

## Competition facts
**Host/format [Kaggle]:** RSNA, Research Code Competition. Submissions run as Kaggle Notebooks; internet disabled; CPU or GPU notebook <= 9 hours; output must be named `submission.csv`. Freely and publicly available external data and pretrained models are allowed.

**Timeline [Kaggle]** (all 11:59 PM UTC):
| Date | Event |
|---|---|
| Jul 30, 2026 | Start |
| **Oct 15, 2026** | Entry deadline (must accept rules) and team-merger deadline |
| **Oct 22, 2026** | Final submission deadline |
| Nov 5, 2026 | Winners' requirement deadline (training code, video, method description) |
Recognition event: RSNA 2026, Chicago, Nov 29 – Dec 3 [3rd-party].

**Rules that matter [Kaggle Rules]:**
- Team max 5; merging allowed until Oct 15 (combined submission count must fit the per-day allowance).
- **5 submissions/day; select up to 2 final submissions.** One Kaggle account per person; no private code sharing outside the team (public sharing on the competition forum/notebooks is fine).
- Winners are decided solely by the **Private Leaderboard**.
- Data: MIRA license (http://rsna.org/mira-license). Do not redistribute the data to non-participants.
- **Winner obligations:** deliver training code, inference code, model weights (as a **public Kaggle dataset**), and environment description (requirements.txt + Kaggle image, or Dockerfile); license the solution under **CC-BY-NC 4.0**; write a reproducible method description; make a short video; post code + weights link on the forum; publish final model for open distribution. Any open-source code used must be under an OSI-approved license permitting commercial use (pretrained weights with incompatible licenses are exempt from the relicensing).
- No hand-labeling or human prediction of validation/test data.
- External data/models allowed if publicly available, reasonably accessible, and minimal cost.

**Prizes [Kaggle]:** $77,000 total.
- Leaderboard: 1st $9,000; 2nd $7,000; 3rd $6,500; 4th $6,000; 5th $5,500; 6th–10th $5,000 each (**top 10 paid**).
- Efficiency Track: 1st $7,000; 2nd $6,000; 3rd $5,000.

## Metric and submission format [Kaggle]
- Score = **macro-averaged AUC-ROC** over the 12 targets (each label weighted equally regardless of prevalence).
- Columns (in this order): `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`. One row per test study; values are confidence scores (probabilities). `sample_submission.csv` sets every value to 0.5.
- Because AUC is rank-based, calibration doesn't matter, but every label needs a meaningful ranking across studies.

## Efficiency Track [Kaggle]
Eligibility: submission must be one of the team's selected final submissions and rank above the `sample_submission.csv` benchmark on the private leaderboard. A submission can win both a leaderboard and an efficiency prize.

Official formula (raw TeX from the page):
`Efficiency = AUC / (Benchmark - maxAUC) + RuntimeSeconds / 32400` — **lower is better**
- `AUC`: the submission's private score. `Benchmark`: score of `sample_submission.csv` (expected 0.5 for constant predictions; UNVERIFIED). `maxAUC`: best private-LB AUC of all submissions. `RuntimeSeconds`: seconds to evaluate the submission. 32400 s = 9 h.
- Since Benchmark < maxAUC the first term is negative; higher AUC lowers the score. The runtime term is 1.0 at the full 9-hour limit, so it is linear in inference wall-clock time.
- **Implication [derived from formula]:** "efficiency" here means end-to-end *inference-notebook runtime* (DICOM decoding + preprocessing + model), not parameter count. Reducing preprocessing and ensemble size matters as much as model size. Tradeoff between AUC gain and seconds is set by `(Benchmark - maxAUC)` (~0.3–0.45 if maxAUC is 0.8–0.95; UNVERIFIED until the leaderboard settles).
- Organizers say more details may come via the discussion forum — check it.

## Data [Kaggle Data page]
- 819,640 files, **569.76 GB**, DICOM + CSV. Test set has **~1,300 studies**; the example `test.csv`/`test_series/` are replaced with the real test data at scoring. **The `Report` column is NOT available at test time.** Prevalence may differ between train / public LB / private LB.
- Hierarchy: `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm` (one .dcm = one slice). Series typically 20–45 slices (median 30), long tail to a few hundred.
- `train.csv`: `StudyInstanceUID`, `Report` (free text, several languages), 12 binary labels — but **only a small subset of training studies carry labels**; the rest must be derived from `Report`.
- `train_series.csv`: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive` (T2/PD/STIR-like), `Fat_Suppression`, `Anatomical_Plane` (Sagittal/Coronal/Axial). ~~Fluid_Sensitive and Fat_Suppression are correlated but not equivalent.~~ **Correction (S00, public notebooks [3rd-party], verify in S01): they are identical on all 24,371 train series rows (1 bit). Derive the contrast class from TR/TE/InversionTime instead.** Same schema for `test_series.csv`, so these are usable model inputs at test time.
- Twelve targets: ACL injury, MCL injury, Medial Meniscus tear, Lateral Meniscus tear, Medial OA, Lateral OA, PF (patellofemoral) OA, Effusion, Synovitis, Baker's cyst, Contusion, Fracture.
- DICOM notes: intensities, orientations, resolutions vary; mixed transfer syntaxes (Explicit VR Little Endian uncompressed, JPEG Lossless, JPEG 2000, Implicit VR Little Endian); **every DICOM is stripped to an allowlisted set of 86 metadata tags**. Public notebooks [3rd-party, verify S01]: every sampled train file is Explicit VR LE (uncompressed); IOP, IPP, PixelSpacing, SliceThickness, Rescale slope/intercept, TR/TE/TI, Manufacturer, FieldStrength, PatientSex (94.6%), Laterality (~50% missing) are present; PatientAge and overlays not seen; no MONOCHROME1. Filename order ≠ physical order: **sort by IPP·normal**. Details: `docs/kaggle/data_and_dicom_from_notebooks.md`.
- **Verified-label studies: 58** (confirmed by Dom and the forum; image-read by two readers + an adjudicator, not derived from the reports; **not representative**: every gold study has ≥1 positive, ACL 41% vs ~20% corpus-wide). **Total training studies: 4,407** (forum + many notebooks, 3rd-party; the old "7,504" was wrong; verify from `train.csv` in S01). 24,371 series, ~819k slice files. The Acknowledgements page credits 22 sites (19+3).
- Reports: ~9 languages (~60% non-English); 46 report texts are shared across 177 studies, so **group folds by report hash**. Host confirmed (forum 733965) that sending report text to commercial LLM APIs for labeling is allowed. DUA-gated external knee datasets (OAI, MRNet, fastMRI, KneeCoT) are effectively not allowed.
- Access: rules must be accepted (signed in) to see/download data.

## Compute [Dom]
- Training target: Dom's **Windows box**: **16 GB RAM**, **RTX 5070 with ~12 GB VRAM**, and an **Intel Core i5** CPU (with **16 cores** per Dom's recollection (`UNVERIFIED: could be 16 logical threads rather than 16 physical cores; TODO confirm exact model`)).
- **Data** (570 GB) is on an external **2 TB Samsung 870 QVO SSD (SATA), formatted NTFS** (per Dom; not currently mounted on the Mac, so unverified from here). NTFS is readable on the Windows box; macOS mounts NTFS read-only by default.
- **Project/code drive:** `/Volumes/DOMS_DRIVE` is a separate ~245 GB drive, formatted **FAT32** (verified via `diskutil`), 166 GiB free. It holds this repo only. **FAT32 has a 4 GB per-file limit and no permissions/symlinks** — do not store the dataset, large caches, or checkpoints > 4 GB there.
- **Constraints this implies [derived]:** 16 GB RAM means the dataloader must stream from disk and cannot hold volumes in memory; 12 GB VRAM limits batch size/token count (use mixed precision, gradient accumulation, gradient checkpointing); SATA SSD read speed will bottleneck DICOM loading, so cache preprocessed volumes to compact files early (ideally on an internal drive). Decoding JPEG-Lossless/JPEG2000 on the CPU may be the real training bottleneck — measure it.
- Inference must fit a Kaggle notebook (9 h cap; Kaggle GPU type/quota: `TODO verify`).
- Dev machine is a Mac (paths under `/Volumes/DOMS_DRIVE/...`); anything CUDA-specific must not be assumed to run there.

## Strategy [Dom]
**Architecture: Vision Transformer, built from scratch first, with a fixed fallback.**
- Reasoning: best performance/consistency among 3D CNN / LSTM / GRU / ViT candidates in Dom's view; LSTM/GRU are inefficient and slow on this much data; ViT is the least familiar, so highest learning value; building it also produces the strongest write-up and portfolio piece. Jason, a mentor who previously worked with medical data, may give occasional guidance (not a team member).
- Open-source ViTs (and anything else public) may be used for design inspiration.
- **Decision date: Oct 8, 2026 gate.** **Updated 2026-09-24:** a pretrained-backbone track (Track A) now runs **in parallel from the start** on the shared pipeline and gives the early valid submission. Track B = Dom's ViT: own code + own weights (random init → MAE self-pretraining on competition slices only). At the Oct 8 gate Dom decides B's role (see `docs/plan/00_MASTER_PLAN.md` §5).
- Fine-tuning is explicitly allowed by the rules and stays the fallback; it also teaches working with pretrained models.

**Labeling plan [Dom], REORDERED 2026-09-24 on evidence (see `docs/decisions.md`):** report-derived labels (our Claude API reader + vetted public LLM label sets, blended) are now **primary**; the 58 gold fit only a tiny OOF calibration + validation; Dom's teacher idea runs later as a noisy-student round (S09). A 58-study image model scored 0.66 macro AUC vs 0.87–0.89 for LLM report labels. Original plan kept below for the record:
1. **First: Dom's own idea** — fine-tune a labeler on the small verified-label subset, use it to pseudo-label the remaining studies, then train the ViT on the enlarged set. Known risks Dom flagged: needs something close to the model being built; verified labels might contain errors; small-n overfit. Mitigation: hold out part of the verified set purely for evaluating the labeler.
2. **Then (fallback / cross-check): report-derived labels** — extract the 12 labels from the multilingual `Report` text (rules and/or an LLM) and validate against the verified subset. This is the path the organizers point to. (A third-party README claims "label quality is the competition" — treat as opinion.)
3. Reports don't exist at test time, so both approaches must yield a model that works from images (+ series metadata) alone.

**Efficiency [Dom]:** plan for PyTorch efficiency features (mixed precision, quantization, `torch.compile`, fast DICOM decoding). **Updated 2026-09-24: train with bf16 autocast from day one** (12 GB VRAM); "get accuracy first, then shrink" now applies to inference (fp16 on the T4 → quantization). Final 2 submissions = 1 accuracy + 1 efficiency. Efficiency score is dominated by inference runtime, so budget preprocessing time too.

## DICOM preprocessing checklist [Dom's list, annotated]
Goal: every series becomes a consistent float32 tensor with metadata preserved.
1. **Decode by transfer syntax** — handle the four syntaxes above (JPEG Lossless / JPEG 2000 need extra decoders, e.g. pylibjpeg/GDCM; `TODO verify installable offline in the Kaggle notebook`). Also verify no non-DICOM/corrupt files exist in the folders.
2. Pixel type: cast everything to float32; apply rescale slope/intercept if present (`TODO check which of the 86 tags exist`).
3. Normalize intensities **per series** (MRI intensities are not standardized across scanners); options to try: z-score, percentile clip + scale. Knob to tune.
4. Orientation: standardize in-plane orientation and slice order using orientation/position tags **if they survive the 86-tag allowlist** (`TODO verify`); otherwise rely on `Anatomical_Plane` from `train_series.csv`.
5. Resample: pixel spacing, slice spacing/thickness, in-plane size, and series length to a common grid (or fixed number of slices). Requires spacing tags — verify availability first.
6. Overlays: check whether overlay data exists in the files (`TODO verify`) and remove/ignore it.
7. Private tags: **already stripped by the host** (allowlist of 86). Dom's "identify and explain private elements" item likely reduces to documenting the 86 kept tags — list them in `docs/dicom_tags.md`.
8. Organize by study → series → instance; sort slices by position (not filename) before stacking.
9. Sequence type: derive from `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`; give it to the model as a conditioning input.
10. Grayscale inversion: **do not invert by default.** Dom's rule: only invert if it's within the same scan type, since fat-suppressed and PD-FS signal differ and inversion could destroy information. Treat as a hyperparameter/ablation.
11. Multi-view input: stack planes/sequences as extra channels vs. separate token streams — ablation. Single channel per slice (grayscale).
12. Metadata: encode + normalize series-level metadata (plane, fluid, fat-sup; demographics **only if they exist in the 86 tags — verify first**, then decide).
13. Output: tensors → batches; disjoint train/val/test splits **by study** (never split slices/series of one study across sets; also consider grouping by source site if it can be inferred, since the test set spans institutions).
14. Optionally cache preprocessed volumes to disk in a compact format (e.g. uint16/npy or PNG) to speed training; keep the exact same preprocessing code for inference.

## Model component notes [Dom's brainstorm, with corrections]
Confirmed-standard pieces to plan for:
- Patch embedding → learned embedding matrix; positional encoding (learned or sinusoidal; in 3D/multi-slice: patch position + slice position, and likely a view/sequence embedding); a learned `[CLS]` token whose final state feeds the classifier.
- Multi-head self-attention in pre-norm transformer blocks with residual connections; MLP with GELU; **LayerNorm** (BatchNorm is not the standard in transformers). Dropout / stochastic depth as regularization.
- **Encoder only.** This is classification, so no decoder is needed (the "encoder-decoder" idea from the brainstorm applies to seq2seq tasks).
- Head: linear layer to **12 independent logits with sigmoid + BCE** (multi-label, not softmax) — "one-hot" encoding is for mutually exclusive classes; here labels are 12 binary flags.
- Optimizer: Adam/**AdamW** (weight decay replaces explicit L2), LR warmup + cosine schedule, gradient clipping. Pos-weight or class-balanced sampling for rare labels.
- Augmentation: spatial (flip only if laterality/anatomy makes it safe — `TODO verify per finding`), intensity, temporal/slice-dropout ("temporal" = slice-order axis).
- Efficient attention variants (windowed, linear, factorized space/slice attention) are worth considering because token count grows fast with volumes — open design question.
- Quantization = storing/computing weights and activations in lower precision (e.g. int8/fp16) to cut memory and runtime; do it post-training after accuracy is established.
- Combining multiple sequences into one feature map / extra channels vs. token sequence: to be decided by ablation.

## Repo layout (proposed; nothing built yet)
```
data/          # symlink or path config only; raw data is NOT committed (MIRA license, 570 GB)
docs/          # dicom_tags.md, decisions, experiment log
src/           # preprocessing, dataset/dataloader, model, train, infer
notebooks/     # Kaggle inference notebook + EDA
```
Current state: see `HANDOFF.md` §4. Private GitHub remote: `dominictedrow/knees`. Code sessions run on the Windows box; Claude writes the code, Dom reviews.

## Open questions (ask Dom / research before acting)
1. Exact i5 model and whether "16" means physical cores or threads (check Task Manager → Performance → CPU: Cores vs Logical processors) (matters for DICOM decoding throughput), and whether the Windows box has a faster internal drive for caching preprocessed volumes.
2. Total train studies from `train.csv` (verified-label count is settled at 58; public evidence says 4,407; confirm in S01).
3. Which of the 86 DICOM tags exist (orientation, spacing, slope/intercept, demographics)?
4. Kaggle notebook GPU type and weekly GPU quota, and whether `RuntimeSeconds` counts the full notebook run.
5. ~~Fallback decision date and teammates~~ → resolved 2026-09-24: Oct 8 gate with a parallel Track A; **solo**.
6. Remaining open items are tracked in `docs/plan/02_vit_detailed_plan.md` §H.

Resolved: Dom's "injury type encoding" idea is dropped (it is essentially the competition task itself). Jason is an informal mentor with past medical-data experience who may give occasional guidance; he is not a team member.
