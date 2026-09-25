# RSNA Kaggle Winners & Efficiency Research

Researched 2026-09-24 for the RSNA Knee Abnormality Detection (2026) project. Sources are primary
(Kaggle competition writeup pages and winners' GitHub repos, Kaggle product-feedback/docs threads,
NVIDIA/pydicom/PyTorch docs) unless marked **[3rd-party]** (blogs/aggregators) or **UNVERIFIED**
(claim could not be confirmed from a fetched primary source in this session). Kaggle "writeups" pages
are a JS single-page app; direct `WebFetch` on `kaggle.com/.../writeups/...` returned only the page
title with no body text, so those pages were read via the `r.jina.ai` text-extraction proxy
(`https://r.jina.ai/<url>`) — noted per citation. Kaggle's own `kaggle.com/docs/*` pages had the same
JS-rendering problem; GPU/CPU/RAM numbers below are cross-checked across 3+ independent secondary
sources instead and flagged accordingly.

---

## Part A — What won past RSNA Kaggle competitions

### 1. RSNA 2024 Lumbar Spine Degenerative Classification (closest analogue: multi-series knee-like MRI)

This is the best available analogue to our knee task: multiple MRI series (sagittal T1, sagittal T2,
axial T2) per study, multiple findings per study, no bounding boxes given at test time, scored by a
leaderboard metric per-label.

**1st place — team "Avengers"**
(source: Kaggle writeup, read via r.jina.ai proxy — https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)

- **Two-stage pipeline**: Stage 1 localizes vertebral-level coordinates (produces something like the
  competition's own `train_label_coordinates.csv`); Stage 2 classifies severity per finding using crops
  around those coordinates.
- Stage 1: a 3D ConvNeXt predicts which slice ("instance number") each vertebral level appears on
  (dual-task: L1 regression + cross-entropy classification over a 32-slice-padded volume), then a 2D
  encoder (ConvNeXt-base and EfficientNetV2-L, ensembled by averaging) with **level-separated heads**
  regresses (x, y) coordinates on a single 512×512 sagittal slice picked by median instance number.
- Stage 2 (severity classification) is **task-specific per finding**, not one shared multi-series model:
  central spinal canal stenosis used sagittal T2 + axial T2; foraminal stenosis used sagittal T1 + axial
  T2; subarticular stenosis used axial T2 only. Each used a 5-slice crop (center slice ±2) around the
  Stage-1 coordinate.
- Classification architecture: CNN backbone (**ConvNeXt-small** and **EfficientNetV2-S**, ensembled) +
  **bidirectional LSTM + attention-based multiple-instance-learning (MIL) pooling** head. Attention-MIL
  alone was worth +0.02, adding bi-LSTM/aux losses/ensembling another +0.02 (final public LB 0.33).
- Augmentation: random ±10 px coordinate jitter and random ±2-slice instance-number jitter *before*
  cropping (explicitly to make the classifier robust to Stage-1 localization error — "random shift of
  instance_number was crucial for robustness"), then RandomBrightnessContrast + ShiftScaleRotate
  (±20° rotation) after cropping.
- CV: 5-fold for the localization models; ensembling by median (instance numbers) / mean (coordinates,
  cross-architecture classification logits).
- **Explicitly tried and rejected**: Mamba and plain self-attention (worse than bi-LSTM), and **Vision
  Transformers** — "convolution models were better than vits." Smaller backbones beat larger ones
  (ConvNeXt-small > base > large).
- Trained on Google Colab (T4, high-RAM instance); no TTA mentioned; no explicit inference-runtime
  number reported.

**2nd place — ianpan / Kevin / Yuji / Bartley**
(source: Kaggle writeup via r.jina.ai — https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution;
corroborated by GitHub — https://github.com/brendanartley/RSNA-2024-Competition)

- Also **two-stage**: Stage 1a slice localization (4 separate models, one per finding-family: foraminal
  on sagittal T1, spinal on sagittal T2, subarticular on axial T2, spinal-axial on axial T2) using
  CNN-transformer hybrid heads predicting distance-to-target + level classification; Stage 1b regresses
  precise (x, y) keypoints on the located slice.
- Stage 2 (grading): for each target, generates **27 augmented crops** (±1 slice × 8 in-plane spatial
  offsets), 3-channel (or 5-channel for subarticular) crops, fed to a variety of 2D CNN / attention-pooling
  / MIL backbones: **MaxViT-tiny-512, CoAtNet-1-224, NFNet-0, CSN-ResNet101** (note: MaxViT/CoAtNet are
  CNN-transformer *hybrids*, not pure ViTs). The 27-crop predictions are aggregated by median/mean —
  effectively heavy TTA baked into training and inference.
- Multiple axial series per study were handled by letting the subarticular localization model tag which
  vertebral level each slice belongs to, so duplicate/extra series each still contribute usable crops
  instead of being ignored or naively concatenated.
- Ensembling: 4 model groups combined with fixed weights (sagittal spinal weight 1.0, axial spinal
  weight 0.5, etc.). CV 0.3805, public LB ~0.35.
- Separate GitHub write-up (brendanartley) confirms: T1 pipeline used the **middle 24 frames**
  through encoder → LSTM → attention pooling with manifold mixup and 9-rotation TTA; T2 pipeline varied
  frame count (5–16) and image size deliberately "for prediction diversity"; training combined
  ground-truth loss and pseudo-label loss `0.5*loss_gt + 0.5*loss_pseudo` with confident-learning-based
  label denoising via an out-of-fold ensemble. Reported hardware: single GPU ≥34 GB VRAM (A100-40GB
  tested), ~24 h training time (training only, not the scored inference).

**Silver-medal repos found but not deep-dived** (noted for completeness, not verified in depth):
GitHub `DaoyuanLi2816/Kaggle-RSNA-2024-Lumbar-Spine-Degenerative-Classification-Silver-Medal`,
`nshen7/rsna-2024`, `hhcgoodluck/Kaggle-RSNA-2024-Lumbar-Spine-Degenerative-Classification`,
`dwest1507/RSNA-2024-Lumbar-Spine-Degenerative-Classification` — UNVERIFIED beyond their existence.

### 2. RSNA 2023 Abdominal Trauma Detection (CT, study-level labels)

**1st place — Team Oxygen (Nischaydnk et al.)**
(source: GitHub README — https://github.com/Nischaydnk/RSNA-2023-1st-place-solution; Kaggle writeup page
returned empty body via both WebFetch and r.jina.ai in this session, so relying on the GitHub README)

- **Three-part pipeline**: (1) 3D segmentation to generate organ masks/crops, (2) 2D CNN + GRU for
  kidney/liver/spleen/bowel, (3) 2D CNN + GRU for bowel + extravasation.
- All classification models trained at **384×384**. Backbones: **CoaT-Lite Medium**, **CoaT-Lite Small**,
  **EfficientNetV2-S**, each + GRU (via `timm`).
- Each study's volume was resampled to **96 equidistant slices**, reshaped into 2.5D triplets:
  `(32, 3, H, W)` — i.e. 32 "super-slices," each a 3-channel stack of 3 adjacent real slices — batched as
  `(2, 32, 3, 384, 384)`. This is an explicit, fixed compute budget per study regardless of original
  slice count.
- **Soft labels**: classification targets multiplied by an organ-visibility score derived from the
  segmentation mask coverage per slice, so slices where an organ is barely visible contribute a weaker
  label signal.
- **Auxiliary segmentation loss** on the last two encoder blocks (weight 0.125) stabilized training and
  added "+0.01 to +0.03" score improvement — multi-task regularization, not just a pure classifier.
- Augmentations: Perspective, HorizontalFlip, VerticalFlip, Rotate(±25°), all p=0.5.
- CV: 4-fold **GroupKFold at patient level** (matches our own "split by study" requirement).
- Ensembling: slice-level aggregation within folds, then cross-architecture ensembling by max
  aggregation. Best ensemble OOF (log-loss, lower better) 0.31 vs best single model 0.326.

### 3. RSNA 2022 Cervical Spine Fracture Detection (CT)

**1st place — Qishen Ha**
(source: Kaggle writeup via r.jina.ai — https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution)

- **Two-stage**: Stage 1 is **3D semantic segmentation** (UNet decoder over ResNet18d / EfficientNetV2-S
  encoders) on 128×128×128 voxel volumes, trained on only 87 samples that had 3D masks, 5-fold CV, to
  segment vertebrae C1–C7 (7-channel output).
- Stage 2 is **2.5D CNN + LSTM classification**: per vertebra, extract 15 slices (±2 adjacent-slice
  context) → 5-channel image stack + the Stage-1 predicted mask as a 6th channel. Two model families:
  vertebra-level (EfficientNetV2-S @512, ConvNeXt-Tiny @384, each + LSTM over the 15 slices, 5-fold) and
  patient-level (processes all 7 vertebrae × 15 slices at once; ConvNeXt-Nano/Pico/Tiny, NFNet-L0,
  2–5 fold).
- Final ensemble: **10 models** (5 segmentation + 5 classification). CNN-based throughout — **no
  transformers** used or mentioned. Inference had to fit a **7.5-hour submission window** — the closest
  direct precedent we found for operating under a hard Kaggle-notebook runtime ceiling similar to our
  9-hour cap.

### 4. RSNA Screening Mammography Breast Cancer Detection (2023)

**1st place — "mr.robot" (Đăng Nguyễn Hồng)**
(sources: Kaggle writeup via r.jina.ai — https://www.kaggle.com/competitions/rsna-breast-cancer-detection/writeups/mr-robot-1st-place-solution;
GitHub — https://github.com/dangnh0611/kaggle_rsna_breast_cancer)

- 2D mammograms (not volumetric), so less directly transferable on the modeling side, but the
  **inference-engineering** choices are directly relevant to our Efficiency Prize:
- Pipeline: DICOM → decode → **YOLOX-nano @416×416** breast-ROI detector → crop → normalize → save as
  8-bit PNG → **4× ConvNeXt-small** classifiers (one per fold, 4-fold CV).
- **All 4 folds were combined into a single TensorRT engine for inference** — the whole ensemble runs as
  one compiled engine rather than 4 separate PyTorch forward passes. Reported TensorRT conversion time:
  **~5–10 minutes on Kaggle's P100**, versus ~1 hour on an A100 (the discrepancy is plausible but the
  underlying cause, e.g. TensorRT/driver version differences, was not explained in the source —
  UNVERIFIED reason).
- Used several public external mammography datasets (BMCD, CMMD, CDD-CESM, MiniDDSM, VinDr) which
  reportedly improved score by ~0.02 — direct precedent for "publicly available external data allowed and
  used by winners," consistent with our competition's rules.

---

## Part A — Recurring patterns across RSNA imaging winners

1. **Two-stage localize-then-classify is the dominant winning architecture** across every RSNA MRI/CT
   competition surveyed (both lumbar spine top-2, cervical spine, abdominal trauma). Stage 1 finds
   *where* in the volume/series the relevant anatomy is (coordinates, slice index, or a segmentation
   mask); Stage 2 classifies a tightly bounded crop/slice-window around that location. None of the
   winners fed a whole raw variable-length volume straight into one end-to-end classifier.
2. **2.5D is the standard token-shape**: stack a small, fixed number of neighboring slices as channels
   (3, 5, 6 channels) or as a short sequence (15, 24, 27, 32 elements) rather than true 3D convolutions
   or unbounded-length sequences. Every winner fixed the per-study compute budget explicitly (96
   equidistant slices; 15 slices/vertebra; 5-slice crops; 24 or 27 frames).
3. **CNN backbone + GRU/LSTM/attention-MIL over the slice axis** is the standard "sequence model" choice
   — not full 3D CNNs, not from-scratch transformers. Backbones seen across all four competitions:
   ConvNeXt (nano/pico/tiny/small/base), EfficientNet/EfficientNetV2 (S/L), NFNet, CoaT-Lite, ResNet18d —
   all ImageNet-pretrained CNNs. Where hybrid attention/CNN backbones appeared (MaxViT-tiny, CoAtNet-1)
   they were used as drop-in `timm` backbones inside the same 2.5D+pooling framework, not as the primary
   architectural bet.
4. **No top-2/1st-place solution found here trained a from-scratch Vision Transformer.** The one team
   that explicitly tested ViTs (lumbar spine 1st place) found plain CNNs and CNN+bi-LSTM/MIL outperformed
   them, and also found Mamba/self-attention pooling underperformed bi-LSTM pooling. This is a direct,
   named data point against the from-scratch-ViT bet on a dataset this size, and reinforces the value of
   Dom's fixed fallback date and decision criterion.
5. **Heavy multi-crop/multi-offset TTA baked into training itself** (27 crops per target in lumbar-spine
   2nd place; 9-rotation TTA in the brendanartley T1 pipeline) rather than TTA bolted on only at
   inference — worth planning for from the start rather than as an afterthought.
6. **Explicit runtime engineering only shows up where the competition format forces it** (code
   competitions with a notebook time cap): TensorRT engine compilation (mammography), fixed small
   slice/vertebra budgets (abdominal trauma, cervical spine) that bound compute independent of study
   size. This confirms bounding compute per-study is both a modeling and a runtime-budget decision.
7. **Multiple series/sequences (sagittal T1, sagittal T2, axial T2) were handled as separate,
   finding-specific input streams**, not naively concatenated or channel-stacked across incompatible
   sequence types — directly informs the "multi-view input: stack vs. separate streams" open ablation
   already flagged in the project's CLAUDE.md.

---

## Part B — Inference/runtime efficiency on Kaggle

### 5. Kaggle notebook hardware, quota, and CPU/RAM (2025–2026)

Kaggle's own `kaggle.com/docs/notebooks` and `kaggle.com/docs/efficient-gpu-usage` pages render as a
JS SPA; `WebFetch` returned only the page title with empty body in every attempt this session (both
directly and via `r.jina.ai`, which did return partial body for the efficient-gpu-usage page). Numbers
below are therefore cross-checked across the community/product-feedback threads and a couple of
secondary write-ups; flagged where only secondary-sourced.

- **GPU options**: NVIDIA Tesla **P100** (16 GB HBM2, Pascal architecture, no tensor cores) or **T4 x2**
  (2× 16 GB GDDR6 = 32 GB total, Turing architecture, has tensor cores). Selectable per-notebook, shared
  from one weekly quota pool.
  Sources: https://www.kaggle.com/getting-started/561774 , https://www.kaggle.com/product-feedback/361104
- **Weekly GPU quota**: **~30 hours/week**, shared across whichever GPU type is selected; Kaggle has
  piloted a "floating" quota that can exceed 30 h depending on demand.
  Sources: https://www.kaggle.com/product-feedback/173129 , https://r.jina.ai/https://www.kaggle.com/docs/efficient-gpu-usage
  (confirmed "30 hours or sometimes higher depending on demand")
- **CPU cores / RAM on GPU notebooks**: as of a **[Notebooks update] GPU Notebooks 4 CPUs & 29 GBs of
  RAM!** change, GPU notebooks now get **4 CPU cores + 29 GB RAM**, up from the older **2 CPU cores +
  13 GB RAM**. Thread title and multiple corroborating secondary posts (LinkedIn, Medium) agree on these
  exact numbers; the underlying Kaggle discussion thread itself did not render its body via WebFetch, so
  the specific rollout date is UNVERIFIED (search results place the change around Nov 2025).
  Source: https://www.kaggle.com/discussions/product-feedback/448251 (title + secondary corroboration)
- **CPU-only baseline (no accelerator)**: reported elsewhere as 2× Intel Xeon vCPUs + ~32 GB RAM —
  **UNVERIFIED**, sourced from a third-party aggregated knowledge-base document, not Kaggle's own docs
  (https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md,
  dated 2025-11-21). Treat as directionally right but re-verify from `kaggle.com/docs/notebooks` directly
  (e.g. via a real Kaggle notebook printing `os.cpu_count()` / `psutil.virtual_memory()`) before relying
  on it for dataloader-worker-count decisions.
- **Session limits**: interactive session idle timeout is **60 minutes** (confirmed via the
  efficient-gpu-usage doc body). Our competition's own **9-hour (32400 s) hard runtime cap** for a
  submission notebook is set by the competition rules (already in CLAUDE.md), not by this general Kaggle
  session limit — the two are separate constraints.
- **L4 GPUs**: mentioned by a couple of third-party blogs/tutorials as available, but **not confirmed**
  in any Kaggle-hosted source fetched this session — treat as **UNVERIFIED / probably not offered** on
  Kaggle competition notebooks; verify directly in the notebook's Accelerator dropdown before planning
  around it.
- **fp16 vs bf16**: T4 (Turing) tensor cores support **fp16, int8, int4** but **not bf16**; bf16 tensor-core
  support arrived with **Ampere** (A100 etc.), which Kaggle does not offer for competition notebooks.
  P100 (Pascal) has **no tensor cores at all**, so fp16 there saves memory/bandwidth but gets much less
  matmul-throughput benefit than on T4.
  Source: NVIDIA T4 datasheet — https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf

### 6. DICOM decode speed and offline installation

- **pylibjpeg** (with its `openjpeg` plugin for JPEG 2000 and `libjpeg` plugin for JPEG Lossless) is
  pydicom's own recommended, "more accessible" path versus GDCM for pixel-data decompression.
  Source: https://pydicom.github.io/pydicom/stable/guides/plugin_table.html
- **pylibjpeg-rle** benchmark (RLE codec, not JPEG, but the only concrete number found): **~4–5x** faster
  decode and **~20x** faster encode than pydicom's native RLE handler (example: 4.89 s pydicom vs 0.79 s
  pylibjpeg-rle on one 480,000-pixel file).
  Source: https://github.com/pydicom/pylibjpeg-rle , https://pypi.org/project/pylibjpeg-rle
- A separate `pydicom-mojo` project claims **3.7x–16.3x** RLE decode speedups over pydicom's own
  implementation depending on content — again RLE-specific, not JPEG2000/JPEG-Lossless.
  Source: https://github.com/pydicom/pydicom/issues/2370
- **dicomsdl**: named in the research brief as a known-fast alternative decoder, but **no primary
  benchmark for dicomsdl vs pydicom/pylibjpeg was found** in this session's searches — **UNVERIFIED**,
  flag for direct benchmarking against a sample of our own JPEG-Lossless/JPEG2000 knee DICOMs before
  committing to it.
- **NVIDIA nvImageCodec**: ships a pydicom plugin that GPU-accelerates decode for JPEG, JPEG 2000, JPEG
  Lossless, and HTJ2K transfer syntaxes via a drop-in decoder-plugin registration with pydicom. The only
  concrete number found was for **HTJ2K encoding** (1.34 ms/slice, 2.39x compression) — not decode, and
  not the classic-JPEG2000 syntax our dataset actually uses — so nvImageCodec's real decode speedup for
  our files is **UNVERIFIED**; worth a direct benchmark. Whether it installs offline inside Kaggle's
  locked CUDA/driver image (internet disabled) is also **UNVERIFIED**.
  Source: https://docs.nvidia.com/cuda/nvimagecodec/samples/DICOM-pydicom.html
- **Offline installation pattern** (needed since our competition disables internet at inference time): the
  standard, well-documented Kaggle approach is to download the wheel(s) with internet on in a throwaway
  notebook, attach them as a private Kaggle **Dataset**, then
  `pip install --no-index --find-links=/kaggle/input/<dataset-name>/ <package>` inside the submission
  notebook. Multiple community notebooks confirm this exact pattern for GDCM and generic pip packages.
  Sources: https://www.kaggle.com/richardepstein/load-gdcm-in-notebook-without-internet ,
  https://www.kaggle.com/code/panozzaj/installing-pip-packages-on-kaggle-without-internet

### 7. Model inference speedups

- **fp16 autocast** is the realistic mixed-precision target on both Kaggle GPU options (not bf16 — see
  §5). Expect a bigger win on T4 (tensor cores) than P100 (no tensor cores).
- **torch.compile**: generally reported speedup **~20–40%** for well-shaped models, driven mainly by
  reduced Python overhead and fewer GPU read/writes; benefit shrinks for small models/small batches where
  GPU utilization is already low, and the **first call pays a compilation warm-up cost** — for a
  single-shot scored inference run, this means warming up on dummy input *before* the timed portion of
  the notebook, and keeping input shapes static (dynamic shapes force recompilation). Whether Kaggle's
  current default image/driver supports `torch.compile`'s Triton codegen without extra setup was **not
  confirmed** in this session — test early. Sources: https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html ,
  https://huggingface.co/docs/transformers/en/perf_torch_compile
- **TensorRT**: proven in production by an actual RSNA 1st-place team — the mammography winner combined
  a 4-fold ConvNeXt-small ensemble into **one TensorRT engine**, with conversion taking only ~5–10 minutes
  on a Kaggle P100 at submission time (see §4). This is the strongest, most directly transferable
  efficiency precedent found in this research: compiling the *whole ensemble* into a single engine rather
  than running N separate PyTorch forward passes.
- **ONNX Runtime** was not found used by name in any winner write-up surfaced this session (TensorRT was
  the concrete example) — no claim either way beyond that.
- **Batching / overlapping CPU decode with GPU compute**: not explicitly detailed in any winner write-up
  fetched here beyond the general 2.5D fixed-batch-shape pattern (§A recurring patterns #2); the general
  best practice (DataLoader workers decoding/prefetching while GPU computes the previous batch) is
  standard PyTorch practice, not something a specific RSNA source confirmed — treat as [Dom]-level
  engineering, not a verified competition-specific finding.

### 8. Efficiency-prize precedent elsewhere on Kaggle

- **"LLM - Detect AI Generated Text" — 1st Place Efficiency Prize** (community discussion thread; read
  via r.jina.ai proxy — https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/471898): the
  winning efficiency submission used a **pure classical-ML pipeline** — a custom byte-pair-encoding
  tokenizer trained on the public+private test text, TF-IDF vectorization, and a `VotingClassifier` of
  `MultinomialNB` + `SGDClassifier` + `LGBMClassifier` — **no GPU, no LLM at all**. Reported **CPU
  inference time: 17 minutes**, private LB score 0.91967 (respectable, though below the very top
  main-leaderboard scores in the mid-0.9x range). The author's own framing was that matching the
  train/test *data distribution* mattered more than model sophistication for this track. This is text,
  not imaging, but the transferable lesson is direct: **efficiency-track winners intentionally pick a
  categorically simpler/faster model, not a lightly-trimmed version of the accuracy-leaderboard model.**
- I specifically searched for **Leash Bio (BELKA)** and **Kaggle LLM Science Exam** efficiency-prize
  write-ups, per the research brief's suggestion, and **did not find any public write-up describing an
  efficiency-specific approach for either** — UNVERIFIED whether those two competitions even had a
  dedicated efficiency track matching our formula's structure; do not assume they do.
- No other RSNA competition (2022–2024) surveyed here documented a dedicated efficiency prize; the
  general engineering choices that *would* matter for one (TensorRT compilation, fixed small
  slice/vertebra budgets, 2.5D instead of full 3D) already appear in the main-leaderboard winners covered
  in Part A, because the competition format (code competition, ≤ a few hours) forced them regardless of
  whether a separate efficiency prize existed.
- Our own competition's efficiency formula, restated from CLAUDE.md for reference:
  `Efficiency = AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400` (lower is better; RuntimeSeconds term is
  linear in wall-clock inference time, worth 1.0 at the full 9 h cap). The "Detect AI Generated Text"
  precedent above is consistent with this: a large, guaranteed runtime win (17 min vs. hours) can be worth
  more to an efficiency score than the AUC/accuracy the heaviest model would add.

---

## Lessons for our knee ViT pipeline

1. **Real risk to the from-scratch-ViT bet.** No RSNA MRI/CT 1st/2nd place solution surveyed here trained
   a from-scratch transformer for its main submission, and the one team that directly tested ViTs (lumbar
   spine 1st place) found CNNs and CNN+bi-LSTM/MIL beat them, and that Mamba/self-attention pooling
   underperformed bi-LSTM pooling. On our much smaller labeled set (58 verified studies, pseudo-labeling
   plan still unproven), this reinforces Dom's Oct 8 fallback checkpoint — and suggests a **hybrid**
   (pretrained CNN/CNN-hybrid backbone per slice + transformer or bi-LSTM+attention-MIL aggregator across
   slices/series) as a concrete middle-ground fallback that's much closer to what's actually won, rather
   than jumping straight to a full generic pretrained backbone.
2. **Adopt the two-stage localize-then-classify pattern, adapted for our lack of bounding-box labels.**
   Every winning MRI/CT solution bounded its per-study compute with a Stage 1 that decides *which slices
   matter* before Stage 2 classifies. We don't have coordinate/bbox labels, but we do have
   `Anatomical_Plane`/`Fluid_Sensitive`/`Fat_Suppression` metadata and can build a lightweight
   slice-relevance or informative-slice-selection step (even unsupervised, e.g. variance/entropy-based, or
   a small self-supervised model) rather than feeding entire variable-length series into the model.
3. **Use 2.5D framing with an explicit, fixed per-study slice budget**, matching the pattern in all four
   competitions (96 equidistant slices; 15/vertebra; 5-slice or 24/27-frame crops). Given our series are
   typically 20–45 slices (median 30) with a long tail to a few hundred, fix a slice budget (e.g. ~24–32
   per series, matching the observed median) both for modeling consistency and for bounding
   `RuntimeSeconds`.
4. **Handle multiple series as separate, metadata-conditioned input streams**, not naive channel-stacking
   across incompatible sequence types — directly matches the CLAUDE.md "multi-view input" ablation; the
   top lumbar-spine solutions picked *which* series feed *which* finding rather than fusing everything
   into one tensor.
5. **Target fp16, not bf16**, for mixed precision on Kaggle's T4/P100 — bf16 tensor-core support doesn't
   exist on either GPU Kaggle offers. Prefer T4 x2 over P100 at submission time if given the choice: it
   has tensor cores (fp16 speedup is real) versus P100's none, and doubles available VRAM (32 GB vs
   16 GB) for ensembling or larger batches.
6. **Budget wheel-bundling time for offline DICOM decoders.** Since the competition notebook runs with
   internet off, plan now to build a small Kaggle Dataset containing pylibjpeg (+ openjpeg/libjpeg
   plugins) and/or GDCM and/or dicomsdl wheels, installed via
   `pip install --no-index --find-links=...` — this is the standard, repeatedly-confirmed pattern for
   every RSNA DICOM competition. Actually benchmark dicomsdl vs pylibjpeg vs GDCM vs nvImageCodec on a
   sample of our own knee JPEG-Lossless/JPEG2000 files early — none of these had a citable decode-speed
   number for our specific transfer syntaxes in this research pass.
7. **Consider compiling the final ensemble into a single TensorRT (or at minimum ONNX) engine before the
   scored run**, following the mammography 1st-place precedent — this cut their multi-fold ensemble down
   to one fast compiled engine at a fixed, small conversion cost (~5–10 min on a P100). If our competition
   allows attaching a pre-built engine/artifact as a Kaggle Dataset (built offline, not during the scored
   run), that avoids spending `RuntimeSeconds` on compilation entirely — verify this is allowed under the
   competition's submission mechanics before relying on it.
8. **Plan two distinct submissions if pursuing the Efficiency Prize seriously**: the CLAUDE.md rules
   already note up to 2 final submissions are allowed. The "Detect AI Generated Text" efficiency-prize
   precedent shows the winning move there was a categorically simpler/faster pipeline, not a lightly
   pruned copy of the top model — so plan an explicit lean "efficiency" variant (smaller backbone, fewer
   slices, no TTA, single compiled engine, batched inference) as a second, distinct submission from the
   max-AUC one.
9. **Kaggle GPU notebooks now have more CPU/RAM headroom than older references assume** (4 CPU cores,
   29 GB RAM on GPU notebooks, up from 2/13) — useful for DICOM-decode worker parallelism, but still far
   below Dom's 16-core/16 GB dev box, so preprocessing code must be profiled on both ends and shouldn't
   assume dev-machine-level parallelism carries over to the scored notebook.

---

## Sources

**RSNA 2024 Lumbar Spine Degenerative Classification**
- [1st place solution (Avengers)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/avengers-1st-place-solution)
- [2nd place solution (ianpan/kevin/yuji/bartley)](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/writeups/ianpan-kevin-yuji-bartley-2nd-place-solution)
- [brendanartley/RSNA-2024-Competition (2nd place code)](https://github.com/brendanartley/RSNA-2024-Competition)
- [DaoyuanLi2816 Silver Medal repo](https://github.com/DaoyuanLi2816/Kaggle-RSNA-2024-Lumbar-Spine-Degenerative-Classification-Silver-Medal) (existence only, not deep-dived)
- [nshen7/rsna-2024](https://github.com/nshen7/rsna-2024) (existence only, not deep-dived)

**RSNA 2023 Abdominal Trauma Detection**
- [1st place writeup (Team Oxygen)](https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/writeups/team-oxygen-1st-place-solution-team-oxygen)
- [Nischaydnk/RSNA-2023-1st-place-solution (code + README, primary source used)](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)

**RSNA 2022 Cervical Spine Fracture Detection**
- [1st place solution (Qishen Ha)](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution)

**RSNA Screening Mammography Breast Cancer Detection (2023)**
- [1st place solution (mr.robot)](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/writeups/mr-robot-1st-place-solution)
- [dangnh0611/kaggle_rsna_breast_cancer (code)](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)

**Kaggle notebook hardware / quota**
- [Differences Between NVIDIA GPU T4 and P100 | Kaggle](https://www.kaggle.com/getting-started/561774)
- [\[Notebooks update\] New GPU (T4s) options & more CPU RAM | Kaggle](https://www.kaggle.com/product-feedback/361104)
- [\[Notebooks update\] GPU Notebooks 4 CPUs & 29 GBs of RAM! | Kaggle](https://www.kaggle.com/discussions/product-feedback/448251)
- [\[Notebooks update\] More GPU hours - "floating" GPU quota | Kaggle](https://www.kaggle.com/product-feedback/173129)
- [Efficient GPU Usage Tips | Kaggle docs](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle notebook specs secondary aggregation, dated 2025-11-21 — UNVERIFIED primary](https://huggingface.co/datasets/John6666/knowledge_base_md_for_rag_1/blob/main/kaggle_20251121.md)
- [NVIDIA T4 Tensor Core datasheet (Turing, no bf16)](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)

**DICOM decoding**
- [pydicom pixel-data plugin table](https://pydicom.github.io/pydicom/stable/guides/plugin_table.html)
- [pylibjpeg-rle (GitHub)](https://github.com/pydicom/pylibjpeg-rle)
- [pylibjpeg-rle (PyPI)](https://pypi.org/project/pylibjpeg-rle)
- [pydicom-mojo RLE benchmark discussion (GitHub issue)](https://github.com/pydicom/pydicom/issues/2370)
- [Accelerated DICOM Decoding with nvImageCodec and pydicom (NVIDIA docs)](https://docs.nvidia.com/cuda/nvimagecodec/samples/DICOM-pydicom.html)
- [Load GDCM in Notebook without Internet | Kaggle](https://www.kaggle.com/richardepstein/load-gdcm-in-notebook-without-internet)
- [Installing pip packages on Kaggle without internet | Kaggle](https://www.kaggle.com/code/panozzaj/installing-pip-packages-on-kaggle-without-internet)

**Inference speedups**
- [torch.compile end-to-end tutorial (PyTorch docs)](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html)
- [torch.compile for inference (Hugging Face docs)](https://huggingface.co/docs/transformers/en/perf_torch_compile)

**Efficiency-prize precedent**
- [\[1st Place Efficiency Prize\] writeup, LLM - Detect AI Generated Text | Kaggle](https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/471898)
