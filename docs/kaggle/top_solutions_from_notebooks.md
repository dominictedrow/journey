# What the strong public RSNA-Knee notebooks actually do

Compiled 2026-09-24 from notebook source text in the session scratchpad (`scratchpad/nb/*.txt`, converted `.ipynb`; **none of the notebooks had saved outputs**, so every number below comes from markdown, code comments, pinned constants, or the attached datasets' descriptions/manifests/training-history JSONs fetched with `kaggle datasets metadata/files/download -f <small file>`). No large weight files were downloaded.

Conventions: **[ref]** = where a claim comes from (key table below). `UNVERIFIED` = stated by a third party and not checkable from here, or my inference. "gold58" = the 58 studies in `train.csv` with official labels. "LB" = public leaderboard unless noted.

## Source keys

| Key | Source |
|---|---|
| dd-inf | `dreaddevelopment/knee-mri-twelve-findings-from-a-single-model` (inference, 0.924) |
| dd-train | `dreaddevelopment/knee-mri-training-the-twelve-finding-model` (training script) |
| dd-ds:* | dreaddevelopment dataset descriptions: `raptor-knee-arms`, `-arms-x`, `-widedense`, `-native384`, `-native384dense`, `-maxspan`, `-finespacing`, `knee-raptor-corpus(-ext)` |
| ma-v3 | `mattiaangeli/bend-the-knee-to-dinov3-ensembled` |
| ma-dino | `mattiaangeli/bend-the-knee-to-the-dinosaurs` |
| ma-speedy | `mattiaangeli/bend-the-knee-to-speedy-raptors-the-original` (0.943, "v34") |
| ma-g96 | dataset `mattiaangeli/rsna-knee-coatnet-global96-top3` (manifest, `training_history_global96_e24.json`, `train_coatnet_global96_exact.py`) |
| ma-rg | dataset `mattiaangeli/rsna-knee-coat-resgated-ep10-top3` (manifest, `training_history_through_epoch10.json`) |
| ma-d4 | dataset `mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2` (manifest, ablation plan md) |
| rt-v5 / rt-train | `romantamrazov/rsna-knee-dinosaur-v5` / `-v5-train` |
| sa-e11 / sa-e13 / sa-v46 | `sofiaanjenje/rsna-knee-e11-train`, `-e13-train`, `-frontier-v46` |
| ag-e9 / ag-e11 | datasets `antoinegg1/rsna-knee-e9-radimagenet-heads-v15` (README + `public_oof_diagnostic.json`), `-e11-diverse-heads-v20` |
| pk | `pilkwang/rsna-knee-baseline-v1` (origin of the 20 DINOv2 members, `pilkwang/rsna-knee-weights`) |
| pv-report | `prvsiyan/rsna-knee-read-the-report-then-the-knee` (origin of RadImageNet "V52" heads) |
| pv-hs / pv-bee | `prvsiyan/head-and-shoulders-knees-and-toes`, `prvsiyan/the-bee-s-knees-final-rsna-push` |
| fm | `fuadmuhammad/rsna-knee-fast-parent-0-957-v2` |
| jl | `jiweiliu/rsna-knee-fast-2xt4-inference` |
| hd | `haideptry/rsna-knee-speedy-raptors-coatnet-d4-0943` |
| nj | `nathanjacob/4-arm-ensemble-explained-rsna-knee-0-937` |
| wg | `wguesdon/rsna-knee-dinov2-at-meniscus-resolution` |
| st | `sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee` |
| rn | `renta0426/rsna-knee-hybrid-raptor-meniscus-specialist` (+ `renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid`) |
| tl | `tonylica/rsna-knee-dino-radimagenet-rank-ensemble` + dataset `tonylica/rsna-knee-bend-dinov3-0917-repro-assets` |
| sk | `starkhushi/rsna-knee-why-public-forks-stop-at-0-941` (not in the assigned list; read for the fork/overfit evidence) |
| sl-ds | dataset `stevenleehans/rsna-knee-llm-report-labels` description |
| pk-ds | dataset `pilkwang/rsna-knee-llm-labels` description |
| eff-lb | output of `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` (`full_leaderboard.csv`, 4,218 teams, pulled 2026-09-24) |
| forum | sibling doc `docs/kaggle/forum_and_official_pages.md` (forum threads 735304, 736678, 737696) |

## 0. Headline facts (read first)

1. **Scale of the data (third-party, strongly corroborated):** 4,407 training studies; 58 gold; 4,349 report-only [dd-train, dd-ds:corpus, st, pv-report]. The corpus row counts (3,200 + 1,207 = 4,407) confirm it [dd-ds:corpus]. 24,371 series (9,864 sagittal / 8,609 coronal / 5,898 axial) [wg, st]. Study averages ~5.5 series x ~34 slices [st]. Patients are 1:1 with studies [wg]. Test is "1,322 hidden studies" per [rn] (`UNVERIFIED`; Kaggle says ~1,300). Still confirm from `train.csv` ourselves.
2. **Every strong public model is ImageNet- or self-supervised-natural-image-pretrained, fine-tuned on ~4,349 studies with LLM-read soft labels from the reports.** No public notebook publishes a random-init or knee-SSL result (see section 5).
3. **The best public single model family is a CoAtNet-RMLP-2 (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, ~73M params) at 384 px on 2.5D three-slice windows with a per-finding attention-MIL head.** Single-model LB 0.914 → 0.932 purely from slice sampling changes [dd-ds:*]. Best gold58 single: 0.9274 (Global96 e16) [ma-g96].
4. **The public ceiling is ~0.943 (Speedy Raptors v34) for a 40+ checkpoint ensemble; ~200 teams sit at 0.941 running forks** [sk]. Top of the public LB is 0.958 [eff-lb]; the #1 *efficiency-ranked* team also shows 0.958 [eff-lb] — so a fast pipeline can be at the top (its runtime is not shown; `UNVERIFIED` how fast). Forum reports single models at 0.950 (Qwen3.5-2B VLM LoRA; single-fold CoAtNet at 224 px) [forum 735304] — not visible in any public notebook here.
5. **Labels are the biggest lever the public notebooks report.** Expanding the corpus from 3,155 to 4,349 studies was worth +0.013 on gold [dd-inf, nj]. Changing slice sampling was worth 0.006–0.011 LB, while a 3-backbone ensemble added about 0.001 [dd-ds:arms].

---

## 1. Every model "arm"

### 1.1 Raptor CoAtNet (dreaddevelopment / Johnathan Wagner). The key single-model recipe

**What "Raptor" is.** "Raptor" is the author's project name. The trainer contains a `load_raptor()` that loads a **self-supervised checkpoint** (it reads `ck["student"]`, strips a `backbone.` prefix, and logs `ssl epoch`). It also has a `--ckpt none` **"RANDOM-INIT control (no SSL weights)"** path [dd-train]. The defaults point at a small ViT: `--arch vit_small_patch16_224 --res 224`, input normalization `"none"` in [0,1], with the comment "matches the SSL input pipeline: ToTensor, no ImageNet norm". A RAM comment gives "~87 MB a copy for the small ViT" [dd-train]. **However, every released arm was trained from ImageNet timm weights, not from the SSL checkpoint.** The code says: "Defaults to ImageNet-pretrained weights, which is what every released arm was trained from. The previous default pointed at a self-supervised checkpoint that is not distributed here" [dd-train].
- **How the Raptor SSL was trained is `UNVERIFIED`.** The method, data, epochs and size are not published in any notebook or dataset description I read. The only evidence is the `student` key, which suggests a DINO/iBOT-style student/teacher (`UNVERIFIED`), a ViT-S/16 at 224 (inferred from the defaults and the 87 MB figure; `UNVERIFIED`), and inputs as [0,1] 3-slice windows. **No SSL-vs-ImageNet or random-init number is published.** The forum thread 737696 might contain it; it was not in my sources.

**Architecture** [dd-inf, dd-train]:
- Backbone: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` (timm), global avg pool, 768-d features, 384 px input.
- 2.5D window: 3 physically adjacent slices → 3 channels (ImageNet mean/std normalization for the CoAtNet arms).
- Head `RaptorClassifier` (per-finding attention MIL): LayerNorm → `att = Linear(F,256) → Tanh → Dropout(0.2) → Linear(256,12)`, softmax over the K windows **separately for each finding**, pooled feature per finding, then a per-finding linear classifier (`clsW[12,F]`, `clsb[12]`). The authors call this "the part that matters most".

**Input construction (corpus geometry)** [dd-inf, dd-ds:*]:
- Five fixed slots in this order: Sagittal fluid-sensitive, Sagittal non-fluid, Coronal fluid, Coronal (other), Axial. A missing slot is zero-filled and masked.
- Slice counts per slot and the resulting LB:

| Corpus | Slots (sag-F, sag-N, cor-F, cor-2, ax) | Span of stack | Eval windows | Stored px | LB (single) | gold58 |
|---|---|---|---|---|---|---|
| v1 corpus (`knee-raptor-corpus`) | 12,10,8,6,8 = 44 | 15–85% | 42 | 336 | 0.914 (3,200-study training) → 0.917 (4,349 studies, `arms-x`) | — |
| widedense (v4) | 18,14,12,8,12 = 64 | 6–94% | 42 | 336 | 0.924 | 0.9167 (README) / 0.9054 (notebook comment, "58-study gate") — the two sources disagree |
| native384 (v8) | 44 slices | 6–94% | 42 | 384 | 0.926 | — |
| 44 slices over 2–98% | 44 | 2–98% | — | 336 | 0.917 (worse: too sparse) | — |
| maxspan (v5) | 64 | 2–98% | 62 | 336 | **0.928** | — |
| native384dense (v10) | 64 | 2–98% | 62 | 384 | not scored at time of writing | 0.9174 |
| finespacing (v9) | 22,18,15,10,15 = 80 | 2–98% | 78 | 336 | **0.932** | — |

- The author's lesson: "widening the slice span only pays if slices are added to keep the sampling dense". The same weights read differently with the window count: 64 slices at 42 windows scored 0.924, and the identical weights at 62 windows scored 0.927 [dd-ds:finespacing]. The forum reports that more test-time windows help monotonically but sub-linearly [forum 737696].
- Preprocessing: slices sorted by `dot(ImagePositionPatient, cross(IOP_row, IOP_col))` (fallback `InstanceNumber`). Then `apply_modality_lut` (slope/intercept), `MONOCHROME1` inverted, and a **per-series 2–98th percentile window** computed over the pixels of the sampled slices. Next a **140 mm centre crop using `PixelSpacing`**, `cv2.INTER_AREA` resize to 336, and storage as uint8. Picks are `np.linspace` over the ordered stack between the span fractions [dd-inf].
- Series choice per slot: first unused series with matching `Anatomical_Plane` and preferred `Fluid_Sensitive` value, else any unused series in that plane [dd-inf].

**Training recipe** [dd-train]:
- Labels: LLM-read soft probabilities for 4,349 report-only studies ("a report that says a tear is suspected becomes ~0.8"). **The 58 gold studies are never trained on and are used only for validation.** The labels parquet is *not* published.
- K = 12 random windows per study in training, 24–42 evenly spaced in eval ("each epoch sees a different view of the same study"). Only augmentation: ±10% intensity gain. **"NO flips (laterality is signal)."**
- AdamW with split LRs: backbone 3e-5, head 1e-3, wd 0.02. OneCycle with pct_start 0.15, grad-clip 3.0, bf16 autocast. `BCEWithLogits(pos_weight = clip((1-prev)/prev, 1, 10))`, with prevalence clipped to [0.03, 0.7].
- Typical run: 16 epochs, bs 8, grad checkpointing. "About three hours on one 4090", and ~23.7 GB VRAM for CoAtNet-384 at bs 8 with grad-ckpt. "On 16 GB use --bs 2 or 4." The best epoch is picked by gold58 macro-AUC, and a top-3-epoch weight average (SWA) is also written. On the widedense run SWA did *not* help [dd-ds:widedense].
- The corpus is a memmap `(4407, 44, 336, 336)` uint8 of 15.9 + 6.0 GB [dd-ds:corpus].

**Backbone bake-off on 45 gold studies** [dd-inf comments]: CoAtNet-384 0.9025, Swin-B-384 0.8825, CNN336 0.8833, EffNetV2-L-480 0.8716, ConvNeXt-B-384 0.8754, ConvNeXt-L-384 0.8752, **MaxViT-384 0.8438**. The blend {coatnet+swin+effv2l} reached 0.9068 on gold, **but on the LB CoAtNet alone scored 0.914 and every blend 0.914–0.915**: "the ~+0.010 it showed on the old 45-study gold set was gold-set noise."

### 1.2 Mattia Angeli's CoAtNet readers (residual-gated, D4 depth-zone, Global96, Repair-v1)

All four use the same backbone as Raptor (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, timm 1.0.22) at 384 px, with a **130 mm physical triplet FOV** ("smart384" cache), 4,349 report-labelled training studies, and **top-3 epochs selected on gold58** [ma-rg, ma-g96, ma-d4].

**Residual-gated CoAt (ep 4/6/8)** [ma-rg]:
- Head: "bounded gated spatial residual plus finding-specific slot/depth window attention", 73.42M params. It uses a 12x12 spatial grid attention mixed with global-average pooling (a "protected GAP residual") [ma-dino, ma-rg].
- Training: K32 x 2 independent sub-bags per study per epoch, effective batch 24 (micro 6 x accum 4). Backbone LR 3e-5, head 1e-3, wd 0.02, warmup 0.15. Loss = "global dualsharp weighted BCE + macro pairwise rank" with rank_w 0.05 and a study-shared intensity jitter. 10 of 24 planned epochs ran, taking 7,836 s (A6000, then H100).
- Gold58 by epoch: 0.748, 0.842, 0.882, **0.9005**, 0.894, 0.898, 0.890, 0.897, 0.886, 0.884. The top-3 rank-ensemble scored 0.9095 and the probability mean 0.9116.
- In the final stack it is 25% of the CoAt family [ma-speedy].

**Global96 top-3 (e16/e23/e18)** [ma-g96]:
- A fixed 96-slice stack with slot widths **27/21/18/12/18** over a 2–98% depth band. Missing blocks are zero-filled. Training uses K12 random global triplets; evaluation uses **all 94 adjacent triplets**, including cross-slot boundary triplets.
- Head: a gated MIL window router (tanh x sigmoid), full width 1024. The spatial reader is 85% patch-attention / 15% GAP, initialised exactly as GAP ("safe start"). Router conditioning uses `centered_depth`, `stride_norm` and the A/B/C slot identities of the 3 channels.
- Recipe: 24 epochs, effective batch 24 (micro 8 x accum 3), 182 steps/epoch. Backbone LR 3e-5 → 3e-7, head 1e-3, wd 0.02, warmup 0.15, grad checkpointing, channels_last. Loss: confidence-weighted soft BCE + macro pairwise rank loss (pos > 0.6 vs neg < 0.4, softplus, rank_w 0.075). Labels come from `targets_dualsharp_soft.csv` plus a confidence-weight CSV (construction `UNVERIFIED`). Intensity jitter 0.1.
- Hardware and time: **RTX 5090, 10,261 s for 24 epochs** (~7 min/epoch); the pixel cache is ~94 GiB [ma-g96].
- Gold58 by epoch: 0.705 (e1) → 0.911 (e8) → **0.9274 (e16)** → 0.924 (e24). Top-3 probability mean **0.9305**; SWA top-3 0.9291.
- Per-finding gold58 (top-3 prob mean): ACL 0.973, MCL 0.977, Med Meniscus 0.975, **Lat Meniscus 0.880**, Med OA 0.983, **Lat OA 0.865**, **PF OA 0.885**, Effusion 0.973, **Synovitis 0.810**, Baker's 0.976, Contusion 0.912, Fracture 0.957. These are selection-optimistic because gold58 picked the epochs.

**D4 depth-zone SWA3** [ma-d4, ma-speedy]:
- Parent: a "rank8" CoAtNet FiLM model (SWA of epochs 15 and 16, trained on the Global96 k12/full94 contract). On top sits a 3,255-parameter adapter that splits each anatomical slot into **three depth zones** (SWA of adapter epochs 11/9/5). Serving caps up to 94 windows.
- The ablation plan (A1 finding x slot radial prior with 60 params; A2 depth spline with 360; A3 content x depth router; A4 slot x depth cross-attention; A5 BiMamba) explicitly says "**Gold58 must not set coefficient signs, spline shapes, caps, or the winner**". It also says none of the arms had "demonstrated an MRI AUC or leaderboard gain" at plan time (2026-09-09) [ma-d4].

**Repair-v1 top-3**: consumed as `coatnet_repairv1_top3_*` in [ma-speedy, rt-v5]. Its architecture and training are `UNVERIFIED` (not described in any source I read).

### 1.3 DINOv2-small "20 members" (Pilkwang Kim) [pk, nj, jl]

- Backbone: `facebook/dinov2-small` (ViT-S/14, 12 blocks). **Only the last 6 blocks plus the final LayerNorm are trainable** (`UNFREEZE_LAST = 6`), with backbone LR 8e-6 vs head LR 1e-3 and wd 0.02. Batch 8 studies, 10 epochs. Augmentation: rotation ±8°, scale 0.08, shift 0.05, intensity 0.10, **no flips** [pk].
- Input: 6 slots (SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1). The weighting is *recovered from DICOM headers* (`SeriesDescription`, `SequenceName`, `ScanOptions`, `ScanningSequence`, `RepetitionTime`, `EchoTime`) because the CSV flags are identical. Each slot contributes 3 adjacent slices as RGB, with a 130 mm crop, cache at 336 px, runs at 224 or 336, and slice band 20–80% [pk]. nj describes the members as 336x336.
- Head "SlotHead": project each slot, add a slot identity, then use **one learned query per finding attending over slots with a masked softmax**. A fixed per-finding slot-prior table (strength 0.55) applies for imported members (e.g. ACL→{SAG_FS, SAG_NOFS, SAG_T1}) [pk].
- Labels: a multilingual rule extractor, or the mounted LLM table. Report-derived confidence becomes the sample weight, and gold studies stay in training at elevated weight. The split is one fixed 1/5 holdout **grouped by a hash of the report text** [pk].
- Performance: the "Pilkwang V15 public anchor" of 20 members scored **0.891 LB** [pv-report]. The "E2" DINO parent family scored gold58 0.840 in the E9 diagnostic [ag-e9]. Its exact composition is `UNVERIFIED`, and gold rows were in these members' training, so that number is contaminated.
- Inference trick: all 20 members share a **bit-identical frozen 6-block prefix**, so the prefix is run once and fanned into 20 tails [jl, fm].

### 1.4 DINOv3-small "A5" 5 folds (Mattia Angeli, `mattiaangeli/knee-mri-fold-weights`) [ma-v3, pv-hs, tl]

- Backbone `vit_small_patch16_dinov3.lvd1689m` (DINOv3 ViT-S/16), 336 px, 130 mm crop, slice band 12–88%, **16 slices per slot**, and 6 slots = plane x fluid [ma-v3 code constants].
- Optional stems: `DepthCompress` maps 16 slices to 3 channels with a gated 1x1 conv; `SlotDepthMixer` is a learned depth kernel per plane/contrast. The per-checkpoint `cfg` chooses between them (`UNVERIFIED` which one is used).
- Head: a series-type embedding (plane x fat-sat) added to the tokens. Tokens from all series are concatenated into one K/V sequence, 12 learned finding queries cross-attend, and the result is concatenated with the mean and max of the per-series CLS embeddings; absent series are key-padding-masked. Pool variants in code: `xcodex` / `xres` gated residual pools [ma-v3].
- Blend: 45% against the 20-member DINOv2 rank-mean ("55/45 transformer blend") [tl, jl]. It was "selected on held-out labelled studies" [ma-v3]. Precision pref 'bf16' "retained including T4 execution" [jl]. Individual LB/gold numbers are `UNVERIFIED` (not given).

### 1.5 RadImageNet ResNet-50 heads (prvsiyan V52 → Antoine E9/E10 → Sofia E11/E13) [pv-report, ag-e9, ag-e11, sa-e11]

- **The encoder is frozen**: the official RadImageNet ResNet-50 (licence **CC-BY-NC-SA-4.0**), 2048-d GAP features per slice.
- Inputs:
  - V52/E13 layout: 3 fat-suppressed slots (SAG/COR/AX), 8 slices each = 24 tokens, 224 px, full frame (`CROP_MM = 10000`, so no crop), band 12–88%.
  - E11 layout: SAG/COR/AX non-FS plus one SAG FS anchor, **130 mm crop**, 8 slices, 224 px [sa-e11, ag-e11].
- Head `FoundationQueryHead`: LN → Linear(2048→512) → GELU, plus plane and slice-position embeddings, 12 queries with 8-head MHA, and a fuse MLP. AdamW 2e-4, wd 3e-3, batch 48, up to 24 epochs, early stop on grouped weak-label AUC. 5-fold **GroupKFold by report hash** (it raises on "report leakage").
- Targets: the mean of 3 LLM teachers (Pilkwang `report_labels_v2`, Steven `llm_labels_v2`, lixin73 `labels_llm_gpt56sol`). Weight = 0.15 + 0.85·(0.65·agreement + 0.35·certainty), and gold rows override with weight 3.0 [sa-e11].
- Numbers:
  - RadImageNet gold58 OOF 0.842 [ag-e9]; E9b weak OOF 0.828, gold OOF 0.854 [sa-e11 pinned constants].
  - Per finding it beats the DINO parent on Lat Meniscus (0.722 vs 0.642), Lat OA (0.812 vs 0.708) and Contusion (0.901 vs 0.842). It is worse on Fracture (0.835 vs 0.894) and Baker's (0.899 vs 0.955), so Baker's and Fracture are excluded from its vote [ag-e9].
  - The V52 20% rank vote moved prvsiyan's public score to **0.906** [pv-report].
- The E11 heads trained on a T4 in 2,065 s [ag-e11].

### 1.6 Other arms and single models worth knowing

| Arm | Recipe | Result | Ref |
|---|---|---|---|
| EfficientNet-B3 MIL (Yash Bishnoi) | single-channel ImageNet B3, 5 folds, 3 fluid-sensitive plane-diverse series, 12 slices/series train, 32 infer, max-pool over slices then mean over series, 288 px, no flip, 8 epochs/fold, bf16, fused soft labels (Qwen3.6-35B + Pilkwang, per-finding reader weights). **Sorted by `InstanceNumber`**. | OOF 0.8544, held-out gold 0.8568, **LB 0.903**; 12.4 h to train 5 folds | pv-report §9 |
| DINOv2 ViT-B/14 fine-tune (sadamtorres) | 5 folds; slice attention pooling with normalised slice position and plane embedding; CLS + patch-mean; per-plane mean, then concat; Qwen3-14B-AWQ labels (gold agreement 0.881) | frozen: gold 0.771 / LB 0.776; FT 224 px: gold 0.824 / **LB 0.866** (~3 h infer); FT 336 px: **LB 0.883** (~4 h infer) | st |
| ConvNeXt-T MRI-adapted (romantamrazov) | `convnext_tiny.fb_in22k_ft_in1k`. Stage A: train only the last stage + a linear head on 4 triplets/study of pseudo-labels for 3 epochs, AdamW 8e-5, eff. batch 64, fp16. Stage B: freeze, extract 5 triplets per series from up to 8 series (150 mm crop, 224 px), then a label-aware head (12 queries, 2-layer label-transformer). Loss 0.88·soft-BCE + 0.12·pairwise rank. 5-fold on gold58 only (non-gold always in train). Blend weight 0.04–0.12 only for targets with OOF AUC ≥ 0.66 | results not stated in source | rt-train |
| 3-backbone Raptor ensemble | ConvNeXtV2-B-336 + CoAtNet-384 + EffV2-L-480 | LB 0.915 vs 0.914 CoAtNet alone | dd-ds:arms |
| Own 2.5D members injected at 15% (ResNet34 / ConvNeXt / RadImageNet, holdout 0.85–0.87) | — | 0.936 (worse than 0.941) | sk |

---

## 2. Ensembling

### 2.1 The 0.941–0.943 public stack (lineage in the final section) [tl, jl, fm, ma-speedy]

```
T (transformer/Rad branch):
  20 DINOv2 members -> per-member percentile rank -> mean                     (base)
  5 DINOv3 A5 folds -> rank mean; blended 45% vs base                          (a5_w=0.45)
  RadImageNet: reference-v15 heads + E13 heads mixed 50/50 on ranks;
     gets rad_alpha 0.50 (0.55 in v7+) on 10 findings (NOT Baker's, Fracture);
     E13 heads re-run on E11 layout (+ h-flip view) blended at 0.15 (0.20 in v7+);
     frozen 88-input linear calibrator adds 40% rank vote on 7 findings.
C (CoAt branch):
  Raptor views: maxspan-v5 0.60 | maxspan-v5-REVERSE 0.10 | native384dense-v10 0.10 | native384-v8 0.20
     (probability mean -> ordinal rank)          [0.941 recipe used 0.55/0.15]
  Private CoAt family (v34): resgated/D4/Global96/Repair-v1 25% each, probability mean -> rank
  H = rank(0.60 * Raptor + 0.40 * CoAt family)
Final[target] = rank((1-w) * T + w * H), w = 0.60 default;
   "probe22" map: ACL 0.75, Medial Meniscus 0.80, Lateral Meniscus 1.00, Lateral OA 0.75, Fracture 0.75
```

- "Reverse" view: the MaxSpan checkpoint is run with the **3-channel (slice) order reversed**, a free second view [tl: "reverse-channel view"; ma-speedy `"reverse": True`]. nj calls it a "horizontal flip", which is `UNVERIFIED` and contradicted by tl.
- **All combining is done on per-target percentile ranks**, because AUC reads only order [pk, st, tl]. Exceptions:
  - Within one training run (the Global96 epochs), a probability mean then rank beat a rank mean by +0.0015 on gold [ma-g96].
  - The CoAt family switched from rank-of-mean-ranks to rank-of-mean-probabilities at v34 [jl changelog].
  - The label blends in [sl-ds] are done "in probability space, not rank space".
- Score ladder [sk, fm, pv-bee, ma-speedy, jl]:
  - v27 missing sources: 0.937 (the 20 DINOv2 members silently dropped)
  - v27 complete: 0.940
  - flat 0.60 routing with resgated CoAt: 0.939
  - probe22 per-finding map: 0.941
  - + D4 (v7): 0.942
  - + Global96 + Repair-v1 (v34): **0.943**
  - Mattia's private team: 0.946 [sk].

### 2.2 Specialists and per-finding routing
- **Lateral Meniscus is routed 100% to the CoAt branch** (3 of 4 stages dropped for that column). Medial Meniscus goes 80% and ACL / Lat OA / Fracture 75% [fm, tl]. These weights came from LB **"probes"** (#15, #22) [rn, fm].
- RadImageNet is kept off Baker's and Fracture [ag-e9, sa-e11].
- Earlier stages that replaced Synovitis, and Lateral Meniscus + Lateral OA, with report/metadata models were removed: "leaving those findings to the image models scored better" [ma-v3]. Two per-finding specialists were also removed after tonylica's run without them scored higher [ma-v3].
- romantamrazov's V16 SlotHead specialist: added to a finding only with weight ≤ 0.20 × 0.85 shrink, and ×0.6 if its correlation with the anchor is > 0.985 [rt-v5]. The ConvNeXt arm uses fold-rank voting with a Gold58 reliability adjustment and a bootstrap on gold for target weights, plus 3-view **depth TTA ±0.035** [rt-v5]. No LB number is reported for these.

### 2.3 TTA
- The only TTA in the winning stack is multi-view reuse: Raptor reverse-channel, the E13 h-flip view on the E11 layout, and all-window evaluation (62/78/94 windows) [tl, ma-speedy].
- **Raptor intensity-gain TTA (gains 0.9/1.0/1.1, weights 1/6, 4/6, 1/6) dropped LB 0.940 → 0.922** in two separate submissions [pv-bee]. The DINO jitter passes are computed but discarded ("no-jitter frontier") [fm].
- Nobody flips left/right for augmentation or TTA on the CoAt/DINO arms, because laterality is signal [dd-train, pk, st].

### 2.4 Warnings about overfitting gold58 / public LB (direct quotes and numbers)
- "This notebook is now likely overfit to the public leaderboard … followed movements of only 0.001–0.003 … sub-resolution blend tweaks should be treated as noise" [ma-v3].
- A +0.010 gold gain from blending → +0.001 on the LB: "gold-set noise" [dd-inf].
- "58 studies is a small ruler. Differences below roughly 0.02 macro are not measurable" [sl-ds]. Per-label Hanley–McNeil SE ≈ 0.09, 95% CI ≈ ±0.17 [pk]. Macro bootstrap ≈ ±0.04 [st]. Per-label SE ≈ 0.07 [wg].
- Gold prevalence is annotator-selected: every gold study has ≥ 1 positive, mean 4.14 positives per study [sl-ds].
- Every CoAtNet release selects its epochs on gold58, so gold58 is **not** an untouched test set for blend selection. Evaluating deployed fold heads on training rows is contaminated [pv-bee].
- The E9/E10 RadImageNet blend weight was chosen by nested grouped CV on gold58. A censored grid (max 0.25) was later extended; nested selection picked 0.60/0.70 in all folds [ag-e9]. There is a 0.01·alpha penalty in the selection objective [sa-e11 code].
- "The per-label outer map was tuned on the public split … Keep a flat-0.60 version as your second final pick" [sk].
- Gold58 → LB offset: +0.044 for the DINOv2-B family and for a B3 system (0.857 → 0.903) [st]. For CoAtNet single models the offset was much smaller (gold 0.9167 → LB 0.924) [dd-ds:widedense]. **The offset is not constant**, so it is not a reliable LB estimator (my inference from these numbers).
- Report-label OOF understates real image gains: the 224→336 px move was +0.005 OOF but +0.017 LB [st].

---

## 3. Inference engineering

| Item | Fact | Ref |
|---|---|---|
| Hardware | Kaggle **T4 x2**, ~29 GB host RAM, 16 GiB per GPU; internet off; `TIME_BUDGET = 8.0*3600` in several notebooks | jl, fm, sk |
| Full 0.941 stack | "~90 min on T4 x2" | sk |
| Raptor stage profile | 20.8 s one-time + **1.10 s/study** ⇒ 24.3 min for 1,300 studies on a dual-V100 proxy; peak 13.84 GiB/GPU; header cache cut the stage 13.5% | fm |
| gold58 gate | 572.7 s → 358.9 s on 2xV100 after optimisation (37% faster) | fm |
| "< 30 min" full v34 | claimed in a banner | hd (`UNVERIFIED`, marketing) |
| One-arm CoAtNet | the 3-arm blend was 3x the runtime for about +0.001 | dd-inf |
| DINOv2-B 5-fold | ~3 h (224) / ~4 h (336) total; "decode once, score five times" | st |
| Forum profile | ~32 min for ~1,300 studies: **slice ordering (header reads) 12.0 min is "the wall"**, decode+resize 4.1, container/imports 5.0, GPU 10.0 | forum 736678 |

Tricks used:
1. **fp16 autocast on T4.** "T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes ('GET was unable to find an engine')". On any `RuntimeError`, retry in fp32 so a study is never dropped [dd-inf]. The A5 DINOv3 still uses bf16 on T4 (works for ViT matmuls; speed `UNVERIFIED`) [jl].
2. **No DataParallel.** On the full hidden test it caused a **system-RAM OOM** (per-forward module replication). Run arms sequentially so only one model is resident [dd-inf]. For large cohorts, run the 4 CoAt readers one at a time [jl v10].
3. `CUDNN_CONV_WSCAP_DBG=1024` before importing torch, which keeps cuDNN workspace under 16 GiB [fm, jl]. `cudnn.benchmark=False` for ragged 94-window chunks [jl], but `True` for fixed shapes [dd-inf].
4. Share work: the frozen DINO prefix runs once for 20 members; forward and reverse MaxSpan share one decode and checkpoint load; slice order and spacing are cached and reused across Raptor views; CPU decode overlaps GPU via a bounded queue; the two T4s run balanced worker processes [jl, fm].
5. **Never glob the competition DICOM tree** when searching for weights; it walks ~800k files [dd-inf, fm].
6. Fallbacks:
   - Per-study try/except. The fallback is 0.5 [dd-inf] or training prevalence [st].
   - A valid `submission.csv` is written first and overwritten later [pk, wg].
   - Checkpoint the submission every 25 studies, with graceful degradation 32→16 slices and 5→3 folds when the projected runtime exceeds 8 h [pv-report on the B3 system].
   - The top stacks instead make **model failures fatal** and study fallbacks logged, and hash-pin every artifact [ma-speedy, tl]. A missing attachment silently dropped 20 members (0.940 → 0.937) [sk].
7. A timed asynchronous traceback dump caused a **segfault** in Kaggle serving [tl].
8. Test data: the visible `test.csv` has 3 rows, and the rerun swaps in the hidden set [pk, wg, rn].

---

## 4. Preprocessing facts in the code

- **Slice order:** project `ImagePositionPatient` onto the normal `cross(IOP[:3], IOP[3:])`. Fallback `InstanceNumber` [dd-inf, pk, rt-train, wg].
  - Filenames are SOP UIDs, which are random: the rank correlation of filename order with position is ≈ 0.01 [pk].
  - The B3 system (LB 0.903) still sorted by `InstanceNumber` [pv-report]. Pilkwang warns that interleaved and multi-echo series break that.
- **Intensity:** `RescaleSlope`/`RescaleIntercept` via `apply_modality_lut` or by hand, invert `MONOCHROME1`, then a percentile window:
  - Raptor: 2–98 per series, over the sampled slices [dd-inf].
  - Pilkwang: 1–99 per series, "so slices keep their relative contrast" [pk].
  - romantamrazov: 1–99 per slice [rt-train].
  - Will argues per-series matters for Effusion, Synovitis and Baker's because "the brightness of fluid is the finding" [wg].
- **Physical crop by `PixelSpacing`:** 140 mm (Raptor), 130 mm (DINO, E11, Mattia CoAts), 150 mm (romantamrazov). Pilkwang says 130 mm is "below the acquired field of view of almost every series"; E11 says 99.6% [pk, sa-e11].
  - Will reports every series is 150 mm FOV across 960/896/800/640 matrices (0.156–0.234 mm/px), measured on 2026-08-05 on a sample (`UNVERIFIED` for the full set) [wg].
  - Nyquist argument: a 1 mm tear needs ≤ 0.5 mm/px. At 130 mm, 224 px gives 0.58 mm/px (too coarse) and 336 px gives 0.39 [pk]. Will measured that 448 px beats 336 on no lesion width [wg].
- **Laterality canonicalisation** (Pilkwang, sadamtorres; *not* Raptor):
  - `Laterality` is empty in ~47–72% of studies, by vendor [pk, st, wg].
  - Infer the side from the sign of the x coordinate of the **image centre** = IPP + row·Δc·Nc/2 + col·Δr·Nr/2. Using the corner drops agreement with the tag from 96.9% to 58.8% [st].
  - Dead zone 20–25 mm, where the side is left unresolved [pk, st].
  - Flip coronal and axial left knees horizontally; reverse the sagittal slice order so that slice 0 is medial [st, pk].
  - About 7 training studies are bilateral [forum 735639].
- **Flags:** `Fluid_Sensitive` == `Fat_Suppression` on all 24,371 series [wg, pk]. Pilkwang recovers T1/PD/T2 and fat-sat from `SeriesDescription`/`SequenceName`/`ScanOptions`/`ScanningSequence` and from TR/TE (T1 when TR ≲ 800 ms; T2 when TR ≳ 800 and TE ≳ 60; PD when TR ≳ 800 and TE ≲ 60).
  - Slot fill with contrast-defined slots: Ax-fluid 100%, Sag-nonfluid 96.8%, Cor-fluid 96.4%, Sag-fluid 94.2%, Cor-nonfluid 77.3%, **Ax-nonfluid 19.4%** [wg].
  - Global96's population audit shows a mean of 4.65 of 5 nominal slots present [ma-g96].
- **Series choice rules:**
  - Raptor: first unused series of the plane with the preferred fluid flag, else any unused series in the plane [dd-inf].
  - romantamrazov: the best series per plane sorted by (fluid, fat), then one per (fluid, fat) combination, up to 8 series [rt-train].
  - Pilkwang does not fill a slot with a mismatched contrast (`slot_fallback False` natively) [pk].
- **Missing series:** zero-filled and masked (Raptor, Global96 "fixed_width_zero_fill"), or key-padding-masked in the attention (DINO, Rad). "Never allow an all-masked study" [pv-report].
- **Transfer syntaxes:** JPEG Lossless and JPEG 2000 need the pylibjpeg trio or GDCM installed offline from wheels [jirkaborovec, st]. sadamtorres found every *sampled* training file uncompressed Explicit VR (`UNVERIFIED` for the full set).
- **Other tags read:** Rows, Columns, `ImageLaterality`, `Laterality`, `SliceLocation` (fallback), BodyPartExamined, Manufacturer, MagneticFieldStrength, SliceThickness, SpacingBetweenSlices [stevenleehans-101, h17ann]. No notebook uses demographics, and patient age/sex tags are only probed in EDA.
- **Duplicate reports:** 49 report texts are shared by 183 studies, the largest block 37. **GroupKFold by report hash** everywhere serious [wg, pk, sa-e11].

---

## 5. Evidence on ViT-from-scratch / random-init / SSL vs pretrained

**There is no published random-init result in any of these notebooks or datasets.** What exists:

1. The Raptor trainer supports an SSL-ViT-S init and a random-init control, but the author shipped only ImageNet-initialised arms, and no numbers for the SSL or random-init runs are published [dd-train]. Circumstantial reading (`UNVERIFIED`): if the knee-SSL ViT-S had beaten ImageNet CoAtNet, it would likely have shipped.
2. **Domain adaptation > frozen features** [st]. Frozen DINOv2-B gave gold 0.771 / LB 0.776. Fine-tuning at the *same* 224 px gave gold 0.824 / LB 0.866, with Medial Meniscus +0.171, MCL +0.118, ACL +0.113 and Contusion +0.099. Resolution (224 → 336) added only +0.017 LB.
3. Pilkwang and Will both argue for freezing the early ViT blocks and adapting only the late ones ("enough supervision to damage them") [pk, wg]. Pilkwang trains the last 6 of 12 DINOv2-S blocks at LR 8e-6.
4. **Medical-domain pretraining (RadImageNet, frozen)** reached only gold ~0.84–0.85 and needed ImageNet-trained arms to be useful. Natural-image-pretrained CoAtNet fine-tuned end to end reached 0.90–0.93 gold [ag-e9, ma-g96].
5. **Among pretrained backbones, pure-attention MaxViT was worst (0.8438 on gold45)** and the conv-attention hybrid CoAtNet best (0.9025). Swin-B was 0.8825 [dd-inf]. This hints that on this data size convolutional inductive bias helps (my inference; `UNVERIFIED` causally).
6. Data-size arithmetic (my inference): ~4,407 studies × ~5.5 series × ~34 slices ≈ 0.8M 2D slices (consistent with the 819,640 files of train+test). That is ImageNet-1k scale in image count but with far less diversity, and only 4,349 *study-level* noisy labels. From-scratch ViTs historically need either ImageNet-21k-scale supervised data or strong SSL; `UNVERIFIED` for this dataset.

**Implication for Dom:** the realistic path for a "from scratch" ViT is **SSL-pretraining on the ~0.8M training slices (test DICOMs are only visible inside the scoring rerun, so they cannot be used)**, followed by fine-tuning with the proven head and labels. The scratch ViT should be benchmarked against the published ImageNet-CoAtNet numbers on the *same* protocol, and it should keep a random-init control, which the Raptor author built but never published.

---

## 6. Per-finding weaknesses and fixes

Gold58 per-finding for three references:

| Finding | DINO "E2" parent [ag-e9] | RadImageNet [ag-e9] | Resgated e4 [ma-rg] | Global96 top-3 [ma-g96] | LLM labels vs gold, Steven v2 [sl-ds] |
|---|---|---|---|---|---|
| ACL | 0.869 | 0.874 | 0.949 | 0.973 | 0.993 |
| MCL | 0.884 | 0.864 | 0.814 | 0.977 | 0.964 |
| Medial Meniscus | 0.850 | 0.772 | 0.959 | 0.975 | — |
| **Lateral Meniscus** | **0.642** | 0.722 | 0.867 | **0.880** | — |
| Medial OA | 0.950 | 0.949 | 0.971 | 0.983 | — |
| **Lateral OA** | **0.708** | 0.812 | 0.841 | **0.865** | — |
| **PF OA** | 0.801 | 0.802 | 0.813 | **0.885** | — |
| Effusion | 0.943 | 0.907 | 0.961 | 0.973 | — |
| **Synovitis** | 0.742 | 0.768 | 0.830 | **0.810** | 0.790 (v1 0.678) |
| Baker's | 0.955 | 0.899 | 0.969 | 0.976 | — |
| Contusion | 0.842 | 0.901 | 0.912 | 0.912 | — |
| **Fracture** | 0.894 | 0.835 | 0.918 | 0.957 | **0.793** |

(All values are gold58, so each carries per-label SE ≈ 0.07–0.09. The CoAt columns are epoch-selected on gold and therefore optimistic.)

- **Hardest: Synovitis, Lateral OA, Lateral Meniscus, PF OA.** Fracture is hard as a *label* problem.
- **Fixes that were tried and reported:**
  - **Synovitis labels:** 83.7% of LLM cells were "not addressed". Filling undecided Synovitis cells from Effusion (P(syn|eff) = 0.63 vs 0.22 without) raised label AUC 0.678 → 0.790 [sl-ds]. B3's labeller broadened Synovitis to indirect inflammatory language [pv-report].
  - **Lateral Meniscus and MCL / peripheral structures:** widen the slice span (15–85% → 6–94% → 2–98%) *and* add slices. "The outer slices are where the collateral ligaments and the lateral meniscus sit". Corpus expansion gave Lat Meniscus +0.071, Fracture +0.057, Lat OA +0.048, Med Meniscus +0.035 and ACL +0.028 on gold [dd-inf].
  - **Lateral Meniscus routing:** CoAt-only (weight 1.00) at the ensemble level [fm, tl]. The meniscus specialists / residual notebooks exist [rn], but their gains are not stated.
  - **Laterality-dependent findings** (Med/Lat Meniscus, Med/Lat OA, MCL): canonicalise left/right [pk, st]. Without it "you lose a third of the score without a symptom" [st] (argument, not a measured ablation).
  - **Fracture:** "97% of studies share one value … the prior assigned when the report does not mention fractures" with a regex/LLM reader, so it is a label problem [st]. Fix: make avulsion/insufficiency fractures explicit in the LLM prompt [pv-report], and exclude weak arms from the Fracture blend [ag-e9].
  - **Contusion:** the B3 labeller stopped counting degenerative marrow oedema as contusion without a traumatic pattern [pv-report].
  - **Two LLM labellers beat one:** Steven v2 plus lixin73 gives 0.8927 vs gold. The weakest standalone labeller (0.835) added the most because it was least correlated [sl-ds]. Regex scores 0.8136 [sl-ds]; Claude Opus 5 labels +0.027 agreement over rules, mostly non-English recall [wg].

---

## Lineage map

```
LABELS
  pilkwang rule extractor + pilkwang/rsna-knee-llm-labels ─┐
  stevenleehans/rsna-knee-llm-report-labels (v2, v4 blend)─┼─> soft targets used by every arm
  lixin73 labels_llm_gpt56sol ─────────────────────────────┘

DINO/Rad lineage (Apache-2.0 fork chain)
  pilkwang/rsna-knee-baseline-v1  (20 DINOv2-S members; 0.891 LB)
    └─ prvsiyan/read-the-report-then-the-knee  (SlotHead; +RadImageNet V52 heads; 0.906)
         └─ antoinegg1 E9/E10 heads, sofiaanjenje E11/E13 train notebooks (Rad heads, alpha 0.35→0.60)
              └─ sofiaanjenje/frontier-v46 ("V41: DINOv3 + E10 alpha 0.60")
         └─ mattiaangeli/bend-the-knee-to-dinov3-ensembled  (+ A5 DINOv3 5 folds; warns of LB overfit)
              └─ evgendvorkin/rsna-baseline (v13–v15: + Raptor CoAtNet views; 0.940–0.941)
                   ├─ tonylica/dino-radimagenet-rank-ensemble (one-dataset repro of v15; 0.941)
                   ├─ nathanjacob/4-arm-ensemble-explained (English translation; 0.937)
                   └─ mattiaangeli/bend-the-knee-to-the-dinosaurs (v27 0.940: + resgated CoAt)
                        ├─ maverickss26/0941-restructured, renta0426/hybrid-raptor (probe22 map; 0.941)
                        ├─ prvsiyan head-and-shoulders / bee's-knees (audit + 5 failed variants)
                        ├─ starkhushi/why-forks-stop-at-0.941
                        └─ mattiaangeli/speedy-raptors-the-original (v34 0.943: + D4, Global96, Repair-v1)
                             ├─ jiweiliu/fast-2xt4-inference (speed only; also derived from maverickss26)
                             │    └─ fuadmuhammad/fast-parent-0-957-v2 (readable edition; "0.957385" is a LOCAL diagnostic, not LB)
                             ├─ haideptry/speedy-raptors-coatnet-d4-0943 (restyled fork)
                             └─ romantamrazov/dinosaur-v5 (+ own ConvNeXt-T arm from v5-train, V16 SlotHead)

CoAtNet lineage
  dreaddevelopment corpus (44x336) → arms (3 backbones) → arms-x → widedense (0.924) → native384 (0.926)
    → maxspan (0.928) → native384dense → finespacing (0.932)
  mattiaangeli CoAt readers (resgated, rank8/D4, Global96, Repair-v1) built on the same backbone,
    130 mm "smart384" cache, own trainers (published inside the datasets)

Independent single-family notebooks
  sadamtorres DINOv2-B fine-tune (0.883), wguesdon frozen DINOv2-S at 336 (unscored), Yash B3 (0.903, via pv-report)
```

Forks that change nothing material: haideptry (styling), fuadmuhammad (readable edition, flat 0.60 "parent" routing), nathanjacob (translation), prvsiyan H&S/Bee (audit only), renta0426 hybrid (probe22 routing), sofiaanjenje e11 vs e13 (differ **only** in the Rad slot layout: FS slots vs non-FS + one FS anchor).

---

## What a from-scratch ViT must match or beat

Targets on the same protocol (4,349 soft-labelled studies, gold58 held out, no gold in training):

| Bar | Public number | Why it is the bar |
|---|---|---|
| Minimum viable single model | gold ≈ 0.85–0.86, LB ≈ 0.90 | EfficientNet-B3 MIL 0.857 gold / 0.903 LB; DINOv2-B FT 0.883 LB |
| Competitive single model | gold ≈ 0.92, LB ≈ 0.924–0.932 | Raptor CoAtNet-384 single model |
| Best public single model | gold 0.927–0.931 (epoch-selected) | Global96 CoAtNet |
| Public ensemble | LB 0.941–0.943, ~90 min on T4x2 | Speedy Raptors stack |
| Top of LB / efficiency #1 | 0.958 public | eff-lb (runtime not visible) |

Per-finding bars (Global96, gold58): ACL 0.97, MCL 0.98, Med Men 0.98, **Lat Men 0.88**, Med OA 0.98, **Lat OA 0.87**, **PF OA 0.89**, Effusion 0.97, **Synovitis 0.81**, Baker's 0.98, Contusion 0.91, Fracture 0.96.

Compute bars: Global96 trains in ~2.85 h on an RTX 5090 (24 epochs, ~94 GiB 384-px cache). Raptor trains in ~3 h on a 4090 (16 epochs, 23.7 GB VRAM at bs 8 with grad checkpointing; bs 2–4 on 16 GB). Dom's RTX 5070 has 12 GB and 16 GB system RAM, so plan for micro-batch 1–2 studies with accumulation, memmapped uint8 caches, and fewer training windows (K = 8–12).

Inference bar for the efficiency track: about 1.1 s/study for one CoAtNet-384 at 62 windows on a V100-class proxy [fm]. Header reading and slice ordering can dominate, at ~12 of 32 minutes [forum 736678].

---

## Transferable design decisions

These are adoptable regardless of backbone. Evidence strength is noted.

1. **Labels:** LLM-read soft labels from ≥ 2 independent readers, blended in probability space. Keep 0.5 or "not addressed" as low-weight, with weight = 2·|p − 0.5| or an agreement × certainty scheme [sl-ds, sa-e11]. Fix Synovitis via Effusion and make Fracture/Contusion explicit. *Strong evidence (+0.013 gold, label AUC 0.81 → 0.89).*
2. **Never train on gold58.** Use it only as a direction check, and select epochs on it at most. Build a large grouped holdout on report labels for ranking runs. **GroupKFold by report-text hash** [pk, sa-e11, wg].
3. **2.5D input:** 3 physically adjacent slices as channels. Sort by IPP·normal. Apply modality LUT and invert MONOCHROME1. Use a per-series 1–99 or 2–98 percentile window. **Crop a fixed mm FOV (130–140 mm) using PixelSpacing**, then resize to 336–384 px (≤ 0.4 mm/px). *Strong: all top arms.*
4. **Slot layout:** 5 fixed slots (sag-fluid, sag-other, cor-fluid, cor-other, axial) in fixed order with presence masks. **Dense sampling over a wide span (2–98%, 64–96 slices per study)**, evaluating every window position. *Strong: +0.004 to +0.018 LB from sampling alone.*
5. **Head:** per-finding attention MIL over windows (12 separate softmaxes). Optionally add finding queries over slot and depth tokens, slot and depth embeddings, and a GAP-residual "safe start" [dd, ma-g96, pk]. For a ViT this maps naturally to **12 finding query tokens cross-attending over per-window CLS/patch tokens** with slot and slice-position embeddings.
6. **Loss:** confidence-weighted soft BCE + a small macro pairwise rank term (weight 0.05–0.12). Optionally pos_weight clip(1..10) [ma-g96, rt-train, dd-train].
7. **Optimisation:** AdamW, backbone LR ≈ 3e-5 (≈ 8e-6 for a partially frozen ViT), head LR 1e-3, wd 0.02, 15% warmup then cosine/OneCycle, grad clip, effective batch ~24 studies with K = 12 train windows, 16–24 epochs, top-3 epoch probability mean or SWA [dd-train, ma-g96].
8. **Augmentation:** mild shared intensity jitter (±10%), small rotate/scale/shift. **No left-right flips.** Intensity-gain TTA hurt badly (−0.018 LB) [pv-bee]. Canonicalise left/right knees geometrically if the model must learn medial vs lateral.
9. **Ensembling:** rank-average across families. Probability-mean within one run's epochs. Expect ~+0.001 from backbone diversity at fixed data, so **spend runtime on better data geometry before more members** [dd-ds:arms]. Keep one flat-weight final pick to hedge against public-split tuning [sk].
10. **Inference:** fp16 with fp32 retry on T4 (not bf16 convs). No DataParallel, run models sequentially, cache header ordering, overlap CPU decode with GPU, never walk `/kaggle/input` competition dirs, and write a valid submission early. For the efficiency track, header reads and slice ordering are the bottleneck, not the GPU.
11. **Licences:** RadImageNet weights are CC-BY-NC-SA-4.0, and DINOv3 has its own licence. The winner obligations require CC-BY-NC 4.0 and OSI code. Pretrained-weight licences are exempt from relicensing per the rules summary in CLAUDE.md, but check before depending on them.

## Open / UNVERIFIED items to chase
- How Raptor SSL was trained (method, data, epochs, ViT size) and any random-init vs SSL vs ImageNet numbers. Check forum thread 737696 and ask the author.
- Repair-v1 architecture and training; DINOv3 A5 per-fold gold/LB; D4 standalone gold/LB.
- Label construction behind `targets_dualsharp_soft.csv` (Mattia) and the Raptor labels parquet (unpublished).
- Hidden test size (1,322 per renta0426) and whether any test DICOMs use JPEG 2000.
- Runtime of the 0.958 efficiency-#1 submission (not in `eff-lb`).
