# Data and DICOM facts from public Kaggle notebooks

Compiled 2026-09-24 from public notebooks for `rsna-knee-abnormality-detection`. Sources were the notebook code (`kaggle kernels pull`) and the **saved run logs** (`kaggle kernels output ... --file-pattern '.*\.log$'`). The saved `.ipynb` files had no cell outputs, so every printed number below comes from a kernel log or a small output file (parquet/json/csv). We did **not** download any competition data.

Citation keys (all at `kaggle.com/code/<key>`):

| Key | Notebook |
|---|---|
| **[101]** | stevenleehans/rsna-knee-101-read-this-before-you-start (full-corpus run, T4) |
| **[CACHE]** | stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache (full-corpus run, CPU kernel) |
| **[H17]** | h17ann/2-rsna-knee-dicom-metadata-explorer (full header dump + parquet of 300 random series / 10,387 slices; we re-analysed the parquet) |
| **[NJ]** | nathanjacob/know-your-knee-eda-dicom-deep-dive-and-baseline |
| **[KITO]** | kitopl/mri-eda (Japanese) |
| **[RR]** | romanrozen/rsna-knee-data-structure-eda-baseline |
| **[SK]** | starkhushi/rsna-knee-eda |
| **[JB]** | jirkaborovec/kneeabdet-12-label-mri-screening-with-ptl-timm |
| **[BARUN]** | barun2104/baseline-rsna-knee-mri-2-5d-resnet34 |
| **[WL]** | wl1982/axial-only-auto-encoder-100-epochs-16k-bn-persi |
| Extra, found by searching `kaggle kernels list`: | |
| **[PROBE]** | zhukovoleksiy/rsna-metadata-probe (tag inventory of 12 train and 12 test files; full 24,371-series header pass) |
| **[GC]** | gchauhan/eda-rsna-knee-what-the-data-tells-us (header scan of 73,113 files) |
| **[WG]** | wguesdon/rsna-knee-dinov2-at-meniscus-resolution (full-corpus encode) |
| **[WGM]** | wguesdon/rsna-knee-baseline-metadata-only-submission |
| **[SAD]** | sadamtorres/domain-adaptation-beats-resolution-dinov2-on-knee |
| **[HAMDI]** | hamdimohamed22/rsna-dataset-discovery-eda |
| **[SIVA]** | sivapnlr80/rsna-knee-fast-robust-dicom-loader |
| **[NEK]** | nekkon/knee-mri-58-labels-for-4-407-studies |
| **[LIAM]** | liamob96/knee-abnormality-eda |
| **[KA]** | karnakbaevarthur/rsna-knee-eda-to-2-5d |
| **[NM]** | noussairmighri/rsna-knee-eda |
| **[PILK]** | pilkwang/rsna-knee-baseline-v1 |
| **[RENTA]** | renta0426/rsna-knee-0-937-weak-label-dinov2-meniscus-resid |

Numbers from a *sample* say so, with the sample size. Where notebooks disagree, both values are given.

---

## 1. DICOM tags (the "86-tag allowlist")

### 1.1 Tags actually observed

No notebook prints the organizers' list of 86. The best inventory is **[PROBE]**, which iterated over every element of 12 train files and 12 test files (one mid-series file per series, excluding PixelData):

- **train files: 67 distinct keywords; test files: 64.**
- Present in the test sample (64): `AcquisitionMatrix, AcquisitionNumber, BitsAllocated, BitsStored, BodyPartExamined, Columns, ContrastBolusAgent, EchoNumbers, EchoTime, EchoTrainLength, FlipAngle, FrameOfReferenceUID, HighBit, ImageOrientationPatient, ImagePositionPatient, ImageType, ImagingFrequency, InPlanePhaseEncodingDirection, InStackPositionNumber, InstanceNumber, InversionTime, Laterality, MRAcquisitionType, MagneticFieldStrength, Manufacturer, ManufacturerModelName, Modality, NumberOfAverages, NumberOfPhaseEncodingSteps, NumberOfTemporalPositions, PatientID, PatientPosition, PatientSex, PercentPhaseFieldOfView, PercentSampling, PhotometricInterpretation, PixelBandwidth, PixelRepresentation, PixelSpacing, ReceiveCoilName, RepetitionTime, RescaleIntercept, RescaleSlope, Rows, SOPClassUID, SOPInstanceUID, SamplesPerPixel, ScanOptions, ScanningSequence, SequenceName, SequenceVariant, SeriesDescription, SeriesInstanceUID, SeriesNumber, SliceLocation, SliceThickness, SoftwareVersions, SpacingBetweenSlices, StackID, StudyInstanceUID, TransmitCoilName, VariableFlipAngleFlag, WindowCenter, WindowWidth`
- In the train sample but not the test sample: `DiffusionBValue, DiffusionGradientOrientation, NumberOfFrames, PlanarConfiguration, RescaleType`. (These are vendor-dependent, e.g. Philips, not a train/test policy difference. Our reading, UNVERIFIED.)
- Union observed = **69 keywords + PixelData = 70 data elements**, plus 5 file-meta elements (group 0002: group length, version, media storage SOP class/instance UID, **TransferSyntaxUID**) seen in the full dump in **[H17]**. That leaves **about 11–16 of the 86 unobserved** (depending on whether the organizers count file-meta). Those are probably vendor-specific tags that did not appear in 24 files. UNVERIFIED.
- Two full single-file header dumps agree with this list: **[H17]** (Philips Ingenia, 1.5 T, COR T1) and **[NJ]** (Philips Ingenia, STIR axial, which also has `InversionTime` = 140).
- `ImageLaterality` is read defensively by several pipelines (e.g. [101]'s `image_laterality` counter) but **was not seen** in any inventory. `SeriesTime`/`AcquisitionTime` are requested by [GC], but nobody printed whether they exist. Both UNVERIFIED.

### 1.2 Specific tags you asked about

| Tag | Present? | Evidence / notes |
|---|---|---|
| ImageOrientationPatient | **Yes** | [PROBE] list; [H17]/[NJ] dumps. Ordering by IPP·(row×col normal) succeeded on **100%** of the 23,089 decoded slot-series ([CACHE], [101] logs: "slice ordering geometry 100.0%"). |
| ImagePositionPatient | **Yes** | Same as above. |
| PixelSpacing | **Yes** | Missing 0% in [H17]'s 300-series sample. 100% of sampled series have square pixels (row = col spacing) [H17]. |
| SliceThickness | **Yes** | Missing 0% [H17]. |
| SpacingBetweenSlices | **Yes, but sometimes absent** | Missing **5.7%** of series [H17]. |
| RescaleSlope / RescaleIntercept | **Yes** | Philips files have non-unit slopes (2.31477, 2.41245, 2.55507), intercept 0, `RescaleType = 'normalized'` ([H17], [NJ], [101]). |
| PhotometricInterpretation | **Yes** | **0 MONOCHROME1** among 207,801 decoded slices ([CACHE] and [101] full-corpus logs). |
| Rows / Columns | **Yes** | See §3. |
| BitsStored | **Yes** | 12 in both dumps (BitsAllocated 16, HighBit 11, PixelRepresentation 0). No corpus-wide distribution was printed. |
| Manufacturer | **Yes** | See §3. `ManufacturerModelName`, `SoftwareVersions`, `ReceiveCoilName`, `TransmitCoilName` also present. |
| MagneticFieldStrength | **Yes** | Missing 4.3% of series [H17]. |
| EchoTime / RepetitionTime | **Yes** | Used to recover T1/PD/T2 contrast ([101], [KITO], [RR]). Missing for some Toshiba series ([NEK] sample shows "—"). |
| InversionTime | **Yes (STIR/IR only)** | Present in [NJ]'s STIR dump and in [PROBE]'s list. |
| SeriesDescription | **Yes** | **14.7%** of series names are anonymized/uninformative [KITO]. Examples: 'COR T1 CS', 'DP SPIR CS_SAG', 'STIR CS_ AX', 'pd_tse_fs_sag_np', 'L- Sag PD FSE FS'. |
| PatientSex | **Yes** | Coverage **94.6%** of studies [PROBE]. 300-series sample: M 138, F 135, O 14, missing 13 [H17]. `PatientSex` is not a train.csv column [KA]. |
| PatientAge | **Not seen** | Not in any tag inventory and no notebook mentions it. Probably stripped. UNVERIFIED. |
| PatientID | Yes (anonymized, e.g. '60b316da-6c3') | 4,407 patients means **exactly 1 study per patient** [GC]. Every gold study has a distinct PatientID [LIAM]. |
| Laterality | **Yes, but often empty** | See §3.7. |
| BodyPartExamined | Yes | KNEE 248, EXTREMITY 27, missing 16, '' 5, **WRIST 3**, PATELLA 1 (300 series, [H17]). Useless as a feature. |
| Overlays (60xx groups) | **Not seen** | No overlay element appeared in [PROBE]'s full-element iteration of 24 files or in the two full dumps. No notebook checks the full corpus. Almost certainly stripped by the allowlist, but UNVERIFIED corpus-wide. |
| Other useful ones | Yes | ScanningSequence (SE 270, GR 9, IR 5, RM 3, missing 13), ScanOptions (fat-sat token, e.g. `FS`), MRAcquisitionType (2D 275 / 3D 12 / missing 13), SliceLocation, InstanceNumber, WindowCenter/Width, ImagingFrequency, FlipAngle, EchoTrainLength, PixelBandwidth, AcquisitionMatrix. (Counts are for 300 series, [H17].) |

The same **13 of 300** series (4.3%) are missing PatientSex, ScanningSequence, MRAcquisitionType and MagneticFieldStrength together [H17 parquet]. That pattern points to one vendor or site stripping them, likely Toshiba/Canon. UNVERIFIED.

---

## 2. Transfer syntaxes, decoders, decode speed, odd files

### 2.1 Distribution: the measurements contradict the Data page

Every measured sample is **100% Explicit VR Little Endian (1.2.840.10008.1.2.1)**:
- [SIVA]: **400/400** files (80 random series × 5 files), all decoded OK.
- [HAMDI]: 15/15 files.
- [SAD]: "every sampled training file was Explicit VR Little Endian, uncompressed". Its test census on the visible test set: 3/3 Explicit VR LE.
- Full-corpus decodes ran with **zero failures** and no extra decoder installs: [CACHE] **207,801 slices, 0 decode failures**; [101] the same count, 0 failures; [WG] **190,849 slices, pixel_fail 0**. All used plain `pydicom 3.0.2` on the stock Kaggle image.
- Claims that JPEG Lossless, JPEG 2000 and Implicit VR are present ([JB] says "a large share of series raise", [SIVA] says "~a third of files", [WG]/[WGM] mention the mix) are **unmeasured restatements of the Data page**. [SIVA]'s own printed census contradicts its TL;DR.
- **Conclusion:** compressed syntaxes are absent or very rare in train. They may still exist in the hidden test set, which spans the same ~19+3 sites. So treat decoders as **insurance** for inference. UNVERIFIED for the hidden test set.
- Caveat: [SIVA] asserts the stock Kaggle image already ships `pylibjpeg` and `gdcm`. That is UNVERIFIED (its own sample needed no decoder). If true, the zero-failure runs would not prove the files are uncompressed.

### 2.2 Decoders offline

- [SAD] installs `pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm` with internet off: `pip install --no-index --find-links /kaggle/input/datasets/sadamtorres/rsna-wheels ...`. This proves the approach works with a private or public **wheels dataset**.
- [JB] builds wheels with `pip download ... --dest frozen_packages/` in an internet-on notebook, whose output then becomes a dataset. Its outputs contain cp312 manylinux wheels.
- Kaggle environment seen in logs: Python **3.12.13**, pydicom **3.0.2**, torch **2.10.0+cu128**, timm **1.0.26**, numpy 2.0.2 [JB]. The current PyTorch dropped **P100** support, so use **T4 ×2** [SK]. A weekly GPU quota of **30 h** is claimed in [CACHE] (UNVERIFIED).

### 2.3 Decode speed (Kaggle hardware)

| Source | Setup | Throughput |
|---|---|---|
| [CACHE] | CPU-only kernel, 12 pixel threads, 16 header threads; 207,801 slices | Decoding 4,128 s ≈ **~50 slices/s** (≈ **6.25 slot-series/s**). Markdown says 64.2 min wall; the saved log says 71.3 min (different runs). Writing 11 GiB took only ~0.6% of the runtime (shards write at ~470 MB/s), so the job is **decode-bound**. |
| [CACHE] | Same decode on the GPU machine | "roughly 55 min", so the accelerator does not help decode. |
| [101] | T4 machine, same pipeline | 207,801 slices in ~3,235 s ≈ **64 slices/s**. Header pass over 24,371 series: ~100 s. |
| [WG] | 4 IO workers + DINOv2-S at 336 px | **1.09 studies/s** (~43 slices/study) incl. GPU. Probe: decode+preprocess **0.91 s/study**, GPU forward 0.24 s/study. Projected **~1,500 s for ~1,300 test studies**. |
| [SIVA] | 553 slices, ThreadPool | 1 worker 81/s; 2 workers **302/s**; 4: 248/s; 8: 209/s. The set is tiny and page-cache effects are likely, so treat it as an upper bound. |
| [WGM] | header-only reads | 0.333 s/study projected (≤32 header reads/study). |
| [PROBE] | full header pass | 24,371 series headers in ~124 s. |
| [GC] | header scan | 73,113 files in 2.7 min (ProcessPool). |
| [101] | header pass | 2,188 series headers from 400 studies in 6 s, 0 failures. |

### 2.4 Odd files

- No corrupt files reported: 0 decode failures over the full corpus ([CACHE], [101], [WG]) and 0 header errors in 200 series ([LIAM]).
- Every train.csv study has a folder and vice versa (4,407 = 4,407) [HAMDI].
- IDs are unique [JB].
- Filenames are SOP Instance UIDs and are **random**: filename order matches physical order for only **~5% of slices** [101].
- **`InstanceNumber` runs against slice position in over a third of series** [GC].
- `ImagingFrequency` takes 8,023 distinct values and varies *within* 3,978 of 4,407 studies. It is re-tuned per acquisition, so it is not a scanner ID [KITO].

---

## 3. Dataset statistics

### 3.1 Sizes
- **train.csv: 4,407 studies × 14 columns** ([101], [NJ], [KITO], [HAMDI], [JB] all agree). The "7,504" in CLAUDE.md is **wrong**.
- **train_series.csv: 24,371 series**; test_series.csv (visible stub): **15 series / 3 studies**.
- **819,078 train slice files** [GC]. The Data page's 819,640 total presumably adds the CSVs and test stub (not checked).
- train.csv is 5.7 MB and train_series.csv 3.5 MB [101].

### 3.2 Series per study
- 3–14, **mean 5.53, median 5** (25% = 5, 75% = 6) ([101], [SK], [HAMDI]).
- Most common (Sag, Cor, Ax) count layouts: **2/2/1 = 1,785**, 2/1/1 = 582, 2/3/1 = 307, 2/1/2 = 262, 3/1/1 = 233 [NEK].
- **12 distinct protocol signatures** [PROBE]. The top one (Ax-FS, Cor-nonFS, Cor-FS, Sag-nonFS, Sag-FS) covers 2,516 studies, then 745, 566, 229, 157, 120, …

### 3.3 Planes and flags
- Series by plane: **Sagittal 9,864, Coronal 8,609, Axial 5,898** (train). [101] counts 9,871 / 8,613 / 5,902 because it includes the 15 test rows.
- **Every study has all three planes; missing-plane frequency = 0** ([101], [SK], [NEK], [NM]).
- **`Fluid_Sensitive` and `Fat_Suppression` are identical on all 24,371 train rows and on the 15 test rows (24,386 total)**: 14,010 = 1 and 10,361 = 0 ([101], [KITO], [NEK], [HAMDI], [WG]). They carry **one** bit, not two. This contradicts CLAUDE.md's "correlated but not equivalent".
- Studies with a sagittal fluid-sensitive series: **4,150 / 4,407 (94.2%)** ([SK], [JB], [WG]). Studies with no fluid-sensitive series or no T1-type series: 0 [NEK].
- Contrast recovered from the header (TR/TE rule: T1 if TR ≲ 800 ms; T2 if TR ≳ 800 and TE ≳ 60; PD if TR ≳ 800 and TE ≲ 60; GRE via ScanningSequence):
  - [101], 2,188 series from 400 studies: **PD 1,238, T1 477, T2 457, GRE 16**. The (fluid, fat-sat) combos recovered: (F, no-FS) 484, (F, FS) 9, (T, no-FS) 453, (T, FS) 1,242, i.e. **4 combos, where the CSV gives 2**.
  - [KITO], 300 series: PD 165, T1 74, T2 48, unknown 13.
- Header-derived slot fill over all 4,407 studies [WG]: **Axial_Fluid 100%, Sagittal_Structural 96.8%, Coronal_Fluid 96.4%, Sagittal_Fluid 94.2%, Coronal_Structural 77.3%, Axial_Structural 19.5%**.
- [101], 400-study sample: Axial/fluid/FS 98.0%, Sagittal/fluid/FS 95.0%, Coronal/fluid/FS 93.8%, Coronal/structural/no-FS 62.7%, Sagittal/fluid/no-FS 62.3%, Sagittal/structural/no-FS 42.2%. Only 3 of 12 slots are present in >80% of studies.
- A repeated series type is mostly a **second contrast**, rarely a retake [GC].
- String traps [CACHE]/[RR]/[101]:
  - GE's `SAT_GEMS` means *spatial* saturation, not fat-sat, so match ScanOptions tokens exactly.
  - `we` (water excitation) hides inside `t2_de3d_we_tra` until you normalise underscores.
  - Turkish "sag" = right, and every sagittal protocol is named `sag_...`.

### 3.4 Slices per series (full corpus, [NJ])
- **mean 33.6, median 30**, 25% = 25, 75% = 34, **min 11, max 320**.
- By plane (median / max): **Axial 32 / 254, Coronal 30 / 80, Sagittal 30 / 320**; means 39.8 / 28.3 / 34.6.
- 300-series sample [H17]: 5th pct 19, 95th pct 45, 99th pct 176.
- About 2% of series exceed a 64-slice header cap ([WG]: 561 of 20,501 slot-series needed subsampling).

### 3.5 Image geometry
- **Rows/Columns:** 256–1024 rows and 256–1280 columns in a 300-series sample; median 512 [H17]. Common matrices: **512 (514 series), 384 (299), 640 (288), 320 (144), 1024 (84), 256 (77), 768 (58), 480 (54)** from 2,188 series [101]. [NEK] also sees 192 and 23 distinct sizes. Median by plane (sample): Axial 640×656, Coronal 512×512, Sagittal 512×512 [SK].
- **Pixel spacing:** **0.1367–1.1719 mm**, median **0.3125** mm, mean 0.339 (2,188 series, [101]). The 300-series sample gives median 0.327, 5–95% 0.184–0.566 [H17]. The same range appears in [NJ] (1,644 series) and [KITO].
- **Field of view** (= spacing × columns): median **160 mm**, 5–95% **140–185.7 mm**, range 70.3–320, **71 distinct values**. Most common: 160 mm (809), 170 (281), 180 (251), 150 (239) [101]. [101] explicitly refutes a claim of a constant 150 mm.
- **Slice thickness:** **0.6–5.0 mm**, median **3.0**, mean 3.11. Values: 3.0 (179/300), 4.0 (53), 3.5 (23), 2.5 (19), 2.0 (10), plus a few 0.6 / 0.8 / 0.89 / 1.39 3D series [H17], [NJ], [NEK].
- **SpacingBetweenSlices:** 0.4–6.78 mm, median **3.3**, mean 3.62; missing 5.7% [H17]. It can differ a lot from thickness (e.g. 4 mm thick at 7.718 mm spacing [NJ]). Slices are about 10× farther apart than pixels (~0.3 mm vs ~3.5 mm) [GC].
- **3D acquisitions:** 12/300 series (4%) [H17].

### 3.6 Scanners and sites
- **Manufacturer** (300 series, [H17]; spellings unmerged): Siemens Healthineers 63, SIEMENS 60, Siemens 5 (**Siemens 128**); GE MEDICAL SYSTEMS 59, GEHC 1 (**GE 60**); Philips 50, Philips Medical Systems 44, Philips Healthcare 3 (**Philips 97**); TOSHIBA 8; CANON_MEC 4; Hitachi 2; FUJIFILM 1.
- **Field strength:** 1.5 T 162, 3.0 T 122, **1.16 T 3**, missing 13 [H17].
- **Scanner fingerprints:**
  - [KITO]: Manufacturer | ModelName | SoftwareVersions | FieldStrength gives **112 groups** over 4,407 studies (used for GroupKFold).
  - [PROBE]: adding ImagingFrequency + ReceiveCoilName gives **265 fingerprints**, and the top 20 cover 45.5% of studies.
- **Metadata-only models** [PROBE] (no pixels; targets from report-derived labels):
  - random folds: macro AUC **0.6515**
  - scanner-grouped folds: **0.5981** (gap 0.053 = site memorisation)
  - series composition alone: **0.5954**
  - The header is **not** a meaningful shortcut, but site leakage in random folds is real, so group folds by scanner.
- The number of sites is not recoverable from the headers (no InstitutionName). UNVERIFIED.

### 3.7 Left vs right knee (Laterality)
- Series-level, 2,188 series / 400 studies [101]: **missing 47.9%, L 27.0%, R 25.2%**. 10 of 400 studies have conflicting tags.
- [NJ], 1,644 series: '' 443, L 427, R 399, 'Unknown' 375.
- Study-level, full train [101 log]: tag available for **2,203 (50.0%)**, from report/description text 167 (3.8%), unknown 2,037 (46.2%).
- A geometric rule (sign of the IPP x-coordinate; negative = patient's right in LPS) **agrees with the tag on 2,042 / 2,199 = 92.9%** of tagged studies.
- Slot-series level, full train [WG]: 4,958 left, 2,986 right coronal/axial (mirrored), 2,029 right sagittal (reversed), **10,528 unlabelled (51.4%)**, 0 self-contradictory.
- **Among tagged series, left and right are about 50/50** (590 L vs 551 R [101]; 4,958 L vs 5,015 R [WG]).

### 3.8 Test set
- The visible test is a stub: **3 studies / 15 series** (5 per study). At scoring it becomes **~1,300** studies (Kaggle page; [GC], [WG]).
- The figure **1,322** in [RENTA] ("expands to 1,322 hidden studies") matches `int(0.3 × 4407)`, a sizing assumption coded in [RR]/[PILK]. UNVERIFIED, so do not treat it as a fact.
- In the stub, 1 of 3 studies has a Laterality tag and 2 were resolved from geometry [PILK]. All 3 descriptions/planes also appear in train [NJ].
- Test ground truth ([GC], citing host discussion 733343; UNVERIFIED by us):
  - **two MSK radiologists read each test exam independently and a third adjudicates**
  - borderline = negative
  - ACL, MCL, the three OA findings, Effusion and Baker's count only above size/severity thresholds

---

## 4. Labels (train.csv)

### 4.1 Gold subset
- **58 / 4,407 studies (1.3%)** carry all 12 labels. 4,349 have none, 0 are partial, and all 4,407 have a non-empty Report (every notebook).
- Prevalence in the 58 ([101], [NJ], [HAMDI]):

| Finding | Pos / 58 | Rate |
|---|---|---|
| Effusion | 35 | 60.3% |
| Synovitis | 27 | 46.6% |
| Medial Meniscus | 26 | 44.8% |
| ACL | 24 | 41.4% |
| Lateral Meniscus | 23 | 39.7% |
| PF OA | 21 | 36.2% |
| Contusion | 19 | 32.8% |
| Fracture | 18 | 31.0% |
| Medial OA | 15 | 25.9% |
| Baker's | 12 | 20.7% |
| Lateral OA | 11 | 19.0% |
| MCL | 9 | 15.5% |

- Findings per gold study: **mean 4.14, median 4, min 1, max 9**. There are **no all-normal studies** among the 58 [SK], so the gold set is enriched for pathology.
- **Noise floor of the gold set** [101]: a random model's 95% AUC interval per label is about ±0.15. Examples: MCL [0.28, 0.64], Baker's [0.12, 0.51], Fracture [0.47, 0.77]. [NEK]: a real +0.10 AUC gain is detected as significant only ~17% of the time on n = 58.
- **Co-occurrence (Pearson r on gold)** [NJ], [SK]:
  - Medial OA↔Baker's +0.476, Medial Meniscus↔Medial OA +0.418, Lateral Meniscus↔Lateral OA +0.417, Lateral Meniscus↔Medial OA +0.407, Lateral OA↔Baker's +0.404, Effusion↔Synovitis +0.403, ACL↔Contusion +0.383, Medial OA↔PF OA +0.374.
  - Negative: Medial OA↔Contusion −0.328, PF OA↔Contusion −0.297, ACL↔PF OA −0.269, ACL↔Medial OA −0.256, MCL↔Medial OA −0.253.
- In reports [GC]: Contusion+Fracture co-occur 2.9× chance, Medial+Lateral OA 2.0×. **Fracture appears in 7% of reports but 31% of expert labels.** For about half of expert-positive Synovitis cases the report is silent.

### 4.2 Reports
- Length: **mean 1,098 chars (147.5 words), median 977, range 52–4,743 chars** ([101], [SK], [HAMDI]).
- Languages (counts vary by method):
  - [101] script/keyword detector: English 1,738, Spanish 578, Turkish 542, unclear 533, Greek 321, German 262, Cyrillic 220, Dutch 132, French 81. 60.6% not English.
  - [NEK]: English 1,731, Turkish 889, Spanish 682, unresolved 428, German 258, Greek 178, Dutch 160, French 81.
  - [HAMDI] langdetect sample: en 98, es 47, tr 42, **hr 38, bg 20**, de 18, el 18, nl 12, fr 7. The Croatian/Serbian and Bulgarian/Russian (Cyrillic) cluster is also noted by [SK].
  - 59.3% of reports contain non-ASCII characters [JB].
- **Duplicate reports:** 4,257 distinct texts. **204 studies (4.6%) share text with another study**; the largest group is 37 [101]. [WG]: 49 duplicate blocks covering 183 studies. One report group is shared between a gold and an unlabelled study ([KA], [LIAM]). Use a normalised-report hash as a GroupKFold group ([101]: raw MD5 splits 25 groups / 125 studies across folds; the normalised hash splits 0).
- **Report-derived labels vs gold (58):**
  - [WG] rule extractor: mean agreement **0.796**, precision 0.680, recall 0.765.
  - Claude Opus 5 (low) LLM labels: agreement **0.823**, recall 0.860 (paired Δ +0.027, McNemar p = 0.02) [WG].
  - Per-target agreement ([RR] table): MCL 0.940, ACL 0.930, Baker's 0.916, Medial Meniscus 0.895, Lateral Meniscus 0.865 … Synovitis 0.628. Synovitis is silent in 88% of reports.
- **LLM-label prevalence over all 4,407** (threshold 0.5, [WG] using `labels_llm_opus_2026_08_05`): ACL 17.5%, MCL 13.1%, Medial Meniscus 41.5%, Lateral Meniscus 16.3%, Medial OA 37.2%, Lateral OA 26.2%, PF OA 45.7%, Effusion 51.3%, Synovitis 21.4%, Baker's 24.4%, Contusion 11.6%, Fracture 6.9%. The public label datasets are `pilkwang/rsna-knee-llm-labels` (CC0) and `stevenleehans/rsna-knee-llm-report-labels`.

---

## 5. Caching and preprocessing recipes

### 5.1 The "500 GB → 11 GiB" cache [CACHE]
- **Output:** `uint8` array **4,407 × 6 slots × 9 slices × 224 × 224 = 11.13 GiB**, written as **4 `.npy` shards of 2.78 GiB** plus `_mask.npy` (slot presence) and `_studies.csv` per shard. A `cache_meta.json` fingerprint (img, crop_mm, window, slot order, seed) lets training refuse a mismatched cache.
- The pipeline is sharded and resumable, and it refuses to write truncated shards.
- **Slots:** `SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, SAG_FLUID_NOFS, COR_T1, SAG_T1`, derived from the header (TR/TE + ScanOptions + description), not the CSV. Mean slot coverage **87.3%** (5.24 / 6).
- **Slices per slot:** 3 anchors × 3 *physically adjacent* slices, so each channel triplet is a genuine 2.5D neighbourhood (~10 mm). The saved run used window 0.35–0.65 of the ordered stack. [101] uses per-plane windows Sag/Ax 0.10–0.90, Cor 0.15–0.85.
- **Ordering:** project `ImagePositionPatient` onto the normal of `ImageOrientationPatient`. Fallbacks: SliceLocation, then InstanceNumber, then filename. Geometry worked on 100% of series.
- **Intensity:**
  - apply RescaleSlope/Intercept
  - invert MONOCHROME1 (never triggered)
  - normalise **per series to the 1st–99th percentile over the whole volume** (not per slice, so relative contrast survives), clip to [0, 1], bilinear resize, store `uint8`.
- **Physical-scale crop:**
  - center-crop to a constant **130 mm** before resizing to 224 px (0.58 mm/px, 8.13 mm per 14-px DINOv2 token).
  - The crop is skipped when FOV ≤ crop_mm. At 160 mm (the median FOV) the crop applied to only **38.9%** of slot-series; at 130 mm it applied to **98.5%**.
  - [KITO]'s failure rates by crop size: 160 mm 23.7%, 150 mm 10.7%, 140 mm 5.3%, 130 mm 0%. It uses 130.
- **Cost:** 207,801 slices, 23,089 slot-series, 0 failures. About 64–71 min on one CPU kernel (§2.3), header pass ~110 s. The job is decode-bound, not write-bound.
- [CACHE] also measured that OOF CV **underestimates ensembling**: the LB gain from a 5- or 10-member rank-mean was +0.024 / +0.031 over single-fold OOF.

### 5.2 Other variants
- **[WG]:** 5 contrast-enforced slots (Sag_Fluid 12, Sag_Struct 12, Cor_Fluid 8, Cor_Struct 6, Ax_Fluid 8 slices), 336 px on a 150 mm FOV (0.446 mm/px, 6.25 mm/token), per-series 1.0/99.5 percentile, per-plane windows Sag 0.12–0.88 / Cor 0.20–0.80 / Ax 0.12–0.88. Pixel cache 7.5 GB for 4,407 studies. Each encoded image = the slice plus its two neighbours.
- **[RR]/[PILK]:** memory-sized cache (1 group × 3 slices at 336 px, 12 slices at test). Crop 130 mm. Per-plane bands Sag 0.08–0.92, Cor 0.22–0.78, Ax 0.10–0.90.
- **[JB]:** one series per study (Sag-fluid, then any Sag, then any), 24 evenly spaced slices, 224 px, per-volume percentile scaling then ImageNet mean/std. 63% of voxels fall strictly inside (0, 1).
- **[BARUN]:** a public dataset of 24,371 per-series `.npz` volumes (24 × 224 × 224 uint8).

### 5.3 Orientation standardisation
- No notebook resamples to a canonical patient frame (LPS). They sort slices geometrically and **normalise laterality to a left-knee convention**:
  - coronal/axial: mirror pixels left-right
  - sagittal: **reverse slice order** (the medial-lateral axis runs through the stack)
  - then **disable horizontal-flip augmentation**
- Laterality resolution chain [101]: tag, then majority across the study's series, then SeriesDescription/report text (with language-aware regexes and the Turkish "sag" trap), then IPP-x geometry sign. [WG] leaves untagged series alone ("a wrong flip is worse than no flip").
- In-plane rotation and oblique stacks are not corrected anywhere. Oblique IOP values are common (e.g. [H17]'s coronal has a row vector [0.995, −0.012, 0.0995]).

---

## 6. Self-supervised / autoencoder results [WL]

- **No results are available.** The notebook's latest version is **not** an autoencoder. Its source is a *supervised* axial-only baseline (128 × 128 slices, study labels repeated per slice, top-k = 5 slice pooling, LLM labels thresholded at 0.5). That run **failed** with `SyntaxError: incomplete input` in cell 2 (log).
- The output listing shows files from an earlier version (`knee_autoencoder_weights.pt`, `recon_plot_test_set.png`, `experiment_results.json`, `axial_chunk_{1,2}.pt`), but they could not be downloaded.
- The title ("100 epochs, 16k bottleneck") is the only information about the autoencoder. UNVERIFIED; there are no numbers.
- Related baselines for scale (not SSL):
  - [JB] EfficientNet-B0 attention-MIL trained on the 58 gold only: pooled OOF macro AUROC **0.536**.
  - [BARUN] ResNet-34 2.5D on 46 gold, validated on 12: best val AUC **0.609**.
  - [NJ] ResNet-18 on 1,935 keyword-labelled studies: val AUC **0.672**.
  - [101] DINOv2-S fine-tune (last 6 blocks), 1 fold: **0.756** vs derived labels, 0.785 on 9 gold.
  - [WG] frozen DINOv2 ridge vs fine-tune at 336 px: 0.788 / 0.788 on one fold vs extracted labels.
  - [SAD] fold OOF ≈ 0.72.

---

## Answers to CLAUDE.md open TODOs

1. **Which of the 86 tags exist.**
   - 69 keywords + PixelData + 5 file-meta elements have been observed ([PROBE] inventory, [H17]/[NJ] dumps; full list in §1.1).
   - The rest of the 86 are unobserved, probably vendor-specific. UNVERIFIED.
   - Action: we still need to write `docs/dicom_tags.md` by iterating all elements over a larger stratified sample (e.g. one file from every series, header-only, ~2 min on Kaggle).
2. **Orientation / spacing availability. Yes.**
   - `ImageOrientationPatient`, `ImagePositionPatient`, `PixelSpacing`, `SliceThickness`, `SliceLocation` and `InstanceNumber` are present. Geometric slice ordering succeeded on 100% of 23,089 decoded series.
   - `SpacingBetweenSlices` is missing on ~5.7% of series; derive the spacing from IPP differences instead.
   - `RescaleSlope/Intercept` are present, with non-unit slopes on Philips. `PhotometricInterpretation` is present, with 0 MONOCHROME1 in the corpus.
   - **Do not** sort by filename (~5% agreement) or trust InstanceNumber direction (reversed in >1/3 of series).
3. **Overlays.** No 60xx element was seen in any inventory or dump. Probably stripped by the allowlist. UNVERIFIED corpus-wide; confirm in the tag scan above.
4. **Demographics.**
   - `PatientSex` is present (94.6% of studies; values M/F/O).
   - `PatientAge` / weight / birth date were **not observed**, so assume absent. UNVERIFIED.
   - `PatientID` is present but anonymized, one per study.
   - Given [PROBE]'s result (full header metadata reaches only 0.598 AUC under scanner-grouped CV), demographics are a weak signal at best.
5. **Total train studies: 4,407** (24,371 series, 819,078 slice files). The 58 gold studies make up 1.3%. The "7,504" figure is wrong.
6. **Decoders installable offline. Yes.**
   - Build a wheels dataset (`pip download pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm --dest ...` in an internet-on notebook matching Kaggle's Python 3.12), then `pip install --no-index --find-links /kaggle/input/<wheels-dataset> ...` [SAD], [JB].
   - Evidence says train is essentially all Explicit VR Little Endian (0 decode failures over 207,801 slices on the stock image), so decoders are insurance for the hidden test set.
   - Whether the stock image already includes pylibjpeg/gdcm is UNVERIFIED (claimed by [SIVA]). Check with `import pylibjpeg, gdcm` in a Kaggle notebook.

Also worth correcting in CLAUDE.md:
- `Fluid_Sensitive` and `Fat_Suppression` are **identical** on every row, one bit and not two. Header TR/TE recovers T1/PD/T2.
- The test set is ~1,300 studies (the 1,322 figure is an unverified guess).
