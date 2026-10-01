# Preprocessing + cache (S02)

Code: `src/knee/data/` (`preprocess.py` is the single entry point used by the cache builder **and** the Kaggle notebook).
Run on Windows: `scripts/run_s02.ps1`. Tests: `tests/test_data.py`.

## 1. What one study becomes

| step | rule | where | knob (HP) |
|---|---|---|---|
| read | every file once, whole file in one read, reads overlapped by a per-process thread pool; bytes freed as each slice is decoded; unreadable files logged and skipped | `dicom_io.read_slices` | `KNEE_IO_THREADS` (16) |
| orientation filter | keep slices whose normal is within ~5.7° of the series' majority normal | `preprocess._read_series` | |
| duplicates | positions within 0.05 mm = several stacks/echoes: keep the biggest (EchoNumbers, AcquisitionNumber) group, then one slice per position | same | |
| slice order | ascending IPP · (true slice normal, signed to the plane's canonical normal) | `geometry.canonical_positions` | |
| decode | `pixel_array` → modality LUT (slope/intercept) → float32; MONOCHROME1 inverted | `dicom_io.decode_pixels` | |
| crop | centre crop `crop_mm` × `crop_mm` in native pixels (PixelSpacing per slice) | `crop_resize.crop_mm_native` | `crop_mm` = 130 |
| resize | to `crop_mm / out_px` mm per pixel (INTER_AREA shrinking, LINEAR enlarging); a smaller FOV is **padded, never stretched** | `crop_resize.resize_crop`, `pad_center` | `out_px` = 384 |
| window | per-series 1st–99th percentile of the (unpadded) resized pixels → uint8 | `normalize` | `pct_lo/hi` = 1/99 |
| canonical display | transpose/flip so every plane has one fixed display (below) | `geometry.canonical_inplane_ops` | |
| laterality | tag (if the study's tags agree) → geometry (median image-centre x, dead zone) → `U` (no mirror) | `geometry.resolve_study_laterality` | `lat_dead_zone_mm` = 20 (**placeholder until S01 part 2**) |
| mirror | left knees → right-knee anatomy: flip columns (cor/ax), reverse slice order (sag) | `preprocess._render_series` | `canonicalize_laterality` |
| slots | sag_fluid, sag_other, cor_fluid, cor_other, axial (fluid preferred); within a slot: contrast rank, then more slices, then lower SeriesNumber | `series_select.assign_slots` | the rank tables |

**Canonical frame** (LPS; "columns →" is the direction of increasing column index):

| plane | columns → | rows → | slice 0 → last | right knee reads as |
|---|---|---|---|---|
| Axial | patient Left | Posterior | Inferior → Superior | anterior at top, fibula on the image left, medial on the image right |
| Coronal | patient Left | Inferior | Anterior → Posterior | head at top, medial on the image right |
| Sagittal | Posterior | Inferior | patient Left → Right (= **medial → lateral** for a right knee) | anterior on the left, head at top |

Every study, left or right, is stored as a **right knee**. The model can then learn "medial = image right" once. Do not use
LR-flip augmentation by default (02 §C12).

## 2. On-disk format (`<cache_root>/cache_v1_<split>/`)

- `shards/shard_XXXXX.u8`: raw uint8, shape `(n, out_px, out_px)`, C order, no header. 16 studies per shard; a study's
  series are consecutive, so **one study = one contiguous read**. Open with `np.memmap(f, np.uint8, "r", shape=(n, px, px))`
  where `n = size // px²` (`dataset.CacheReader` does this lazily per worker).
- `series_index.parquet`: one row per series: `study_uid, series_uid, shard, offset, n_slices, slot, plane, plane_csv,
  fluid, fluid_csv, fat_sat_header, contrast_class, tr, te, ti, field_strength, manufacturer, fingerprint,
  series_description, series_number, laterality_tag, center_x, obliquity_deg, slice_spacing_mm, slice_gap_ratio,
  pixel_spacing_mm, n_files, n_off_orientation, n_dup_dropped, n_decode_fail, window_lo, window_hi, transpose, flip_rows,
  flip_cols, mirrored`, plus the list columns `positions_mm` (ascending, mm) and `valid` (per-slice decode mask).
- `study_index.parquet`: `study_uid, laterality, laterality_source, mirrored, n_series, n_slices, slot_<name>` (series uid
  per filled slot), `error, seconds, shard`.
- `cache_manifest.json`: params, version `v1`, git sha, counts, bytes, timings. A build refuses to resume into a cache
  made with different params.

**Size:** 147 KB per slice at 384 px → **≈120 GB for ~819k slices** (projection; the build prints the real figure after each
shard). At 336 px it would be ≈92 GB; at 256 px ≈54 GB.

**Derived lower-resolution copy:** `python -m knee.data.derive_cache --src <cache_v1_train> --px 256` writes
`<cache_v1_train>_256` with the same layout and indexes (manifest `params.out_px` = 256, plus `derived_from`). It uses
the same INTER_AREA resize as `gather_windows`, so Track B with `out_px: 256` reads identical windows from either cache.

## 3. Datasets (`src/knee/data/dataset.py`)

- 2.5D window = slices `[c-1, c, c+1]` of one series as 3 channels (clamped at the ends). Train centres: stratified
  jitter over 2–98% of the stack; eval: evenly spaced (or every slice).
- `StudyWindowsDataset` (Track A): `x` uint8 `(5 slots, W, 3, h, w)` + `slot_mask`, `z` (relative position), labels + mask.
- `StudyBagDataset` (Track B): variable bag over **all** series; per window: `series_idx`, `z`, `z_mm`, and type ids
  `plane/fluid/contrast/field/slot`; `collate_bag` pads and returns `window_mask`.
- Pixels leave the workers as uint8 (convert to float on the GPU). Optional `out_px` resizes at load (INTER_AREA).

## 4. Measurements

### 4a. Mac, visible test stub (2026-09-24, S02 part 1; M3, 8 cores, warm page cache, 3 studies / 14 series / 512 slices)
- Single process: **0.31–1.40 s/study, ≈4.2 ms/slice** (≈240 slices/s). The 1.4 s study has a 160-slice 3D GRE.
- 3 workers: 2.5 s wall for the 3 studies (process start-up dominates at this size).
- DataLoader (warm cache, 2 workers, 384 px): Track A train bs 2 × 5 × 12 windows ≈ 225 studies/s; Track A eval
  5 × 16 ≈ 150 studies/s; Track B train bs 4 ≈ 210 studies/s. With the resize to 224 px, single process: 17–19 studies/s.
- **Not representative of the real build:** the stub is in the page cache. The full build reads ~570 GB from a SATA SSD:
  at ~500 MB/s that alone is ≥ 20 min, so expect it to be disk-bound. Estimate ≈1 h (UNVERIFIED).

### 4b. Windows, full train (fill in from `scripts/run_s02.ps1`)
| measure | value |
|---|---|
| s/study, 1 worker (16 studies) | |
| studies/s and slices/s, N workers | |
| full build wall-clock | |
| cache size on disk | |
| studies with errors / decode failures | |
| laterality: tag / geometry / U counts | |
| DataLoader studies/s (cold), each dataset | |
