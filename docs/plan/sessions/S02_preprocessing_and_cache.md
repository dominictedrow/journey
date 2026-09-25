# S02 — Preprocessing library + cache v1

**Track:** D (shared) · **Machine:** Windows · **GPU:** no (CPU decode; the full build runs overnight)
**Depends on:** S01 (env, `docs/dicom_tags.md`, `docs/data_audit.md`, cache-drive choice) · **Unblocks:** S03 (MAE), S05 (Track A)

## Goal
One preprocessing module used **identically** by training, the cache build, and the Kaggle notebook, plus a full
memory-mapped cache of all 4,407 train studies.

## Read first
`02_vit_detailed_plan.md` §A (the checklist table and slot layout) · `docs/kaggle/data_and_dicom_from_notebooks.md` (cache recipes,
speeds) · `docs/kaggle/top_solutions_from_notebooks.md` §preprocessing.

## Build (`src/knee/data/`)
1. `dicom_io.py`: `read_series_headers(series_dir)` → a DataFrame (file, IPP, IOP, PixelSpacing, SliceThickness, TR, TE,
   TI, Rows/Cols, …); `read_pixels(file)` → float32 with the modality LUT (and MONOCHROME1 inversion if it ever appears).
   Robust to corrupt files (returns None and logs).
2. `geometry.py`: slice order by IPP·normal; **laterality detection** (the geometry rule from S01) and **canonicalization**
   (mirror to a right-knee view: in-plane LR for cor/ax, slice reversal for sag; the exact axis mapping verified in S01).
3. `series_select.py`: the contrast class from TR/TE/TI (PD/T1/T2/GRE/STIR; thresholds from S01's audit, documented);
   the slot assignment (sag-fluid, sag-other, cor-fluid, cor-other, axial) + the full series list for Track B.
4. `normalize.py`: per-series percentile window (HP: 1–99 or 2–98) → uint8.
5. `crop_resize.py`: centred fixed-mm crop (HP ≈ 130–140 mm) using PixelSpacing → `H`×`H` (HP; store at the **largest**
   resolution any track needs, e.g. 384, since downsampling at load time is cheap and re-caching isn't).
6. `build_cache.py`: multiprocessing over studies (workers = physical cores from S01), resumable (skips finished studies),
   writes:
   - `vols_<plane>.npy` or per-series shards, memmap-friendly. **Design so one study reads with one contiguous access.**
   - `series_index.parquet`: study, series, slot, plane, fluid, contrast_class, field_strength, laterality, n_slices,
     slice positions (mm), offset into the shard, plus a per-slice presence mask.
   - `cache_manifest.json`: preprocessing params + git sha + **cache version** (`v1`).
7. `dataset.py`: `StudyBagDataset` (Track B: variable-length window bag + metadata + masks) and `StudyWindowsDataset`
   (Track A: slots × windows); both do 2.5D window assembly (C=3) and train-time sampling vs eval-time dense windows.
   Spawn-safe (opens the memmap lazily in the worker).

## Measure (log in `docs/data_audit.md` §Cache)
- Decode + preprocess s/study (single process and N processes); total build time; cache size on disk.
- The DataLoader throughput (studies/s) for each dataset at the planned batch shapes: it must beat the GPU step time.

## Tests (`tests/test_data.py`)
- Slice order is monotonic in physical position; the laterality rule matches the tag where present (report accuracy).
- Plot 3 random studies (all slots, a few windows each) → `docs/figures/cache_check_*.png` for **Dom to eyeball**.
- Parity: a study processed through the notebook code path == the cache bytes.

## Done when
- [ ] Full cache built; the manifest is written; the size/time are logged.
- [ ] Both datasets load a batch fast enough; the figures are committed (small PNGs only).
- [ ] The HANDOFF/master plan are updated; `S03` notes any cache-format facts it needs.

## Review guide for Dom (write in the session log)
Learning notes: why a mm-based crop beats a pixel crop; how IPP·normal gives physical order; what 2.5D windows are;
why the cache is uint8 memmap (the RAM limit).
