# S02 — Preprocessing library + cache v1

**Track:** D (shared) · **Machine:** Windows · **GPU:** no (CPU decode; the full build runs overnight)
**Depends on:** S01 (env, `docs/dicom_tags.md`, `docs/data_audit.md`, cache-drive choice) · **Unblocks:** S03 (MAE), S05 (Track A)

## Status after part 1 (2026-09-24, on the Mac)
Part 1 wrote and tested all of this session's code on synthetic DICOMs and the real test stub; part 2 runs it on the
full train set. Session log: `docs/sessions_log/S02_2026-09-24.md`. Format/rules: `docs/cache.md`.

| Brief step | Code | Status |
|---|---|---|
| 1 dicom_io | `src/knee/data/dicom_io.py` | done (deferred pixel reads; robust) |
| 2 geometry / laterality / canonicalization | `geometry.py` (canonical frame, `resolve_study_laterality`) | done; **dead zone = placeholder 20 mm** until S01 part 2 |
| 3 series_select | `series_select.py` (+ contrast class from `series_meta.py`) | done; rank tables are placeholders for Dom's review |
| 4 normalize, 5 crop_resize | `normalize.py`, `crop_resize.py` | done (1–99 %, 130 mm, 384 px; decision 2026-09-24) |
| (new) one entry point | `preprocess.py::preprocess_study` | done: used by the cache **and** the notebook |
| 6 build_cache | `build_cache.py` | done; stub cache built; **full build on Windows** |
| 7 dataset | `dataset.py` | done; loader bench on the stub |
| Measure | `inspect_cache.py bench`, `docs/cache.md` §4 | stub numbers in §4a; **§4b on Windows** |
| Tests | `tests/test_data.py` (32) | pass; parity test included |

**Part 2 checklist** (after S01 part 2):
1. `git pull`; `pip install -e . --no-deps` if needed; `configs/paths.yaml` `cache_root` on an internal drive with ≥ ~130 GB free.
2. `powershell -ExecutionPolicy Bypass -File scripts\run_s02.ps1 -DeadZoneMm <from S01 part 2> -SmokeOnly`, then check
   the smoke numbers and the projected size.
3. Run it again without `-SmokeOnly` (the full build is resumable; it refuses to mix params). Expect ≈1 h, disk-bound (UNVERIFIED).
4. Fill `docs/cache.md` §4b; look at `docs/figures/cache_check_*.png` with Dom (every knee must read as a RIGHT knee).
5. If S01 part 2 changes the contrast thresholds or the CSV-flag meaning, update `series_meta.py`/`series_select.py`
   **before** the full build (slots and contrast classes are written into the index at build time).

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
