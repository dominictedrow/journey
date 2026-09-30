# Kaggle notebook environment (measured)

Source: private kernel `dominictedrow/knee-probe` (`notebooks/kaggle_probe/probe.py`), **CPU session, internet on**,
run 2026-09-29, 62 s total. Raw output: `probe.json` (kept out of the repo; re-run the kernel to regenerate).
Nothing from the dataset was downloaded; only timings, counts and the decoder wheels came back.

## 1. Machine (CPU notebook)

| | measured |
|---|---|
| CPU | 4 vCPUs (cgroup quota 4.0), Intel Xeon @ 2.20 GHz |
| RAM | 32 GB (cgroup limit 30 GiB) |
| `/kaggle/working` (persisted output) | 20.9 GB |
| `/kaggle/tmp` | does not exist |
| `/tmp` (overlay on the host disk) | 1.1 TB free of 8 TB, **shared host disk; session-only; usable quota UNVERIFIED** |
| `/dev/shm` | 14 GB |
| GPU | none (CPU session). GPU sessions: T4 ×2 / P100 and 4 CPU / 29 GB RAM per earlier research (UNVERIFIED here) |

## 2. Image

Python **3.12.13**, torch **2.10.0+cpu** (CPU image), torchvision 0.25.0, timm 1.0.26, numpy **2.0.2**, pandas 2.3.3,
pydicom 3.0.2, scikit-learn 1.6.1, pyarrow 24.0.0, PyYAML 6.0.3, OpenCV 4.13.0, Pillow 11.3.0, psutil 5.9.5.
**Missing:** pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg, gdcm.

**Wheels built on the image** (cp312, manylinux): pylibjpeg 2.1.0, pylibjpeg-libjpeg 2.4.0, pylibjpeg-openjpeg 2.6.0,
python-gdcm 3.2.6. `pip download` also fetched **numpy 2.5.3** as a dependency. Installing that wheel would replace the
image's numpy under torch, so `notebooks/infer.py` installs with `--no-deps` and `build_kaggle_bundle.py` skips `numpy-*`.

## 3. Competition data mount

- Path: **`/kaggle/input/competitions/rsna-knee-abnormality-detection/`** (the `/kaggle/input/<slug>` layout isn't used;
  `infer.py` tries both). It's an NFS mount.
- Contents: `sample_submission.csv, test.csv, test_series/, test_series.csv, train.csv, train_series/, train_series.csv`.
- **4,407** train study folders (matches `train.csv`); 3 test study folders in the visible stub.
- Files per study: mean **193.9** over 200 random studies, which puts the corpus at ≈ **855k** DICOM files (estimate; the
  Data page says 819,640 files in total, including CSVs and test).

## 4. I/O and decode speed (one process, 30 random series = 895 files, 570 MB)

| | measured |
|---|---|
| cold read from the NFS mount | **63 MB/s**, ≈ 99 files/s |
| pydicom `dcmread` + `pixel_array` (warm page cache) | **449 files/s** |
| transfer syntax | 895/895 Explicit VR Little Endian (no decoder needed for these) |
| slice shapes (top) | 512², 256², 384², 528², 1024² |
| listing 4,407 study dirs | 0.08 s; counting files in 200 studies: 23 s |

**So I/O is the bottleneck, not decoding.** Parallel readers do scale on the NFS mount (`knee-audit` decode bench, 300
fresh files per row: read + decode + rescale):

| workers | slices/s | MB/s |
|---|---|---|
| 1 | 62 | 43 |
| 4 | 173 | 122 |
| 8 | 259 | 176 |
| 16 | 325 | 213 |

At 16 workers, a full pass over the ≈ 575 GB of train files takes ≈ 45 min (819k slices / 325 per s ≈ 42 min).
The full header scan (16 workers, headers only) took 21 min.

## 5. What this means for the plan

1. **S06 offline wheels: solved.** The private dataset `dominictedrow/knee-wheels` (4 wheels, no numpy) installed cleanly
   offline in `knee-audit` (`pip install --no-index --no-deps`), and all decoders were bit-exact on every syntax.
2. **S01 part 2 ran on Kaggle** (2026-09-30, `notebooks/kaggle_audit`, 31 min, CPU, no GPU quota): the header scan, series summary and reports read the mount
   directly, with `src/knee` from the private code-only dataset `dominictedrow/knee-src`. The decode benchmark on Kaggle doesn't replace the
   Windows-box throughput numbers S02 needs.
3. **The 120 GB cache can't be a notebook output** (20.9 GB cap). A session could build a cache in `/tmp` (1.1 TB free,
   quota UNVERIFIED) and train in the same session, but it's lost afterwards. A persistent Kaggle-side cache would need
   ≤ 20 GB per kernel output, e.g. a smaller resolution or a series subset, saved as datasets.
