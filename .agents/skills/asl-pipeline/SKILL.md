---
name: asl-pipeline
description: Safely operate and modify this repository's ASL landmark extraction, signer-disjoint splitting, cache generation, PyTorch training, checkpoint evaluation, and MediaPipe live-inference workflows. Use for requests involving data/raw_landmarks, Kaggle parquet extraction, split.json, feature dimensions, dataset.py, model.py, training, checkpoints, inference, webcam translation, cache migration, or ASL pipeline debugging. Do not use for the vendored youtube-dl project.
---

# Operate the ASL pipeline

## Establish state

1. Work from the repository root.
2. Read root `AGENTS.md` and inspect `git status --short` before edits.
3. Preserve existing user changes, `.claude/`, `.gstack/`, raw data, generated caches, logs, and checkpoints unless the request explicitly targets them.
4. Read the current module docstrings and constants instead of relying on `bug.md`, `kag.md`, or `over.md`; those are historical/design notes.

## Trace the relevant path

- Extraction: `extract_kaggle_landmarks.py` or `process_video.py` -> per-video JSON.
- Discovery/splits: `ingest.py` + `split_dataset.py` -> `(path, label)` entries selected by `data/split.json`.
- Features: `dataset.py` -> fused `COORD_DIM` cache -> `finalize_sequence` -> `TOTAL_FEATURE_DIM` tensors.
- Model: `model.py` -> one class-logit vector per 60-frame isolated-sign sequence.
- Training/evaluation: `train.py` -> `models/checkpoint.pt` -> `infer.py`.
- Live path: `mac.py` -> gesture buffer -> the same feature finalization/model -> `speak.py`.

Keep training and live preprocessing equivalent. When changing a record shape, feature slot, sequence length, vocabulary, or model layer, trace every downstream consumer before editing.

## Choose the safe validation tier

Run focused synthetic checks first:

```powershell
python src/<changed-module>.py --self-test
```

Modules without that flag run their self-check directly: `ingest.py`, `dataset.py`, and `model.py`. Run dependent checks when a shared contract changes:

- Ingestion or filenames: `ingest.py`, then `dataset.py`.
- Feature fusion/layout/augmentation: `dataset.py`, `model.py`, `train.py --self-test`, `infer.py --self-test`, and `mac.py --self-test`.
- Model/training: `model.py`, `train.py --self-test`, and `infer.py --self-test`.
- MediaPipe extraction/live wiring: `process_video.py --self-test` and `mac.py --self-test`.
- Cache behavior: `build_landmark_cache.py --self-test` plus `dataset.py`.

Do not substitute a full corpus run for a synthetic check.

## Guard expensive and compatibility-sensitive actions

Obtain explicit approval before:

- extracting the full Kaggle corpus;
- regenerating `data/split.json`;
- deleting or rebuilding `.npy` caches;
- starting a real training run;
- overwriting `models/checkpoint.pt`;
- opening a camera or performing audible speech;
- deleting, moving, or restoring dataset files.

Before an approved migration, report the exact targets, expected generated outputs, compatibility impact, and recovery path. Cache creation is resumable but skips existing files. Never remove caches until the resolved target directory has been checked and the user has approved the migration.

## Preserve data and ML invariants

- Keep train/validation/test signer-disjoint; do not evaluate tuning decisions on the held-out test split.
- Keep vocabulary order stable unless intentionally retraining a newly sized classifier.
- Import layout constants from `dataset.py`; never hardcode a duplicate width.
- Validate both axes of cached arrays: sequence length and `COORD_DIM`.
- Derive velocity and presence after loading coordinates so caches cannot silently stale for derived features.
- Apply augmentation only to training data and only when label-preserving after normalization.
- Keep missing detections distinguishable from valid zero coordinates through presence flags.
- Keep personal-recording and Kaggle filename parsing compatible.

## Report results

State which checks ran and which were skipped. Distinguish synthetic self-check success from real-data quality. For experiments, record command, split, checkpoint, seed/configuration, train/validation metrics, and whether artifacts were overwritten. Never claim accuracy improvement without comparable measured runs.
