# ASL Translator Repository Guide

## Scope and safety

- The application is the Python code in `src/`; run its scripts from the repository root so sibling imports and repository-relative paths resolve correctly.
- Treat existing working-tree changes as user work. Inspect `git status` before editing; never restore deleted datasets, reset the checkpoint, or rewrite unrelated files.
- Treat `.claude/` and `.gstack/` as user-owned infrastructure. Do not modify them. Do not inspect `.claude/worktrees/` as if it were the active source tree.
- `youtube-dl/` is a vendored third-party project used by legacy dataset downloaders, not part of the ASL model. Avoid modifying or testing it unless explicitly requested.
- Large data/model operations are consequential: ask before full extraction, cache rebuilds, training, checkpoint replacement, camera use, or bulk deletion. Never commit restricted/raw clips.

## Architecture

The project recognizes one isolated ASL sign per video:

1. `src/extract_kaggle_landmarks.py` reshapes Kaggle parquet landmarks, while `src/process_video.py` extracts MediaPipe Holistic landmarks from ordinary videos.
2. `src/ingest.py` discovers `*_landmarks.json` clips and applies `data/split.json`.
3. `src/dataset.py` fuses and normalizes landmarks, resamples to 60 frames, derives velocity/presence features, augments training data, and lazily serves tensors. It owns all feature-layout constants.
4. `src/model.py` defines the stacked bidirectional LSTM with residual LayerNorm blocks and attention pooling.
5. `src/train.py` trains and writes `models/checkpoint.pt`; `src/infer.py` evaluates a checkpoint; `src/mac.py` buffers a webcam gesture and classifies it, then `src/speak.py` uses macOS `say`.

## Important paths

- `src/vocab.txt`: ordered 250-class vocabulary; order is part of the checkpoint contract.
- `data/raw_landmarks/`: generated per-clip JSON and `.npy` cache pairs; very large and ignored by Git.
- `data/split.json`: generated signer-disjoint train/validation/test assignment.
- `data/kaggle_raw/`: downloaded Kaggle source data; very large.
- `models/holistic_landmarker.task`: MediaPipe model required by video/live paths.
- `models/checkpoint.pt`: trained weights tied to the current vocabulary and feature/model shapes.
- `logs/`: generated training logs. `reference/` contains upstream examples, not application entry points.
- `bug.md`, `kag.md`, and `over.md`: investigation/design notes; verify claims against current code before acting.

## Development and verification

There is no package manifest or centralized test runner. Dependencies visible in imports include PyTorch, NumPy, OpenCV, MediaPipe, pandas, and a parquet engine. Prefer the script-local synthetic self-checks:

```powershell
python src/ingest.py
python src/dataset.py
python src/model.py
python src/train.py --self-test
python src/infer.py --self-test
python src/split_dataset.py --self-test
python src/extract_kaggle_landmarks.py --self-test
python src/process_video.py --self-test
python src/mac.py --self-test
python src/build_landmark_cache.py --self-test
python src/speak.py
```

Run the smallest checks relevant to a change. MediaPipe checks require `models/holistic_landmarker.task`; `speak.py` skips real playback off macOS. Do not run the vendored `youtube-dl` suite for application changes.

## Pipeline commands

```powershell
python src/extract_kaggle_landmarks.py
python src/split_dataset.py
python src/build_landmark_cache.py
python src/train.py --epochs 50
python src/infer.py --test-dir data/raw_landmarks --checkpoint models/checkpoint.pt
python src/process_video.py input.mp4
python src/mac.py --checkpoint models/checkpoint.pt
```

The first four can be expensive or destructive to derived artifacts; do not run them merely as verification. Use `$asl-pipeline` for detailed workflow guardrails.

## Conventions and invariants

- Keep one video/one label/one sequence semantics and signer-disjoint splits.
- Keep feature dimensions centralized in `dataset.py`; import constants instead of repeating numeric widths.
- Cached `.npy` files contain only `COORD_DIM` coordinates; `finalize_sequence` derives velocity and presence to produce `TOTAL_FEATURE_DIM` tensors.
- Preserve missing-part zero fill plus explicit presence masks, right/left slot ordering, mirror-pair logic, and train-only augmentation.
- Preserve filename contracts: personal clips use `<label>_landmarks.json`; Kaggle clips use `<sign>__<participant>-<sequence>_landmarks.json`.
- Treat cache shape, vocabulary order, feature layout, model architecture, and checkpoint weights as one compatibility boundary. A change to one may require cache regeneration or retraining.
- Follow the existing style: module docstrings explain design, `pathlib.Path`, type hints, snake_case, constants in uppercase, concise comments for non-obvious ML/data decisions, and colocated `_self_check()` assertions.

## Common pitfalls

- Cache generation skips existing `.npy` files; a layout change requires an intentional cache migration, not a blind rebuild.
- `train.py` and split-aware loading expect `data/split.json`; files absent from it are skipped when a split is requested.
- Windows DataLoader workers use spawn and can duplicate startup/memory costs; keep validation workers conservative and use `num_workers=0` in small tests.
- Full training writes a timestamped log and may overwrite the tracked checkpoint. Inference fails when checkpoint, vocabulary, input width, or architecture disagree.
- `mac.py` is buffered isolated-sign inference, not causal frame-by-frame translation, and speech output is macOS-specific.
- Dataset licenses and `.gitignore` intentionally prevent redistributing raw videos and large regenerated artifacts.
