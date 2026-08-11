# Lazy-load the landmark corpus (fix OOM risk in train.py / infer.py)

## Context

We just extended the Kaggle ingestion pipeline (`extract_kaggle_landmarks.py`, `dataset.py`) to pull face (468 points) and pose (33 points) landmarks in addition to the two hands (21 points each) — `dataset.TOTAL_FEATURE_DIM` went from 126 to 1629, and `model.py`'s `INPUT_SIZE`/`HIDDEN_SIZE` (now 384) were updated to match. That re-extraction already ran against all 94,477 Kaggle sequences and `data/raw_landmarks/` now totals **~137GB** (was 3.9GB), with `split_dataset.py` re-run to regenerate `data/split.json` over the new files. All of that is done and self-checked.

The problem: `src/ingest.py`'s `load_records(data_dir, split)` globs every `*_landmarks.json` file matching a split and eagerly reads + `json.loads()`s all of them into one big in-memory list, before returning. `src/dataset.py`'s `SignSequenceDataset.__init__` then eagerly fuses+resamples that entire list into a fully-materialized `self.sequences` array, all at construction time. This was fine when the corpus was a few GB, but the train split alone is now **~79GB of raw JSON text** — trying to load that into one Python process (twice — once in `load_records`, again in `SignSequenceDataset.__init__`) will not fit in this machine's 16GB RAM + 16GB pagefile. A live test attempting this got stuck this session and had to be killed.

This is the same failure mode (Windows `ERROR_COMMIT_LIMIT`) already hit and fixed once before in this project, when `make_dataloader`'s `num_workers=8` + `persistent_workers=True` caused each of 8 spawned Windows worker processes to hold its own duplicate of an eagerly-built sequence array. That fix doesn't help here because the problem now starts one step earlier, before the DataLoader/workers are even involved — in the plain single-process `load_records` call itself. Both problems share the same root cause: everything is loaded eagerly into process memory instead of being read from disk on demand, per training example.

Goal: make the real training/inference data path lazy (read + fuse one video's landmarks from disk only when that item is actually requested), so memory use no longer scales with corpus size, while leaving everything else (self-checks, live camera inference, training loop mechanics) untouched.

**Confirmed via code tracing (not just docstrings):** every `*_landmarks.json` file already corresponds to exactly one video — `process_video.py::process_and_save` writes one file per `.mp4`, `extract_kaggle_landmarks.py::process_row` writes one file per Kaggle `(participant_id, sequence_id)` row. So a lazy per-file loader never needs to join records across files.

**Call-graph audit (verified):** only two real production call sites touch this eager path — `train.py::train()` and `infer.py::run_inference()`. Every `_self_check()` in `ingest.py`/`dataset.py`/`train.py`/`infer.py` uses a handful of synthetic in-memory records/temp files and stays on the existing eager API unchanged. `mac.py` (live webcam) calls `fuse_landmarks`/`resample_sequence` directly per-frame already and never touches `load_records`/`make_dataloader` — unaffected.

## Design

Add a lazy sibling to the existing eager path rather than converting it in place — the two `__getitem__` bodies and `__init__`s genuinely differ (index into a pre-built array vs. read+group+fuse+resample per call), and a dual-mode class would need branching in both, which is worse for a beginner reading this code than two short, single-purpose classes. Share the one piece of logic that actually matters (the per-video fuse+resample math) via a small extracted helper, not by merging the classes.

### `src/ingest.py`

- **`load_video(path: Path, label: str) -> list[dict]`** (new, public): the current loop body of `load_records` — read one file, `json.loads`, pass pre-tagged Kaggle detections through unchanged, synthesize `{"video": label, "label": label, **detection}` for untagged personal-recording detections. Move the "two record shapes" docstring explanation here, since this is where that logic actually lives now.

- **`load_records`** (existing, keep): shrinks to a thin loop — glob, split-filter, compute `label = path.stem.removesuffix("_landmarks").split("__", 1)[0]`, `records.extend(load_video(path, label))`. Behavior-preserving (the `.split("__", 1)[0]` addition is a no-op for every real file: Kaggle detections never use this filename-derived label since they already carry their own `"label"` field, and personal-recording filenames never contain `"__"`). Docstring gets one line pointing at `list_video_files` + `dataset.LazySignVideoDataset` as the lazy alternative for full-corpus use.

- **`list_video_files(data_dir: Path = DATA_DIR, split: str | None = None) -> list[tuple[Path, str]]`** (new): same glob + split-filter loop as `load_records`, but appends `(path, label)` pairs instead of reading file contents — **no `path.read_text()` or `json.loads()` anywhere in this function**, that's the whole point. Same label formula as above.

- Self-check additions (extend `_self_check`, don't restructure): `list_video_files` on a personal-recording-style filename with deliberately invalid/garbage file content still returns the correct `(path, "hello")` pair without raising (proves no per-file I/O happens); `list_video_files` on a Kaggle-style filename (`"book__1-1_landmarks.json"`) returns label `"book"`, not the full stem; split-filtering assertions for `list_video_files` mirroring the existing `load_records` ones (reuse the existing `cat__1`/`cat__2`/`dog__1` + `split.json` fixture already in `_self_check`).

### `src/dataset.py`

- **`_video_to_sequence(video_records: list[dict], sequence_length: int) -> np.ndarray | None`** (new, extracted verbatim from `_build_sequences`'s per-video loop body): group by `"frame"`, `fuse_landmarks` each frame in order, `resample_sequence`, return `None` if there were no frames. Place after `resample_sequence`, before `_build_sequences`.

- **`_build_sequences`** (existing, keep): unchanged video-grouping loop, now delegates per-video work to `_video_to_sequence`. Pure refactor, behavior identical.

- **`LazySignVideoDataset(Dataset)`** (new):
  - `__init__(self, files: list[tuple[Path, str]], vocab: list[str], sequence_length: int = SEQUENCE_LENGTH, augment: bool = False)` — stores the four args, no I/O, no `self.sequences`.
  - `__len__` → `len(self.files)`.
  - `__getitem__(self, idx)`: `path, label = self.files[idx]`; `load_video(path, label)` → `_video_to_sequence(...)`; if that comes back `None`, raise a loud `ValueError` (a fixed-length `Dataset` can't silently drop an index the way the eager list-building path can silently drop an empty video — and both writers guarantee this never happens for a file that actually got written, so failing loud is correct, not a gap to paper over); apply `augment_sequence` if `self.augment`; return `(torch.from_numpy(coords), torch.from_numpy(one_hot(label, self.vocab)))` — same shape contract as `SignSequenceDataset.__getitem__`.
  - New import: `from ingest import load_video` (no cycle — `ingest.py` doesn't import `dataset.py`).

- **`make_lazy_dataloader(files, vocab, sequence_length=SEQUENCE_LENGTH, batch_size=16, shuffle=True, num_workers=8, augment=False) -> DataLoader`** (new): mirrors `make_dataloader` exactly (`persistent_workers=num_workers > 0`, `pin_memory=True`), builds a `LazySignVideoDataset`. Comment noting laziness is what fixes the per-worker duplication problem too — each of the 8 Windows `spawn`-ed workers now pickles a small `(Path, str)` list instead of a multi-GB pre-fused array.

- Self-check additions (extend `_self_check`; local `import tempfile` + `from ingest import list_video_files, load_video`, same style `ingest.py`'s own self-check already uses):
  - Write the existing `book_c` fixture (already includes the handless middle frame) to a temp file named the real Kaggle way (`"book__1-1_landmarks.json"`), and a second untagged personal-recording-style file (`"hello_landmarks.json"`). For each, build via `list_video_files → LazySignVideoDataset` and, separately, `load_records → SignSequenceDataset`, and assert the two paths produce equal `(x, y)` tensors — proves lazy matches eager, and since `book_c`'s zero-fill frame is already the proven eager test case, matching against it directly carries that proof over. Add one explicit `torch.allclose(lazy_x[1], torch.zeros(TOTAL_FEATURE_DIM))` assertion too, not just the transitive one.
  - `assert not hasattr(lazy_dataset, "sequences")` — fails immediately if eager materialization ever creeps back in.
  - One `make_lazy_dataloader(files, vocab, sequence_length=3, batch_size=2, shuffle=False, num_workers=0)` smoke test, mirroring the existing `make_dataloader` one.

### `src/train.py`

Only the imports and the two loader lines in `train()` change:

```python
from dataset import SEQUENCE_LENGTH, load_vocab, make_lazy_dataloader
from ingest import list_video_files
...
train_loader = make_lazy_dataloader(list_video_files(data_dir, split="train"), vocab, SEQUENCE_LENGTH, batch_size,
                                     augment=True, num_workers=num_workers)
val_loader = make_lazy_dataloader(list_video_files(data_dir, split="val"), vocab, SEQUENCE_LENGTH, batch_size,
                                   shuffle=False, num_workers=min(num_workers, VAL_NUM_WORKERS))
```

Everything else (optimizer, scheduler, `run_epoch`, checkpointing, early stopping, logging) is untouched — it only ever consumed `train_loader`/`val_loader` as generic `(x, y)` iterables. `_self_check` needs no change: it calls `make_dataloader`/`run_epoch` directly and never calls `train()`, so it never exercised the eager path in the first place.

### `src/infer.py`

Same swap in `run_inference()`:

```python
from dataset import load_vocab, make_lazy_dataloader
from ingest import list_video_files
...
loader = make_lazy_dataloader(list_video_files(test_dir), vocab, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False)
```

`_self_check` needs no code change, but it calls `run_inference()` directly against real temp files, so after this swap it becomes a free end-to-end exercise of `list_video_files` + `make_lazy_dataloader` on top of the dedicated `dataset.py`/`ingest.py` unit tests.

## Known limitation (deliberate, not fixed now)

Lazy loading trades memory for repeated work: eager loading parses the whole corpus once, ever; lazy loading re-reads + re-`json.loads()`s every file **every epoch**, and with `shuffle=True` those reads hit files in random order across 8 workers instead of sequentially — real overhead on Windows, more so if the data directory is ever on a network/cloud-sync drive. This might end up GPU-starving training if disk I/O can't keep up. Not fixing it speculatively now — the natural next step if it becomes a real bottleneck would be a one-time offline conversion of each `_landmarks.json` into a faster-to-parse per-video cache (e.g. `.npz` of the already-fused array), but that's a separate, bigger change to make only once it's actually measured as a problem.

## Verification

1. Run each touched file's self-check directly: `python src/ingest.py`, `python src/dataset.py`, `python src/train.py --self-test`, `python src/infer.py --self-test`.
2. Real-scale smoke test (bounded, not the full corpus): call `list_video_files(data_dir, split="val")` (smaller split) and build a `LazySignVideoDataset` + `make_lazy_dataloader`, pull a handful of batches via `next(iter(loader))` a few times, confirm shapes are `(batch, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM)` and memory stays flat (spot-check via Task Manager / `Get-Process python` working set) rather than climbing — without iterating the entire split.
3. Only after that: actually launch `train.py` for real (background, monitored) — ask before doing this, same as prior runs in this session, since it's a long-running job.
