"""One-time offline conversion of every *_landmarks.json into a per-video
.npy cache of its already-fused+resampled (SEQUENCE_LENGTH, COORD_DIM) array
-- dataset.py's blocks A+B; velocity and presence are derived at load time by
dataset.finalize_sequence, so changing those needs no rebuild here (see
ingest.cache_path_for for the naming), so dataset.LazySignVideoDataset
stops re-parsing multi-MB JSON text every training epoch -- np.load on a flat
binary array is far cheaper than json.loads + fuse_landmarks/resample_sequence
on the same video's text. Read-only with respect to the source *_landmarks.json
files; only ever adds a sibling *_landmarks.npy next to each one.

Resumable like extract_kaggle_landmarks.py's process_batch: a file that
already has a cache is skipped, so an interrupted run just picks back up.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from dataset import COORD_DIM, SEQUENCE_LENGTH, _video_to_sequence
from ingest import RAW_LANDMARKS_DIR, cache_path_for, list_video_files, load_video


def build_one(path: Path, label: str, sequence_length: int = SEQUENCE_LENGTH) -> str:
    """One *_landmarks.json -> its sibling .npy cache. Returns "ok"/"empty"/"skipped"."""
    cache_path = cache_path_for(path)
    if cache_path.exists():
        return "skipped"
    coords = _video_to_sequence(load_video(path, label), sequence_length)
    if coords is None:
        return "empty"
    np.save(cache_path, coords)
    return "ok"


def build_cache(data_dir: Path = RAW_LANDMARKS_DIR, workers: int = os.cpu_count(),
                 sequence_length: int = SEQUENCE_LENGTH) -> None:
    files = list_video_files(data_dir)
    print(f"{len(files)} landmark files in {data_dir}, {workers} workers")

    t0 = time.time()
    ok, empty, skipped, failed = 0, 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_one, path, label, sequence_length): path for path, label in files}
        for i, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                result = future.result()
                ok += result == "ok"
                empty += result == "empty"
                skipped += result == "skipped"
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(files)}] FAILED {path.name}: {e}")
            if i % 2000 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] {ok} cached, {skipped} already cached, {empty} empty, {failed} failed")

    elapsed = time.time() - t0
    print(f"\n{ok} cached, {skipped} already cached, {empty} empty (no frames), {failed} failed, "
          f"{elapsed:.1f}s for {len(files)} files")


def _self_check():
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        book_records = [{"video": "book__1-1", "label": "book", "frame": f, "handedness": "Right",
                          "landmarks": [{"x": 0.1 * f, "y": 0.2, "z": 0.0}] * 21} for f in range(3)]
        (tmp_path / "book__1-1_landmarks.json").write_text(json.dumps(book_records))
        (tmp_path / "empty_landmarks.json").write_text(json.dumps([]))

        build_cache(tmp_path, workers=1, sequence_length=3)

        cache_path = tmp_path / "book__1-1_landmarks.npy"
        assert cache_path.exists()
        cached = np.load(cache_path)
        # Imported, not restated as a literal: dataset.py owns the feature
        # layout, and a hardcoded width here would keep passing after the
        # layout changes -- exactly the silent-staleness failure the cache
        # guard in LazySignVideoDataset exists to prevent.
        assert cached.shape == (3, COORD_DIM)
        expected = _video_to_sequence(load_video(tmp_path / "book__1-1_landmarks.json", "book"), sequence_length=3)
        assert np.allclose(cached, expected)

        assert not (tmp_path / "empty_landmarks.npy").exists()  # no frames -> no cache file written

        # resumability: a second run must skip the already-cached file, not overwrite it
        cache_path.write_bytes(b"corrupt")  # would fail to np.load if re-read
        build_cache(tmp_path, workers=1, sequence_length=3)
        assert cache_path.read_bytes() == b"corrupt"  # untouched -- build_one saw the cache and skipped

    print("self-check passed: per-video .npy caches get written with the same coords "
          "fuse_landmarks/resample_sequence would produce, frameless videos write no cache, "
          "and a second run skips files already cached instead of overwriting them")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=RAW_LANDMARKS_DIR,
                         help="folder of *_landmarks.json files to cache (default: data/raw_landmarks)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--self-test", action="store_true", help="run wiring smoke test instead of building the real cache")
    args = parser.parse_args()

    if args.self_test:
        _self_check()
    else:
        build_cache(args.data_dir, args.workers)
