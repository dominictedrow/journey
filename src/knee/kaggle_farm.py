"""Rebuild the train cache on Kaggle from its 7 datasets (AB.md Part 1): `knee-cache-v1-index` (manifest, series/study
index, labels, folds, configs) + `knee-cache-v1-s0..s5` (flat shard_XXXXX.u8). CacheReader needs one root with
`shards/`, so this builds a symlink farm (no copy). Staged by scripts/stage_kaggle_cache.py."""
from __future__ import annotations

import glob
from pathlib import Path

# /kaggle/input/datasets/<user>/<slug>/ or /kaggle/input/<slug>/ (docs/kaggle_env.md); shallow globs only
DS = ["/kaggle/input/datasets/*/{s}", "/kaggle/input/datasets/*/{s}/*", "/kaggle/input/{s}", "/kaggle/input/{s}/*"]
INDEX_FILES = ("cache_manifest.json", "series_index.parquet", "study_index.parquet")


def find(slug: str, marker: str, roots: list[str] = DS) -> Path:
    for pat in roots:
        for d in sorted(glob.glob(pat.format(s=slug))):
            if glob.glob(f"{d}/{marker}"):
                return Path(d)
    raise FileNotFoundError(f"{marker} not found for dataset {slug}")


def build_farm(dst: Path, parts: int = 6, roots: list[str] = DS) -> tuple[Path, Path]:
    """→ (cache root for CacheReader, index dir with labels/folds/configs)."""
    idx = find("knee-cache-v1-index", "cache_manifest.json", roots)
    (dst / "shards").mkdir(parents=True, exist_ok=True)
    for f in INDEX_FILES:
        if not (dst / f).exists():
            (dst / f).symlink_to(idx / f)
    for i in range(parts):
        for s in sorted(find(f"knee-cache-v1-s{i}", "shard_*.u8", roots).glob("shard_*.u8")):
            if not (dst / "shards" / s.name).exists():
                (dst / "shards" / s.name).symlink_to(s)
    return dst, idx
