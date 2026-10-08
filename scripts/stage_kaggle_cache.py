"""Stage the train cache as 6 flat shard datasets + 1 index dataset for Kaggle (AB.md Part 1; decision 2026-10-07).

    python scripts/stage_kaggle_cache.py [--cache C:/knee_cache/cache_v1_train] [--out C:/knee_cache/kaggle_stage]

- `<out>/knee-cache-v1-s<i>/`: shard_XXXXX.u8 files, hard-linked (zero copy, same NTFS volume), flat in the root
  (Kaggle allows ≤ 50 top-level files). The per-shard parquets are skipped; CacheReader only reads the .u8 files.
- `<out>/knee-cache-v1-index/`: the cache manifest + series/study index, labels_v0, folds_v0, the train configs. Flat.
- Each dir gets a dataset-metadata.json (private by default). Then it prints the upload commands; **run them only
  with Dom's OK** (MIRA data: the datasets stay private, never public).
A Kaggle kernel rebuilds `<cache>/shards/*.u8` as a symlink farm over the 7 mounts (notebooks/kaggle_cachecheck).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from knee.paths import load_paths

REPO = Path(__file__).resolve().parents[1]


def stage(cache: Path, labels: Path, out: Path, parts: int, user: str) -> list[Path]:
    man = json.loads((cache / "cache_manifest.json").read_text(encoding="utf-8"))
    shards = sorted((cache / "shards").glob("shard_*.u8"))
    if len(shards) != man.get("n_shards", len(shards)):
        raise SystemExit(f"{len(shards)} .u8 files but the manifest says {man['n_shards']}")
    per = -(-len(shards) // parts)
    if per > 49:  # + dataset-metadata.json ≤ 50 top-level files
        raise SystemExit(f"{per} shards per part exceeds Kaggle's 50-file root limit; raise --parts")
    dirs = []

    def meta(d: Path, slug: str, title: str) -> None:
        d.mkdir(parents=True, exist_ok=True)
        (d / "dataset-metadata.json").write_text(json.dumps(
            {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}, indent=1), encoding="utf-8")
        dirs.append(d)

    for i in range(parts):
        slug = f"knee-cache-v1-s{i}"
        d = out / slug
        meta(d, slug, f"knee cache v1 shards {i}")
        for s in shards[i * per:(i + 1) * per]:
            dst = d / s.name
            if not dst.exists():
                os.link(s, dst)
    d = out / "knee-cache-v1-index"
    meta(d, "knee-cache-v1-index", "knee cache v1 index")
    for f in [cache / "cache_manifest.json", cache / "series_index.parquet", cache / "study_index.parquet",
              labels / "labels_v0.parquet", labels / "folds_v0.csv",
              *[p for p in (REPO / "configs").glob("*.yaml") if p.name not in ("paths.yaml", "paths.example.yaml")]]:
        shutil.copy2(f, d / f.name)
    return dirs


def main() -> None:
    p = load_paths()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=p.cache_root / "cache_v1_train")
    ap.add_argument("--labels", type=Path, default=p.cache_root / "labels")
    ap.add_argument("--out", type=Path, default=p.cache_root / "kaggle_stage")
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--user", default="dominictedrow")
    a = ap.parse_args()
    dirs = stage(a.cache, a.labels, a.out, a.parts, a.user)
    for d in dirs:
        n = sum(1 for f in d.iterdir() if f.name != "dataset-metadata.json")
        gb = sum(f.stat().st_size for f in d.iterdir()) / 1e9
        print(f"{d.name}: {n} files, {gb:.1f} GB")
    print("\nUpload (Dom's OK first; one per terminal runs them in parallel):")
    for d in dirs:
        print(f'kaggle datasets create -p "{d}" -r skip')


if __name__ == "__main__":
    main()
