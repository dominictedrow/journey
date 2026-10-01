"""Derive a lower-resolution cache from a built one (docs/plan/03_performance_plan.md, A5).

    python -m knee.data.derive_cache --src <cache_root>/cache_v1_train --px 256 [--dst <dir>] [--workers 8]

Every slice is resized with cv2 INTER_AREA, **the same call `dataset.gather_windows` makes when a dataset asks for
out_px at load time**, so a Track B dataset reading the 256-px cache gets bit-identical windows to one that reads the
384-px cache with out_px=256, just without the per-window resize and with 2.25× fewer bytes to read (~54 GB vs
~120 GB). For MAE it is a change ([val]): random crops are then cut from the 256-px slice and enlarged, instead of cut
from the 384-px slice and shrunk, so they carry less detail.

Same layout as the source (docs/cache.md §2): shards/shard_XXXXX.u8 with the same slice offsets, the same index
parquets, and the manifest with params.out_px replaced and a `derived_from` record. Resumable: a shard is written to
a .tmp file and renamed when complete, and existing shards are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

CHUNK = 256  # slices resized per write (bounds memory: 256 × 384² ≈ 38 MB in)


def derive_shard(src: Path, dst: Path, src_px: int, px: int) -> int:
    """One shard → its resized copy. Returns the number of slices."""
    if dst.exists():
        return dst.stat().st_size // (px * px)
    n = src.stat().st_size // (src_px * src_px)
    tmp = dst.with_suffix(".u8.tmp")
    if n == 0:  # a shard whose studies all failed is 0 bytes; np.memmap can't map an empty file
        tmp.write_bytes(b"")
        os.replace(tmp, dst)
        return 0
    mm = np.memmap(src, dtype=np.uint8, mode="r", shape=(n, src_px, src_px))
    with open(tmp, "wb") as fh:
        for i in range(0, n, CHUNK):
            block = np.asarray(mm[i: i + CHUNK])
            out = np.stack([cv2.resize(s, (px, px), interpolation=cv2.INTER_AREA) for s in block])
            fh.write(np.ascontiguousarray(out, dtype=np.uint8).tobytes())
    os.replace(tmp, dst)
    return n


def derive(src_dir: Path, dst_dir: Path, px: int, workers: int = 4) -> dict:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    man = json.loads((src_dir / "cache_manifest.json").read_text())
    src_px = int(man["params"]["out_px"])
    if px >= src_px:
        raise SystemExit(f"--px {px} must be smaller than the source's {src_px} (this only shrinks)")
    (dst_dir / "shards").mkdir(parents=True, exist_ok=True)
    derived = {"src": str(src_dir), "src_out_px": src_px, "interp": "cv2.INTER_AREA",
               "src_git_sha": man.get("git_sha")}
    mf = dst_dir / "cache_manifest.json"
    if mf.exists() and json.loads(mf.read_text()).get("derived_from", {}).get("src_out_px") != src_px:
        raise SystemExit(f"{mf} exists and wasn't derived from a {src_px}-px cache; use a new --dst")
    shards = sorted((src_dir / "shards").glob("shard_?????.u8"))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:  # cv2 releases the GIL
        n_slices = sum(ex.map(lambda f: derive_shard(f, dst_dir / "shards" / f.name, src_px, px), shards))
    for name in ("series_index.parquet", "study_index.parquet"):
        shutil.copy2(src_dir / name, dst_dir / name)
    out = {**man, "params": {**man["params"], "out_px": px}, "derived_from": derived,
           "bytes": int(sum(f.stat().st_size for f in (dst_dir / "shards").glob("shard_?????.u8"))),
           "derived_seconds": round(time.time() - t0, 1), "derived_n_slices": int(n_slices)}
    mf.write_text(json.dumps(out, indent=2))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--px", type=int, default=256)
    ap.add_argument("--dst", type=Path, help="default: <src>_<px>")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args(argv)
    dst = a.dst or a.src.with_name(f"{a.src.name}_{a.px}")
    r = derive(a.src, dst, a.px, a.workers)
    print(f"{dst}: {r['derived_n_slices']} slices, {r['bytes'] / 1e9:.1f} GB, {r['derived_seconds']:.0f} s")


if __name__ == "__main__":
    main()
