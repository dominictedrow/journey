"""MAE loader throughput (docs/plan/03_performance_plan.md §3, A3–A5): windows/s the DataLoader delivers, with no GPU
and no model. Compare it with what the GPU consumes (T4 ≈ 315 windows/s measured, 5070 ≈ 2× that, UNVERIFIED) to see
whether MAE pretraining is loader-bound.

    python scripts/perf/mae_loader_bench.py --cache <cache_v1_train> [--cache <cache_v1_train_256>] \
        [--workers 4 8] [--read 1 4] [--batches 60]

Each combination of cache × workers × windows_per_read × crop (cpu = cv2 in the workers; gpu = workers only read, the
crop runs later on the GPU) prints windows/s and MB/s read. The first `--skip` batches are excluded (worker start-up,
cold page cache). Run it on the machine and drive that will train: the SATA SSD on Windows, or NFS on Kaggle.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from knee.data.dataset import CacheReader, worker_init  # noqa: E402
from knee.ssl.mae_data import MAEWindowDataset, collate_windows  # noqa: E402


def bench(cache: Path, workers: int, kr: int, gpu_crop: bool, batch: int, batches: int, skip: int, out_px: int,
          limit_studies: int, cv2_1thread: bool = True) -> dict:
    r = CacheReader(cache)
    uids = r.study_uids[:limit_studies] if limit_studies else None
    ds = MAEWindowDataset(r, uids, out_px, windows_per_series=max(4, kr), windows_per_read=kr, gpu_crop=gpu_crop)
    dl = DataLoader(ds, batch_size=batch // kr, shuffle=True, num_workers=workers, collate_fn=collate_windows,
                    drop_last=True, prefetch_factor=4 if workers else None, persistent_workers=False,
                    worker_init_fn=worker_init if workers and cv2_1thread else None)
    n = nbytes = 0
    t0 = None
    for i, b in enumerate(dl):
        if i == skip:
            t0 = time.perf_counter()
        elif i > skip:
            x = b if not isinstance(b, tuple) else b[0]
            n += len(x)
            nbytes += len(x) * 3 * r.out_px * r.out_px  # slices actually read are fewer when windows overlap
        if i >= skip + batches:
            break
    dt = time.perf_counter() - t0 if t0 else float("nan")
    return {"cache_px": r.out_px, "workers": workers, "windows_per_read": kr, "crop": "gpu" if gpu_crop else "cpu",
            "cv2_threads": "1" if cv2_1thread else "default",
            "windows_per_s": n / dt, "MB_per_s_upper": nbytes / dt / 1e6, "batches": batches}


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, action="append", required=True)
    ap.add_argument("--workers", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--read", type=int, nargs="+", default=[1, 4], help="windows_per_read values")
    ap.add_argument("--crop", nargs="+", default=["cpu", "gpu"], choices=["cpu", "gpu"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--out-px", type=int, default=256)
    ap.add_argument("--limit-studies", type=int, default=0)
    ap.add_argument("--cv2-threads", nargs="+", default=["1"], choices=["1", "default"],
                    help="'1' = the trainers' worker_init (cv2.setNumThreads(1)); 'default' = cv2's own pool. Pass "
                         "both to measure the oversubscription (bug.md #9)")
    a = ap.parse_args(argv)
    rows = []
    print("| cache px | workers | windows/read | crop | cv2 threads | windows/s | MB/s (upper) |\n"
          "|---|---|---|---|---|---|---|")
    for cache, w, kr, crop, ct in itertools.product(a.cache, a.workers, a.read, a.crop, a.cv2_threads):
        r = bench(cache, w, kr, crop == "gpu", a.batch, a.batches, a.skip, a.out_px, a.limit_studies, ct == "1")
        rows.append(r)
        print(f"| {r['cache_px']} | {w} | {kr} | {r['crop']} | {ct} | {r['windows_per_s']:.0f} | "
              f"{r['MB_per_s_upper']:.0f} |", flush=True)
    return rows


if __name__ == "__main__":
    main()
