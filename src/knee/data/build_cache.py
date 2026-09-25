"""Build the preprocessed pixel cache (S02 step 6). Resumable; one process per shard of studies.

    python -m knee.data.build_cache --split train --workers 12
    python -m knee.data.build_cache --split train --limit-studies 40 --out C:/knee_cache/smoke   # smoke test

Layout of <cache_root>/cache_<version>_<split>/:
    shards/shard_00000.u8          raw uint8, shape (n_slices_in_shard, out_px, out_px), C order, no header.
                                   A study's series are written back to back, so one study = one contiguous read.
    shards/shard_00000.parquet     one row per series (the index rows of this shard)
    shards/shard_00000.studies.parquet   one row per study (laterality, slots, errors, seconds)
    series_index.parquet           all shards' series rows merged (written at the end)
    study_index.parquet            all shards' study rows merged
    cache_manifest.json            params, version, git sha, counts, size, timing

A shard counts as done when its .parquet exists (it is written last, after the .u8 is renamed into place), so an
interrupted build resumes at shard granularity. Memory per worker ≈ one study, never a whole shard.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from knee.data.preprocess import PreprocParams, preprocess_study, series_info_from_csv
from knee.paths import REPO_ROOT, load_paths

log = logging.getLogger(__name__)


def _bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes"):
        return True
    if s.lower() in ("0", "false", "no"):
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {s}")


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def cache_dir_for(cache_root: Path, split: str, params: PreprocParams) -> Path:
    return Path(cache_root) / f"cache_{params.version}_{split}"


def build_shard(shard_id: int, study_uids: list[str], split_dir: str, shard_dir: str, params_d: dict,
                series_info: dict) -> dict:
    """Worker: preprocess each study and append its slices to one raw uint8 file."""
    import cv2
    cv2.setNumThreads(1)  # one process per core; no nested threading
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    params = PreprocParams(**params_d)
    shard_dir_p = Path(shard_dir)
    tmp = shard_dir_p / f"shard_{shard_id:05d}.u8.tmp"
    series_rows, study_rows = [], []
    offset = 0
    t_shard = time.time()
    with open(tmp, "wb") as fh:
        for uid in study_uids:
            t0 = time.time()
            err = None
            try:
                series, smeta = preprocess_study(Path(split_dir) / uid, series_info, params)
            except Exception as e:  # noqa: BLE001 - record and keep going
                log.exception("study %s failed", uid)
                series, smeta, err = [], {"study_uid": uid, "slots": {}}, f"{type(e).__name__}: {e}"[:300]
            for s in series:
                fh.write(np.ascontiguousarray(s.pixels).tobytes())
                row = {"study_uid": uid, "shard": shard_id, "offset": offset, **s.meta,
                       "positions_mm": s.positions_mm.astype(np.float32).tolist(),
                       "valid": s.valid.astype(bool).tolist()}
                series_rows.append(row)
                offset += len(s.pixels)
            study_rows.append({**{k: v for k, v in smeta.items() if k != "slots"},
                               **{f"slot_{k}": v for k, v in smeta.get("slots", {}).items()},
                               "shard": shard_id, "error": err, "seconds": time.time() - t0,
                               "n_slices": int(sum(len(s.pixels) for s in series))})
    final = shard_dir_p / f"shard_{shard_id:05d}.u8"
    os.replace(tmp, final)
    pd.DataFrame(study_rows).to_parquet(shard_dir_p / f"shard_{shard_id:05d}.studies.parquet", index=False)
    pd.DataFrame(series_rows).to_parquet(shard_dir_p / f"shard_{shard_id:05d}.parquet", index=False)  # done marker
    return {"shard": shard_id, "n_studies": len(study_uids), "n_slices": offset, "bytes": final.stat().st_size,
            "seconds": time.time() - t_shard}


def merge_index(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sd = out_dir / "shards"
    ser = pd.concat([pd.read_parquet(p) for p in sorted(sd.glob("shard_?????.parquet"))], ignore_index=True)
    stu = pd.concat([pd.read_parquet(p) for p in sorted(sd.glob("shard_?????.studies.parquet"))], ignore_index=True)
    ser.to_parquet(out_dir / "series_index.parquet", index=False)
    stu.to_parquet(out_dir / "study_index.parquet", index=False)
    return ser, stu


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out", type=Path, help="cache dir (default <cache_root>/cache_<version>_<split>)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--studies-per-shard", type=int, default=16)
    ap.add_argument("--limit-studies", type=int, default=0, help="first N studies only (smoke test)")
    for f in fields(PreprocParams):
        if f.name != "version":
            typ = _bool if isinstance(f.default, bool) else type(f.default)
            ap.add_argument(f"--{f.name.replace('_', '-')}", type=typ, default=f.default)
    a = ap.parse_args()
    params = PreprocParams(**{f.name: getattr(a, f.name) for f in fields(PreprocParams) if f.name != "version"})

    if a.data_root is None or a.out is None:
        p = load_paths()
        a.data_root = a.data_root or p.data_root
        a.out = a.out or cache_dir_for(p.cache_root, a.split, params)
    split_dir = a.data_root / f"{a.split}_series"
    shard_dir = a.out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest_f = a.out / "cache_manifest.json"
    if manifest_f.exists():
        old = json.loads(manifest_f.read_text())
        if old.get("params") != params.to_dict():
            raise SystemExit(f"{manifest_f} was built with different params:\n{old.get('params')}\nvs\n"
                             f"{params.to_dict()}\nUse a new --out or delete the old cache.")

    series_csv = pd.read_csv(a.data_root / f"{a.split}_series.csv")
    series_info = series_info_from_csv(series_csv)
    studies = sorted(e.name for e in os.scandir(split_dir) if e.is_dir())
    if a.limit_studies:
        studies = studies[: a.limit_studies]
    shards = [studies[i: i + a.studies_per_shard] for i in range(0, len(studies), a.studies_per_shard)]
    todo = [i for i in range(len(shards)) if not (shard_dir / f"shard_{i:05d}.parquet").exists()]
    manifest = {"version": params.version, "split": a.split, "params": params.to_dict(), "git_sha": git_sha(),
                "n_studies": len(studies), "studies_per_shard": a.studies_per_shard, "n_shards": len(shards),
                "dtype": "uint8", "layout": "shards/shard_XXXXX.u8 raw (n, out_px, out_px) C-order",
                "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    manifest_f.write_text(json.dumps(manifest, indent=2))
    print(f"{len(studies)} studies in {len(shards)} shards; {len(todo)} to do; {a.workers} workers → {a.out}",
          flush=True)

    t0, done_studies, done_bytes, done_slices = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(build_shard, i, shards[i], str(split_dir), str(shard_dir), params.to_dict(), series_info)
                for i in todo]
        todo_studies = sum(len(shards[i]) for i in todo)
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            done_studies += r["n_studies"]
            done_bytes += r["bytes"]
            done_slices += r["n_slices"]
            el = time.time() - t0
            rate = done_studies / max(el, 1e-9)
            eta = (todo_studies - done_studies) / max(rate, 1e-9)
            proj_gb = done_bytes / done_studies * len(studies) / 1e9
            print(f"[{k}/{len(todo)}] shard {r['shard']}: {r['n_studies']} studies {r['n_slices']} slices "
                  f"| {rate:.2f} studies/s {done_slices / max(el, 1e-9):.0f} slices/s | {el / 60:.1f} min, "
                  f"ETA {eta / 60:.0f} min | projected {proj_gb:.1f} GB", flush=True)

    ser, stu = merge_index(a.out)
    total_bytes = sum(p.stat().st_size for p in shard_dir.glob("shard_?????.u8"))
    manifest.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "wall_seconds_this_run": time.time() - t0,
                     "workers": a.workers, "n_series": int(len(ser)), "n_slices": int(ser.n_slices.sum()),
                     "n_study_errors": int(stu.error.notna().sum()), "bytes": int(total_bytes),
                     "sum_study_seconds": float(stu.seconds.sum())})
    manifest_f.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("n_studies", "n_series", "n_slices", "n_study_errors", "bytes",
                                                "wall_seconds_this_run")}, indent=2))


if __name__ == "__main__":
    main()
