"""AB.md Part 1.3: check the 7 cache datasets on Kaggle (private CPU script kernel, no GPU quota).

1. finds the 7 mounts and builds the symlink farm /tmp/cache = what CacheReader expects (farm.py)
2. byte-compares every shard against the series index (max(offset + n_slices) × out_px²)
3. Track A loader speed: StudyWindowsDataset (slots, train) through a DataLoader at workers 1/2/3, a fresh set of
   studies per setting (no page-cache reuse); reports studies/s and the decoded-window MB/s (output, not disk reads)
4. df -h, nproc, free -g
5. kernel-output attachment: is knee-audit's output (a kernel_source) mounted, and where? (decides Part 3.4)

Push (Dom's OK): kaggle kernels push -p notebooks/kaggle_cachecheck
"""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

src = next(Path(d) for p in ("/kaggle/input/datasets/*/knee-src", "/kaggle/input/datasets/*/knee-src/*",
                             "/kaggle/input/knee-src", "/kaggle/input/knee-src/*")
           for d in sorted(glob.glob(p)) if (Path(d) / "knee/__init__.py").exists())
sys.path.insert(0, str(src))
from knee.kaggle_farm import build_farm  # noqa: E402

out = {}
cache, _ = build_farm(Path("/tmp/cache"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from knee.data.dataset import CacheReader, StudyWindowsDataset, worker_init  # noqa: E402

# 2. bytes
man = json.loads((cache / "cache_manifest.json").read_text())
px = int(man["params"]["out_px"])
ser = pd.read_parquet(cache / "series_index.parquet")
need = (ser.offset + ser.n_slices).groupby(ser.shard).max() * px * px
bad = {int(s): [int(n), os.path.getsize(cache / "shards" / f"shard_{int(s):05d}.u8")] for s, n in need.items()
       if os.path.getsize(cache / "shards" / f"shard_{int(s):05d}.u8") != n}
out["shards_found"] = len(list((cache / "shards").glob("*.u8")))
out["shards_expected"] = man["n_shards"]
out["shard_size_mismatches"] = bad
out["bytes_total"] = sum(os.path.getsize(p) for p in (cache / "shards").glob("*.u8"))
out["bytes_manifest"] = man["bytes"]
print(json.dumps(out, indent=1), flush=True)

# 3. loader speed
reader = CacheReader(cache)
uids = list(np.random.default_rng(0).permutation(reader.study_uids))
N = 48
speed = {}
for k, w in enumerate([1, 2, 3]):
    ds = StudyWindowsDataset(reader, uids[k * N:(k + 1) * N], None, windows_per_slot=3, train=True, out_px=px,
                             slot_dropout=0.1)
    dl = DataLoader(ds, batch_size=1, num_workers=w, worker_init_fn=worker_init, collate_fn=None)
    t0, mb = time.time(), 0.0
    for b in dl:
        mb += b["x"].numel() / 1e6
    dt = time.time() - t0
    speed[w] = {"studies_per_s": round(N / dt, 2), "window_MB_per_s": round(mb / dt, 1), "s": round(dt, 1)}
    print("workers", w, speed[w], flush=True)
out["loader"] = speed

# 4. box
for c in (["df", "-h", "/tmp", "/kaggle/working"], ["nproc"], ["free", "-g"]):
    out[" ".join(c)] = subprocess.run(c, capture_output=True, text=True).stdout

# 5. kernel output as input
att = [str(p) for pat in ("/kaggle/input/*knee-audit*", "/kaggle/input/*/*/knee-audit*", "/kaggle/input/notebooks/*/*")
       for p in Path("/").glob(pat.lstrip("/"))]
out["kernel_output_mounts"] = {a: sorted(os.listdir(a))[:20] for a in att}
out["input_tree"] = sorted(str(p) for p in Path("/kaggle/input").glob("*/*"))[:60]

Path("/kaggle/working/cachecheck.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
