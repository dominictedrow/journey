"""S01 part 2 on Kaggle (private CPU script kernel, internet off): the data audit against the mounted competition data,
so neither the Windows box nor a local copy of the 570 GB is needed.

Inputs: the competition data + two private datasets, `knee-src` (a copy of src/knee, code only) and `knee-wheels`
(decoder wheels built on this image by notebooks/kaggle_probe). Runs the same modules as scripts/run_s01.ps1, minus
env_check (Windows-box facts):
    csv_audit → header_scan test/train → series_summary test/train → decode_bench (train) → write_reports train/test
Outputs in /kaggle/working/: audit/ (parquet/json, UIDs inside: MIRA-derived, stays private) and docs/*.md
(aggregates only, the files to commit). Fetch only what's needed with `kaggle kernels output dominictedrow/knee-audit`.

Push (Dom's OK given 2026-09-29): kaggle kernels push -p notebooks/kaggle_audit
"""
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")


def find(marker: str, pats: list[str]) -> Path:
    for pat in pats:
        for d in sorted(glob.glob(pat)):
            if (Path(d) / marker).exists():
                return Path(d)
    raise FileNotFoundError(f"{marker} not found under {pats}")


comp = find("train.csv", ["/kaggle/input/competitions/*", "/kaggle/input/*"])
# Dataset mounts: /kaggle/input/datasets/<user>/<slug>/ or /kaggle/input/<slug>/ (UNVERIFIED which). Shallow globs only:
# a recursive glob over /kaggle/input would walk the 855k-file competition tree.
DS = ["/kaggle/input/datasets/*/{s}", "/kaggle/input/datasets/*/{s}/*", "/kaggle/input/{s}", "/kaggle/input/{s}/*"]
src = find("knee/__init__.py", [p.format(s="knee-src") for p in DS])
wheels = next((d for p in DS for d in sorted(glob.glob(p.format(s="knee-wheels"))) if glob.glob(f"{d}/*.whl")), None)
print("input:", sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/datasets/*/*")))
print("comp:", comp, "| src:", src, "| wheels:", wheels, flush=True)

if wheels:  # --no-deps: keep the image's numpy (docs/kaggle_env.md §2)
    whl = sorted(glob.glob(f"{wheels}/*.whl"))
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *whl],
                       capture_output=True, text=True)
    print("wheels rc", r.returncode, r.stderr[-400:], flush=True)

env = {**os.environ, "PYTHONPATH": str(src), "KNEE_DATA_ROOT": str(comp),
       "KNEE_AUDIT_OUT": str(WORK / "audit"), "KNEE_CACHE_ROOT": str(WORK / "cache")}
(WORK / "docs").mkdir(parents=True, exist_ok=True)
NOTE = ("Generated on a Kaggle CPU notebook (4 vCPU Xeon 2.2 GHz, 32 GB; data read from the competition NFS mount) "
        f"by notebooks/kaggle_audit, {time.strftime('%Y-%m-%d')}. Decode throughput here is Kaggle's, not the Windows box's.")
IO_WORKERS = "16"  # header reads are NFS-latency bound, not CPU bound

steps = [
    ["csv_audit"],
    ["header_scan", "--split", "test", "--workers", "2"],
    ["header_scan", "--split", "train", "--workers", IO_WORKERS],
    ["series_summary", "--split", "test"],
    ["series_summary", "--split", "train"],
    ["decode_bench", "--split", "train", "--files-per-config", "300", "--workers", "1", "4", "8", "16"],
    ["write_reports", "--split", "train", "--docs", str(WORK / "docs"), "--note", NOTE],
    ["write_reports", "--split", "test", "--docs", str(WORK / "docs"), "--note", NOTE],
]
status = {}
for s in steps:
    t0 = time.time()
    print(f"\n===== {' '.join(s)}", flush=True)
    r = subprocess.run([sys.executable, "-m", f"knee.audit.{s[0]}", *s[1:]], env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-6000:], flush=True)
    status[" ".join(s[:3])] = {"rc": r.returncode, "s": round(time.time() - t0, 1)}
    print(f"===== rc={r.returncode} in {time.time() - t0:.0f}s", flush=True)

import json  # noqa: E402
(WORK / "audit" / "kaggle_run.json").write_text(json.dumps(status, indent=1))
print(json.dumps(status, indent=1))
