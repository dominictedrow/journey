"""AB.md Part 3.3: Track A training on Kaggle T4 ×2 (private GPU script kernel). One session = `multi_fold` over FOLDS
for HOURS, then STOP → last.pt; the next session resumes from the previous session's runs (RUNS_SLUGS mount).

Inputs: the competition (gold train.csv), knee-src, the 7 cache datasets (knee.kaggle_farm), optionally the previous
runs. Internet on: timm downloads the ImageNet backbone from the HF hub (a training kernel, not a submission).
Outputs: /kaggle/working/runs/trackA/{fold<k>/, multi_fold/}, nvidia_smi.csv. train_sup already keeps only the
top-k epoch snapshots (~2 GB per fold; the output cap is ~20.9 GB).

Push (Dom's OK per session): kaggle kernels push -p notebooks/kaggle_train --accelerator NvidiaTeslaT4
"""
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

FOLDS = [1, 2]
HOURS = 8.5            # session cap − 30 min (9 h plan; AB.md Part 3.6); smoke session was 0.75
REQUIRE_PREV = False   # True on resume sessions: fail fast instead of silently training from scratch
# knee-profile v2 (2026-10-08): ckpt off + cl + compile = 527 ms/step median, 9.7 GB (vs 722 ms uncompiled)
EXTRA_SETS = ["model.cfg.grad_checkpointing=false", "perf.channels_last=true", "perf.compile=true"]
RUNS_SLUGS = []        # previous runs to resume (v1 session 2: ["knee-train-v1"]). v0 runs must never be mounted here
LABELS_SLUG, LABELS_FILE = "knee-labels-v1", "labels_v1.parquet"   # E018: labels_v1 beats v0
RUN_NAME = "trackA_coatnet384_lv1"

WORK = Path("/kaggle/working")
src = next(Path(d) for p in ("/kaggle/input/datasets/*/knee-src", "/kaggle/input/datasets/*/knee-src/*",
                             "/kaggle/input/knee-src", "/kaggle/input/knee-src/*")
           for d in sorted(glob.glob(p)) if (Path(d) / "knee/__init__.py").exists())
sys.path.insert(0, str(src))
from knee.kaggle_farm import build_farm  # noqa: E402

comp = next(Path(d) for p in ("/kaggle/input/competitions/*", "/kaggle/input/*") for d in sorted(glob.glob(p))
            if (Path(d) / "train.csv").exists())
cache, idx = build_farm(Path("/tmp/cache"))
print("input:", sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*/*")), flush=True)

labels_file = next(Path(d) for p in (f"/kaggle/input/datasets/*/{LABELS_SLUG}", f"/kaggle/input/{LABELS_SLUG}")
                   for d in sorted(glob.glob(f"{p}/{LABELS_FILE}")))
runs = WORK / "runs" / "trackA"
prev = [Path(d) for s in RUNS_SLUGS for p in ("/kaggle/input/{s}", "/kaggle/input/*/*/{s}", "/kaggle/input/{s}/*")
        for d in glob.glob(f"{p.format(s=s)}/runs/trackA") + glob.glob(f"{p.format(s=s)}/trackA")]
if REQUIRE_PREV and not prev:
    raise SystemExit(f"no previous runs found for {RUNS_SLUGS}: {sorted(glob.glob('/kaggle/input/*/*/*'))}")
if prev and not runs.exists():
    print("resuming from", prev[0], flush=True)
    shutil.copytree(prev[0], runs)

smi = open(WORK / "nvidia_smi.csv", "w")
subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,clocks.sm,power.draw,temperature.gpu,"
                  "memory.used,clocks_throttle_reasons.active", "--format=csv", "-l", "60"], stdout=smi)

cmd = [sys.executable, "-m", "knee.train.multi_fold", "--config", str(idx / "trackA_coatnet.yaml"),
       "--folds", *map(str, FOLDS), "--hours", str(HOURS), "--set", "cache_dir=/tmp/cache", f"run_dir={runs}",
       f"labels.file={labels_file}", f"folds.file={idx / 'folds_v0.csv'}", f"run_name={RUN_NAME}",
       f"gold.file={comp / 'train.csv'}", *EXTRA_SETS]
print(" ".join(cmd), flush=True)
rc = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": str(src)}).returncode
for f in sorted(runs.glob("multi_fold/fold*.log")):
    print(f"===== {f.name} (tail)\n" + f.read_text()[-3000:], flush=True)
print("multi_fold rc", rc)
