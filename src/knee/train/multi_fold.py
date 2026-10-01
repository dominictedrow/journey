"""Train several folds at once, one per GPU (Kaggle's T4 ×2): each fold is an ordinary train_sup process pinned to
one GPU with CUDA_VISIBLE_DEVICES, so nothing inside the trainer changes and every fold keeps its own resumable run.

    python -m knee.train.multi_fold --config configs/trackA_coatnet.yaml --folds 0 1 2 3 4 [--gpus 0 1]
                                    [--hours 11] [--workers-per-job 1] [--set optim.epochs=16 ...]

Folds wait in a queue; when a GPU frees up, the next fold starts on it. With one GPU this is the same as running the
folds one after another. Why not DistributedDataParallel: we need 5 folds anyway, and independent folds on separate
GPUs give the same throughput with no gradient sync, no change to the trainer, and no shared failure.

--hours: a wall-clock budget (e.g. a little under the Kaggle session limit). When it runs out, every running fold
gets a STOP file (train_sup saves last.pt at its next step and exits) and no new fold starts. Re-running the same
command resumes each unfinished fold from its last.pt. Logs: <run_name dir>/multi_fold/fold<k>.log.

Quota: billed per session wall-clock, not per GPU (P04 addendum), so the second fold is free.
UNVERIFIED: the Kaggle GPU session time limit,
and whether 4 vCPUs feed two Track A loaders without starving them (watch studies_per_s in each fold's log.jsonl).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable


def run_folds(folds: list[int], gpus: list[str], cmd_for_fold: Callable[[int], list[str]], log_dir: str | Path,
              poll_s: float = 5.0, deadline: float | None = None,
              stop_file_for_fold: Callable[[int], Path] | None = None,
              env: dict[str, str] | None = None) -> dict[int, int | None]:
    """Run each fold's command with CUDA_VISIBLE_DEVICES set to a free GPU, at most one fold per GPU at a time.
    Returns fold → exit code (None = never started because the deadline passed). A failed fold doesn't stop the
    others. At `deadline` (time.time() value) running folds get their STOP file and are waited for."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = list(folds)
    free = list(gpus)
    running: dict[int, tuple[subprocess.Popen, str, object, float]] = {}
    codes: dict[int, int | None] = {}
    stopping = False
    while pending or running:
        if deadline is not None and time.time() >= deadline and not stopping:
            stopping = True
            print(f"[multi_fold] time budget reached: STOP to folds {sorted(running)}; not starting {pending}",
                  flush=True)
            for f in running:
                if stop_file_for_fold is not None:
                    p = stop_file_for_fold(f)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.touch()
            for f in pending:
                codes[f] = None
            pending = []
        while pending and free and not stopping:
            f, g = pending.pop(0), free.pop(0)
            fh = open(log_dir / f"fold{f}.log", "a")
            e = {**os.environ, **(env or {}), "CUDA_VISIBLE_DEVICES": g, "PYTHONUNBUFFERED": "1"}
            proc = subprocess.Popen(cmd_for_fold(f), stdout=fh, stderr=subprocess.STDOUT, env=e)
            running[f] = (proc, g, fh, time.time())
            print(f"[multi_fold] fold {f} → GPU {g} (pid {proc.pid}); log {log_dir / f'fold{f}.log'}", flush=True)
        for f in list(running):
            proc, g, fh, t0 = running[f]
            rc = proc.poll()
            if rc is None:
                continue
            fh.close()
            codes[f] = rc
            free.append(g)
            del running[f]
            print(f"[multi_fold] fold {f} on GPU {g} exited {rc} after {(time.time() - t0) / 3600:.2f} h", flush=True)
        if pending or running:
            time.sleep(poll_s)
    return codes


def visible_gpus() -> list[str]:
    """GPU indices to use: the parent's CUDA_VISIBLE_DEVICES if set, else every device torch sees."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    import torch
    return [str(i) for i in range(torch.cuda.device_count())]


def main(argv=None) -> int:
    from knee.train.train_sup import load_cfg, resolve_dirs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpus", nargs="*", default=None, help="GPU indices (default: all visible)")
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget; then STOP + resume later")
    ap.add_argument("--workers-per-job", type=int, default=None,
                    help="data.workers per fold (default: CPUs // jobs - 1, at least 1)")
    ap.add_argument("--set", nargs="*", default=[], help="train_sup overrides, passed to every fold")
    a = ap.parse_args(argv)
    gpus = a.gpus if a.gpus is not None else visible_gpus()
    if not gpus:
        raise SystemExit("no GPU visible; run knee.train.train_sup directly on CPU")
    n_jobs = min(len(gpus), len(a.folds))
    workers = a.workers_per_job if a.workers_per_job is not None else max(1, (os.cpu_count() or 2) // n_jobs - 1)

    def cfg_for(fold: int) -> dict:
        return resolve_dirs(load_cfg(a.config, [*a.set, f"folds.fold={fold}"]))

    run_root = Path(cfg_for(a.folds[0])["run_dir"]).parent
    cmd = lambda f: [sys.executable, "-m", "knee.train.train_sup", "--config", str(a.config),  # noqa: E731
                     "--set", *a.set, f"folds.fold={f}", f"data.workers={workers}"]
    import knee
    pkg_root = str(Path(knee.__file__).resolve().parents[1])  # so the children import the same knee (Kaggle: a dataset)
    env = {"PYTHONPATH": os.pathsep.join([pkg_root, *filter(None, [os.environ.get("PYTHONPATH")])])}
    print(f"[multi_fold] folds {a.folds} on GPUs {gpus}, {workers} loader workers each; runs under {run_root}",
          flush=True)
    codes = run_folds(a.folds, gpus, cmd, run_root / "multi_fold",
                      deadline=time.time() + a.hours * 3600 if a.hours else None,
                      stop_file_for_fold=lambda f: Path(cfg_for(f)["run_dir"]) / "STOP", env=env)
    # exit 0 also covers a fold paused by STOP, so "finished" = train_sup wrote its final metrics.json
    done = {f: (Path(cfg_for(f)["run_dir"]) / "metrics.json").exists() for f in a.folds}
    print(f"[multi_fold] exit codes {codes}; finished {[f for f in a.folds if done[f]]}; "
          f"to resume {[f for f in a.folds if not done[f]]}", flush=True)
    return 0 if all(done.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
