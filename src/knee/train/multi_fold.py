"""Train several folds at once, one per GPU (Kaggle's T4 ×2): each fold is an ordinary train_sup process pinned to
one GPU with CUDA_VISIBLE_DEVICES, so nothing inside the trainer changes and every fold keeps its own resumable run.

    python -m knee.train.multi_fold --config configs/trackA_coatnet.yaml --folds 0 1 2 3 4 [--gpus 0 1]
                                    [--hours 11] [--workers-per-job 1] [--set optim.epochs=16 ...]

Folds wait in a queue; when a GPU frees up, the next fold starts on it. With one GPU this is the same as running the
folds one after another. Why not DistributedDataParallel: we need 5 folds anyway, and independent folds on separate
GPUs give the same throughput with no gradient sync, no change to the trainer, and no shared failure.

--hours: a wall-clock budget (e.g. a little under the Kaggle session limit). When it runs out, every running fold
gets a STOP file (train_sup saves last.pt at its next step and exits) and no new fold starts. Re-running the same
command resumes each unfinished fold from its last.pt. Logs: <run root>/multi_fold/fold<k>.log.

Resuming: a fold whose metrics.json says it finished (same fold, epochs >= the configured optim.epochs) is not
launched again, so a resumed command spends the GPUs on the unfinished folds only. Raising optim.epochs re-launches
finished folds to extend them; --rerun-finished re-launches them anyway (e.g. to redo the top-k average after changing
select.top_k). A STOP file left in a fold's run dir by an earlier deadline is removed just before that fold starts.

Run dirs: with run_dir null, each fold gets <runs>/<run_name>/fold<k> from configs/paths.yaml (run root =
<runs>/<run_name>). An explicit run_dir X (e.g. /kaggle/working/... without paths.yaml) is the run root: fold k runs in
X/fold<k>, never in X itself, so two folds can never share (and resume) one directory.

Quota: billed per session wall-clock, not per GPU (P04 addendum), so the second fold is free.
UNVERIFIED: the Kaggle GPU session time limit,
and whether 4 vCPUs feed two Track A loaders without starving them (watch studies_per_s in each fold's log.jsonl).
"""
from __future__ import annotations

import argparse
import json
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
            if stop_file_for_fold is not None:  # a STOP that exists before launch is left over from an old deadline
                stop_file_for_fold(f).unlink(missing_ok=True)
            fh = open(log_dir / f"fold{f}.log", "a", encoding="utf-8")
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


def is_finished(run_dir: str | Path, fold: int, epochs: int) -> bool:
    """train_sup wrote its final metrics.json for this fold with at least `epochs` epochs. A metrics.json written
    before it recorded `epochs` counts as finished."""
    p = Path(run_dir) / "metrics.json"
    if not p.exists():
        return False
    m = json.loads(p.read_text(encoding="utf-8"))
    return int(m.get("fold", fold)) == fold and int(m.get("epochs", epochs)) >= epochs


def main(argv=None) -> int:
    from knee.train.train_sup import load_cfg, resolve_dirs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--gpus", nargs="*", default=None, help="GPU indices (default: all visible)")
    ap.add_argument("--hours", type=float, default=None, help="wall-clock budget; then STOP + resume later")
    ap.add_argument("--workers-per-job", type=int, default=None,
                    help="data.workers per fold (default: the config's data.workers, capped at CPUs // jobs - 1, at least 1)")
    ap.add_argument("--rerun-finished", action="store_true",
                    help="also launch folds whose metrics.json says they finished (redo top-k average + eval)")
    ap.add_argument("--set", nargs="*", default=[], help="train_sup overrides, passed to every fold")
    a = ap.parse_args(argv)
    gpus = a.gpus if a.gpus is not None else visible_gpus()
    if not gpus:
        raise SystemExit("no GPU visible; run knee.train.train_sup directly on CPU")

    explicit = load_cfg(a.config, a.set)["run_dir"]
    cfgs = {f: resolve_dirs(load_cfg(a.config, [*a.set, f"folds.fold={f}"])) for f in a.folds}
    if explicit is not None:  # explicit run_dir = run root; one subdir per fold (see the module docstring)
        run_root = Path(explicit)
        for f in a.folds:
            cfgs[f]["run_dir"] = str(run_root / f"fold{f}")
        print(f"[multi_fold] explicit run_dir {run_root} is the run root: fold k runs in {run_root}/fold<k>",
              flush=True)
    else:
        run_root = Path(cfgs[a.folds[0]]["run_dir"]).parent
    run_dir = {f: Path(c["run_dir"]) for f, c in cfgs.items()}
    if len(set(run_dir.values())) != len(a.folds):
        raise SystemExit(f"folds share a run dir: {run_dir}")

    finished = [f for f in a.folds if is_finished(run_dir[f], f, int(cfgs[f]["optim"]["epochs"]))]
    pending = a.folds if a.rerun_finished else [f for f in a.folds if f not in finished]
    if finished:
        print(f"[multi_fold] already finished: {finished}" + (" (re-running: --rerun-finished)"
                                                             if a.rerun_finished else ""), flush=True)
    if not pending:
        print("[multi_fold] nothing to run", flush=True)
        return 0
    n_jobs = min(len(gpus), len(pending))
    # cap at the config's data.workers: on one GPU, CPUs - 1 workers (15 on the Windows box) ran 16 GB of RAM out
    workers = a.workers_per_job if a.workers_per_job is not None else max(
        1, min(int(cfgs[a.folds[0]]["data"]["workers"]), (os.cpu_count() or 2) // n_jobs - 1))

    def cmd(f: int) -> list[str]:
        extra = [f"run_dir={json.dumps(str(run_dir[f]))}"] if explicit is not None else []  # JSON = valid YAML
        return [sys.executable, "-m", "knee.train.train_sup", "--config", str(a.config),
                "--set", *a.set, f"folds.fold={f}", f"data.workers={workers}", *extra]
    import knee
    pkg_root = str(Path(knee.__file__).resolve().parents[1])  # so the children import the same knee (Kaggle: a dataset)
    env = {"PYTHONPATH": os.pathsep.join([pkg_root, *filter(None, [os.environ.get("PYTHONPATH")])])}
    print(f"[multi_fold] folds {pending} on GPUs {gpus}, {workers} loader workers each; runs under {run_root}",
          flush=True)
    codes = run_folds(pending, gpus, cmd, run_root / "multi_fold",
                      deadline=time.time() + a.hours * 3600 if a.hours else None,
                      stop_file_for_fold=lambda f: run_dir[f] / "STOP", env=env)
    # exit 0 also covers a fold paused by STOP, so "finished" = train_sup wrote its final metrics.json
    done = {f: is_finished(run_dir[f], f, int(cfgs[f]["optim"]["epochs"])) for f in a.folds}
    print(f"[multi_fold] exit codes {codes}; finished {[f for f in a.folds if done[f]]}; "
          f"to resume {[f for f in a.folds if not done[f]]}", flush=True)
    return 0 if all(done.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
