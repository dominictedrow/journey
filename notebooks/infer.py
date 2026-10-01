"""Kaggle inference script kernel (S06): test DICOMs → /kaggle/working/submission.csv. Internet off.

Inputs attached to the kernel:
- the competition data, mounted at `/kaggle/input/competitions/<slug>/` or `/kaggle/input/<slug>/` (public notebooks use
  both; resolved at runtime, never by walking the 570 GB tree);
- our **bundle** dataset (built by `scripts/build_kaggle_bundle.py`, uploaded only with Dom's OK):
      knee_src/knee/...   the repo's `src/knee` package: the same preprocessing + window code as training
      arms.json           {"arms": [{"ckpt": "weights/<name>.pt", "weight": 1.0}, ...], "how": "mean"|"rank",
                           "prior": [12 floats] (fallback row), "preproc": {PreprocParams overrides},
                           "perf": {"half", "compile", "channels_last", "window_chunk", "studies_per_forward", ...}
                           (predict.PERF_DEFAULTS; an arm's own "perf" overrides it, except studies_per_forward)}
      weights/*.pt        train_sup `model_avg.pt` checkpoints
      wheels/*.whl        optional offline wheels (pylibjpeg, openjpeg, libjpeg, gdcm; timm if the image lacks it)

Stages (each timed; the efficiency score is linear in total runtime): setup → probe (optional) → load arms →
preprocess + model per study (DataLoader workers overlap DICOM decode with the GPU) → write + assert submission.
With 2 GPUs (Kaggle T4 ×2, `--gpus auto`) the predict stage runs one process per GPU, each loading every arm and
taking a cost-balanced half of the studies (`predict.predict_dirs_multi`); the arm blend still runs once over all
studies, so the submission doesn't depend on the GPU count.

Local test: `python notebooks/infer.py --comp-dir <dir with test.csv, test_series.csv, test_series/> --bundle-dir <dir>
--out <dir>` (tests/test_notebook.py does this on synthetic DICOMs with a tiny checkpoint).
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

COMP_SLUG = "rsna-knee-abnormality-detection"
BUNDLE_SLUG = "knee-bundle"  # UNVERIFIED: whatever name Dom approves for the private dataset
WHEEL_PKGS = ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "python-gdcm"]
WHEEL_MODULES = {"pylibjpeg": "pylibjpeg", "pylibjpeg-libjpeg": "libjpeg", "pylibjpeg-openjpeg": "openjpeg",
                 "python-gdcm": "gdcm"}

log = logging.getLogger("infer")


def find_dir(explicit: str | None, candidates: list[str], marker: str) -> Path:
    """First candidate (glob patterns allowed, shallow) that contains `marker`."""
    if explicit:
        return Path(explicit)
    for pat in candidates:
        for d in sorted(glob.glob(pat)):
            if (Path(d) / marker).exists():
                return Path(d)
    raise FileNotFoundError(f"no directory with {marker} among {candidates}")


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:  # noqa: BLE001 - hardware info is informational only
        return f"({e})"


def install_wheels(bundle: Path) -> None:
    """pip install --no-index from the bundle, only for decoders the image lacks.

    --no-deps: `pip download` also fetches numpy (2.5.3 on 2026-09-29) while the image ships numpy 2.0.2 under torch;
    the decoders only need the image's numpy, so never let pip touch it (knee-probe run, docs/kaggle_env.md)."""
    wheels = bundle / "wheels"
    missing = [p for p in WHEEL_PKGS if importlib.util.find_spec(WHEEL_MODULES[p]) is None]
    if not missing or not wheels.is_dir():
        log.info("wheels: nothing to install (missing=%s, wheels dir=%s)", missing, wheels.is_dir())
        return
    out = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--find-links", str(wheels), *missing],
                         capture_output=True, text=True)
    log.info("wheels: pip install %s → rc %d %s", missing, out.returncode, out.stderr[-500:])


def hardware() -> dict:
    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "nproc": os.cpu_count(), "free_g": sh("free -g | head -2"), "nvidia_smi": sh("nvidia-smi -L")}
    try:
        import pydicom
        info["pydicom"] = pydicom.__version__
    except ImportError:
        info["pydicom"] = None
    for m in ["timm", *WHEEL_MODULES.values()]:
        info[f"has_{m}"] = importlib.util.find_spec(m) is not None
    return info


def probe(comp: Path, test_series, max_series: int = 400) -> dict:
    """Open questions 1 (02 §H): transfer syntaxes on the hidden test set; Fluid ≡ FatSup on test_series.csv.
    Reads one header per series (stop_before_pixels), capped, so it costs seconds, not minutes."""
    import pydicom
    out = {"series_rows": len(test_series),
           "fluid_eq_fatsup": float((test_series.Fluid_Sensitive == test_series.Fat_Suppression).mean())
           if {"Fluid_Sensitive", "Fat_Suppression"} <= set(test_series.columns) else None,
           "planes": test_series.Anatomical_Plane.value_counts().to_dict()
           if "Anatomical_Plane" in test_series.columns else None}
    syntaxes: Counter = Counter()
    for r in test_series.head(max_series).itertuples():
        d = comp / "test_series" / str(r.StudyInstanceUID) / str(r.SeriesInstanceUID)
        try:
            f = next(d.glob("*.dcm"))
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            syntaxes[str(ds.file_meta.TransferSyntaxUID.name)] += 1
        except Exception as e:  # noqa: BLE001
            syntaxes[f"error:{type(e).__name__}"] += 1
    out["transfer_syntaxes"] = dict(syntaxes)
    return out


def check_preproc(spec: dict, params_cls, version: str, code_hash: str, allow_drift: bool = False):
    """Train/infer parity (review 2026-09-30): the bundle must carry the training cache's full PreprocParams (a
    missing block or field would silently fall back to the code defaults), the same PREPROC_VERSION, and the same
    preprocessing code hash. Returns the PreprocParams to use."""
    from dataclasses import fields
    pp = spec.get("preproc")
    if not pp:
        raise RuntimeError("arms.json has no preprocessing block ('preproc'); rebuild the bundle with --cache-manifest")
    missing = [f.name for f in fields(params_cls) if f.name not in pp]
    if missing:
        raise RuntimeError(f"arms.json preprocessing block lacks {missing}; rebuild the bundle with --cache-manifest")
    if pp["version"] != version:
        raise RuntimeError(f"bundle was trained on preprocessing {pp['version']}, code is {version}")
    want = spec.get("preproc_code_hash")
    if want is None:
        log.warning("bundle has no preproc_code_hash (cache built before 2026-09-30): preprocessing code unchecked")
    elif want != code_hash:
        msg = (f"preprocessing code hash {code_hash} differs from the training cache's {want}: the test windows "
               f"may not match training. Rebuild the cache, or pass --allow-preproc-drift if the change is known "
               f"not to affect pixels")
        if not allow_drift:
            raise RuntimeError(msg)
        log.warning(msg)
    return params_cls(**pp)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comp-dir")
    ap.add_argument("--bundle-dir")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) - 1, 0))
    ap.add_argument("--probe", type=int, default=1, help="1 = run the test-data probe (first submissions)")
    ap.add_argument("--deadline-hours", type=float, default=8.0,
                    help="multi-GPU: a shard still running this long after the start is killed and its studies get "
                         "the fallback row (the notebook cap is 9 h; a lost submission scores nothing)")
    ap.add_argument("--allow-preproc-drift", action="store_true",
                    help="run even if the preprocessing code differs from the one that built the training cache")
    ap.add_argument("--gpus", default="auto",
                    help="'auto' = every visible GPU (Kaggle T4 ×2 → one process per GPU, studies split between them); "
                         "an int caps the count; 1 = the single-process path. --workers is split across the GPUs")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    T: dict[str, float] = {}
    t_all = t = time.time()

    comp = find_dir(a.comp_dir, [f"/kaggle/input/competitions/{COMP_SLUG}", f"/kaggle/input/{COMP_SLUG}"],
                    "test_series.csv")
    bundle = find_dir(a.bundle_dir, [f"/kaggle/input/datasets/*/{BUNDLE_SLUG}", f"/kaggle/input/{BUNDLE_SLUG}",
                                     "/kaggle/input/datasets/*/*", "/kaggle/input/*"], "arms.json")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log.info("competition dir %s; bundle dir %s", comp, bundle)
    install_wheels(bundle)
    if (bundle / "knee_src").is_dir():
        sys.path.insert(0, str(bundle / "knee_src"))

    import numpy as np
    import pandas as pd
    import torch

    from knee.data.preprocess import PREPROC_VERSION, PreprocParams, preproc_code_hash, series_info_from_csv
    from knee.infer.predict import load_arm, predict_dirs, predict_dirs_multi, write_submission
    from knee.paths import LABELS
    hw = hardware()
    log.info("hardware %s", json.dumps(hw))
    T["setup"] = time.time() - t

    t = time.time()
    test = pd.read_csv(comp / "test.csv")
    test_series = pd.read_csv(comp / "test_series.csv")
    expected = test.StudyInstanceUID.astype(str).tolist()
    pr = probe(comp, test_series) if a.probe else {}
    if pr:
        log.info("probe %s", json.dumps(pr))
    T["probe"] = time.time() - t

    t = time.time()
    spec = json.loads((bundle / "arms.json").read_text())
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    n_dev = n_gpu if a.gpus == "auto" else min(int(a.gpus), n_gpu)
    arm_specs = [(str(bundle / arm["ckpt"]), {**spec.get("perf", {}), **arm.get("perf", {})}) for arm in spec["arms"]]
    device = torch.device("cuda" if n_gpu else "cpu")
    # several GPUs: each shard process loads its own arms (timed inside predict); one GPU/CPU: load them here
    arms = [load_arm(ck, device, perf) for ck, perf in arm_specs] if n_dev <= 1 else None
    weights = [float(arm.get("weight", 1.0)) for arm in spec["arms"]]
    prior = np.asarray(spec.get("prior", [0.5] * len(LABELS)), dtype=np.float64)
    params = check_preproc(spec, PreprocParams, PREPROC_VERSION, preproc_code_hash(), a.allow_preproc_drift)
    T["load_arms"] = time.time() - t

    t = time.time()
    dirs = [comp / "test_series" / u for u in expected]  # from the CSV: never walk the input tree
    spf = int(spec.get("perf", {}).get("studies_per_forward", 1))
    multi: dict = {}
    if arms is not None:
        pred, timing = predict_dirs(dirs, series_info_from_csv(test_series), arms, device, params,
                                    workers=a.workers, weights=weights, how=spec.get("how", "mean"), fallback=prior,
                                    studies_per_forward=spf)
    else:
        # the series count is the only per-study cost known before reading a file (UNVERIFIED as a proxy for slices)
        cost = test_series.groupby(test_series.StudyInstanceUID.astype(str)).size().astype(float).to_dict()
        pred, timing, multi = predict_dirs_multi(
            dirs, series_info_from_csv(test_series), arm_specs, [f"cuda:{i}" for i in range(n_dev)], out / "shards",
            params, workers_per_device=-(-a.workers // n_dev), weights=weights, how=spec.get("how", "mean"),
            fallback=prior, studies_per_forward=spf, cost=cost, deadline=t_all + a.deadline_hours * 3600)
        log.info("multi-GPU %s", json.dumps(multi))
    T["predict"] = time.time() - t

    t = time.time()
    sub = write_submission(pred, expected, out / "submission.csv", prior)
    T["write"] = time.time() - t
    T["total"] = time.time() - t_all

    timing.to_csv(out / "timing_per_study.csv", index=False)
    summary = {"stages_s": {k: round(v, 2) for k, v in T.items()}, "n_studies": len(sub),
               "n_errors": int(timing.error.notna().sum()) if len(timing) else 0,
               "preprocess_s_sum": round(float(timing.preprocess_s.sum()), 2) if len(timing) else 0.0,
               "model_s_sum": round(float(timing.model_s.sum()), 2) if len(timing) else 0.0,
               "hardware": hw, "probe": pr, "multi_gpu": multi}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    log.info("summary %s", json.dumps(summary["stages_s"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
