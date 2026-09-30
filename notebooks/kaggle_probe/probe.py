"""Kaggle machine probe (private script kernel; ~5-10 min on CPU). Measures what our plan marks UNVERIFIED:
hardware, disks, image versions, how the competition data is mounted, DICOM read/decode speed from /kaggle/input,
and whether the offline decoder wheels can be built on Kaggle's Python. Writes /kaggle/working/probe.json and
/kaggle/working/wheels/*.whl (download with `kaggle kernels output dominictedrow/knee-probe -p <dir>`).

Push (only with Dom's OK): kaggle kernels push -p notebooks/kaggle_probe   [--accelerator ... for a GPU run]
Reads no labels and writes no data derived from the images except timings and header statistics.
"""
import glob
import importlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time

OUT = "/kaggle/working"
R: dict = {"t_start": time.time()}


def sh(cmd, timeout=600):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()[-4000:]
    except Exception as e:  # noqa: BLE001
        return f"({e})"


# ---- hardware
R["python"] = sys.version
R["platform"] = platform.platform()
R["cpu_count"] = os.cpu_count()
R["sched_affinity"] = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
R["cpu_model"] = sh("grep -m1 'model name' /proc/cpuinfo")
R["cgroup_cpu"] = sh("cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us")
R["meminfo"] = sh("grep -E 'MemTotal|MemAvailable' /proc/meminfo")
R["cgroup_mem"] = sh("cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes")
R["gpu"] = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1")
R["disks"] = {d: (shutil.disk_usage(d)._asdict() if os.path.exists(d) else None)
              for d in ["/kaggle/working", "/kaggle/tmp", "/tmp", "/kaggle/input", "/dev/shm"]}
R["df"] = sh("df -h 2>/dev/null | head -30")

# ---- image versions
vers = {}
for m in ["torch", "torchvision", "timm", "numpy", "pandas", "pydicom", "pylibjpeg", "libjpeg", "openjpeg", "gdcm",
          "sklearn", "pyarrow", "yaml", "cv2", "PIL", "psutil"]:
    try:
        mod = importlib.import_module(m)
        vers[m] = getattr(mod, "__version__", "present")
    except Exception as e:  # noqa: BLE001
        vers[m] = f"missing ({type(e).__name__})"
R["versions"] = vers
if vers.get("torch", "").startswith("missing") is False:
    import torch
    R["torch_cuda"] = {"available": torch.cuda.is_available(), "cuda": torch.version.cuda,
                       "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}

# ---- competition mount (shallow; never walk the whole tree)
R["input_top"] = sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*"))[:40]
comp = next((d for d in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                         "/kaggle/input/rsna-knee-abnormality-detection"] if os.path.isdir(d)), None)
R["comp_dir"] = comp
if comp:
    R["comp_listing"] = sorted(os.listdir(comp))
    tr = os.path.join(comp, "train_series")
    t0 = time.time()
    studies = sorted(os.listdir(tr)) if os.path.isdir(tr) else []
    R["n_train_study_dirs"] = len(studies)
    R["t_listdir_studies_s"] = time.time() - t0
    R["n_test_study_dirs"] = len(os.listdir(os.path.join(comp, "test_series"))) \
        if os.path.isdir(os.path.join(comp, "test_series")) else None

    # ---- DICOM read speed: 30 random series, every file (header + pixels), cold reads from /kaggle/input
    random.seed(0)
    files = []
    t0 = time.time()
    for s in random.sample(studies, min(30, len(studies))):
        sd = os.path.join(tr, s)
        ser = sorted(os.listdir(sd))
        files += sorted(glob.glob(os.path.join(sd, ser[0], "*.dcm")))
    R["t_list_30_series_s"] = time.time() - t0
    try:
        import pydicom
        nbytes, syntaxes, shapes = 0, {}, {}
        t0 = time.time()
        for f in files:
            with open(f, "rb") as fh:
                b = fh.read()
            nbytes += len(b)
        t_read = time.time() - t0
        t0 = time.time()
        errors = {}
        for f in files:
            ds = pydicom.dcmread(f)
            ts = str(ds.file_meta.TransferSyntaxUID.name)
            syntaxes[ts] = syntaxes.get(ts, 0) + 1
            try:
                a = ds.pixel_array
                shapes[str(a.shape)] = shapes.get(str(a.shape), 0) + 1
            except Exception as e:  # noqa: BLE001 - a missing decoder shows up here
                errors[f"{ts}: {type(e).__name__}"] = errors.get(f"{ts}: {type(e).__name__}", 0) + 1
        t_decode = time.time() - t0
        R["dicom_bench"] = {"files": len(files), "MB": nbytes / 1e6, "read_s": t_read,
                            "MB_per_s_cold_read": nbytes / 1e6 / max(t_read, 1e-9),
                            "decode_s_warm": t_decode, "files_per_s_decode_1proc": len(files) / max(t_decode, 1e-9),
                            "syntaxes": syntaxes, "decode_errors": errors, "shapes_top": dict(sorted(shapes.items(), key=lambda x: -x[1])[:5])}
    except Exception as e:  # noqa: BLE001
        R["dicom_bench"] = f"error: {e!r}"

    # ---- total file count estimate: count files in 200 random studies → scale
    random.seed(1)
    n = 0
    samp = random.sample(studies, min(200, len(studies)))
    t0 = time.time()
    for s in samp:
        for ser in os.listdir(os.path.join(tr, s)):
            n += len(os.listdir(os.path.join(tr, s, ser)))
    R["files_per_study_sample_mean"] = n / max(len(samp), 1)
    R["t_count_200_studies_s"] = time.time() - t0

# ---- offline wheels for the inference notebook (needs internet on)
os.makedirs(f"{OUT}/wheels", exist_ok=True)
R["pip_download"] = sh(f"{sys.executable} -m pip download pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg "
                       f"python-gdcm --only-binary=:all: --dest {OUT}/wheels 2>&1 | tail -15")
R["wheels"] = sorted(os.listdir(f"{OUT}/wheels"))

R["t_total_s"] = time.time() - R["t_start"]
with open(f"{OUT}/probe.json", "w") as fh:
    json.dump(R, fh, indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
