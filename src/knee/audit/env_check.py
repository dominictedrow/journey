"""S01 step A: environment smoke test. Records hardware + software facts and runs the checks the plan depends on.

Checks: CPU (physical vs logical cores), RAM, drives, GPU/driver, torch + CUDA, compute capability (RTX 5070 =
sm_120), bf16/fp16 matmul accuracy and speed, which scaled_dot_product_attention backends run, torch.compile,
a 2-worker DataLoader (spawn on Windows), the DICOM decoders on all four transfer syntaxes, and the Kaggle CLI.

    python -m knee.audit.env_check                    # writes docs/env.md (generated block) + <audit_out>/env_check.json
    python -m knee.audit.env_check --dicom some.dcm   # use a specific uncompressed DICOM for the decoder test

Every check records ok/error instead of crashing, so one missing piece doesn't hide the rest.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from knee.paths import REPO_ROOT, load_paths

BEGIN, END = "<!-- env_check:begin -->", "<!-- env_check:end -->"


def _run(cmd: list[str], timeout: int = 60) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout + out.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"ERROR {type(e).__name__}: {e}"


def _guard(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=3)[-600:]}


def system_info() -> dict:
    import psutil

    info = {
        "platform": platform.platform(), "python": sys.version.split()[0], "executable": sys.executable,
        "cpu_physical_cores": psutil.cpu_count(logical=False), "cpu_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "in_venv": sys.prefix != sys.base_prefix, "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    if platform.system() == "Windows":
        ps = ["powershell", "-NoProfile", "-Command"]
        info["cpu_name"] = _run(ps + ["(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name)"])
        info["physical_disks"] = _run(ps + ["Get-PhysicalDisk | Select-Object FriendlyName,MediaType,BusType,"
                                            "@{n='SizeGB';e={[math]::Round($_.Size/1GB)}} | Format-Table -AutoSize | Out-String -Width 200"])
        info["volumes"] = _run(ps + ["Get-Volume | Where-Object DriveLetter | Select-Object DriveLetter,FileSystemLabel,"
                                     "FileSystem,@{n='SizeGB';e={[math]::Round($_.Size/1GB)}},"
                                     "@{n='FreeGB';e={[math]::Round($_.SizeRemaining/1GB)}} | Format-Table -AutoSize | Out-String -Width 200"])
        info["disk_to_letter"] = _run(ps + ["Get-Partition | Where-Object DriveLetter | Select-Object DriveLetter,DiskNumber "
                                            "| Format-Table -AutoSize | Out-String"])
    elif platform.system() == "Darwin":
        info["cpu_name"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    else:
        info["cpu_name"] = _run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo"])
    info["partitions"] = []
    for p in psutil.disk_partitions(all=False):
        try:
            u = psutil.disk_usage(p.mountpoint)
            info["partitions"].append({"mount": p.mountpoint, "fstype": p.fstype, "total_gb": round(u.total / 2**30),
                                       "free_gb": round(u.free / 2**30)})
        except (PermissionError, OSError):
            pass
    if shutil.which("nvidia-smi"):
        info["nvidia_smi"] = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                                   "--format=csv,noheader"])
    return info


def torch_checks() -> dict:
    import torch

    r: dict = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return r
    dev = torch.device("cuda")
    r["device_name"] = torch.cuda.get_device_name(0)
    r["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    r["arch_list"] = torch.cuda.get_arch_list()
    r["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    r["bf16_supported"] = torch.cuda.is_bf16_supported()

    def matmul(dtype):
        torch.manual_seed(0)
        a = torch.randn(4096, 4096, device=dev)
        b = torch.randn(4096, 4096, device=dev)
        ref = a @ b
        x, y = a.to(dtype), b.to(dtype)
        for _ in range(3):
            x @ y
        torch.cuda.synchronize()
        t = time.perf_counter()
        n = 20
        for _ in range(n):
            out = x @ y
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / n
        rel = ((out.float() - ref).norm() / ref.norm()).item()
        return {"ok": True, "rel_err_vs_fp32": round(rel, 5), "tflops": round(2 * 4096**3 / dt / 1e12, 1)}

    r["matmul_fp32"] = _guard(lambda: matmul(torch.float32))
    r["matmul_bf16"] = _guard(lambda: matmul(torch.bfloat16))
    r["matmul_fp16"] = _guard(lambda: matmul(torch.float16))

    def sdpa():
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v = (torch.randn(2, 8, 256, 64, device=dev, dtype=torch.bfloat16) for _ in range(3))
        ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float())
        out = {}
        for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH"):
            be = getattr(SDPBackend, name, None)
            if be is None:
                out[name] = {"ok": False, "error": "backend not in this torch"}
                continue
            try:
                with sdpa_kernel(be):
                    o = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                out[name] = {"ok": True, "max_abs_err": round((o.float() - ref).abs().max().item(), 4)}
            except Exception as e:  # noqa: BLE001
                out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0]}"[:200]}
        return out

    r["sdpa_backends"] = _guard(sdpa)

    def compile_check():
        f = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2 + 1)
        x = torch.randn(1024, device=dev)
        t = time.perf_counter()
        y = f(x)
        torch.cuda.synchronize()
        ok = torch.allclose(y, torch.nn.functional.gelu(x) * 2 + 1, atol=1e-5)
        return {"ok": bool(ok), "first_call_s": round(time.perf_counter() - t, 1)}

    r["torch_compile"] = _guard(compile_check)
    return r


class _ToyDataset:
    """Module-level (picklable) so DataLoader workers can be spawned on Windows."""

    def __len__(self):
        return 64

    def __getitem__(self, i):
        import torch

        return torch.full((3, 32, 32), float(i)), i


def dataloader_check() -> dict:
    import torch
    from torch.utils.data import DataLoader

    t = time.perf_counter()
    dl = DataLoader(_ToyDataset(), batch_size=8, num_workers=2, persistent_workers=False)
    seen = []
    for x, y in dl:
        assert torch.all(x[:, 0, 0, 0] == y.float())
        seen.extend(y.tolist())
    return {"ok": sorted(seen) == list(range(64)), "workers": 2,
            "start_method": torch.multiprocessing.get_start_method(), "seconds": round(time.perf_counter() - t, 2)}


def _synthetic_dicom(path: Path) -> Path:
    """A small uncompressed MR slice, used when no real DICOM is available (e.g. on the Mac)."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = MRImageStorage, meta.MediaStorageSOPInstanceUID, "MR"
    ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = 256, 256, 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    yy, xx = np.mgrid[:256, :256]
    ds.PixelData = ((np.sin(xx / 9.0) + np.cos(yy / 13.0) + 2) * 1000).astype(np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def decoder_check(dicom: Path | None) -> dict:
    import pydicom

    from knee.audit.decode_bench import check_variants, make_syntax_variants

    r: dict = {"pydicom": pydicom.__version__}
    for mod in ("pylibjpeg", "openjpeg", "libjpeg", "gdcm"):
        try:
            m = __import__(mod)
            r[mod] = getattr(m, "__version__", None) or getattr(m, "Version", None) and m.Version.GetVersion() or "installed"
        except Exception as e:  # noqa: BLE001
            r[mod] = f"MISSING ({type(e).__name__})"
    tmp = Path(tempfile.mkdtemp(prefix="knee_env_"))
    src = dicom if dicom else _synthetic_dicom(tmp / "synthetic.dcm")
    r["source"] = "real" if dicom else "synthetic"
    r["syntaxes"] = check_variants(make_syntax_variants(src, tmp), repeats=3)
    return r


def to_markdown(res: dict) -> str:
    s, t, d, dl = res["system"], res.get("torch", {}), res.get("decoders", {}), res.get("dataloader", {})
    L = [BEGIN, f"_Generated by `python -m knee.audit.env_check` on {res['timestamp']}. Re-run to refresh; "
         "edit notes outside this block._", "", "### Hardware", "",
         f"- CPU: {s.get('cpu_name')} — **{s.get('cpu_physical_cores')} physical cores / {s.get('cpu_logical')} logical processors**",
         f"- RAM: {s.get('ram_gb')} GB", f"- GPU (nvidia-smi): {s.get('nvidia_smi', 'n/a')}",
         f"- OS: {s.get('platform')}", ""]
    for k in ("physical_disks", "volumes", "disk_to_letter"):
        if k in s:
            L += [f"**{k}**", "```", s[k], "```", ""]
    L += ["| mount | fs | total GB | free GB |", "|---|---|---|---|"]
    L += [f"| {p['mount']} | {p['fstype']} | {p['total_gb']} | {p['free_gb']} |" for p in s.get("partitions", [])]
    L += ["", "### Python / PyTorch", "",
          f"- Python {s.get('python')} ({'venv' if s.get('in_venv') else 'conda: ' + str(s.get('conda_env'))}) at `{s.get('executable')}`",
          f"- torch {t.get('torch')} (CUDA build {t.get('cuda_build')}); cuda available: **{t.get('cuda_available')}**"]
    if t.get("cuda_available"):
        L += [f"- Device: {t.get('device_name')}, capability **sm_{str(t.get('capability')).replace('.', '')}**, "
              f"{t.get('vram_gb')} GB; arch list in wheel: {t.get('arch_list')}",
              f"- bf16 supported: {t.get('bf16_supported')}"]
        for k in ("matmul_fp32", "matmul_bf16", "matmul_fp16"):
            L.append(f"- {k}: {t.get(k)}")
        L.append("- SDPA backends (bf16, 2×8×256×64):")
        for name, v in (t.get("sdpa_backends") or {}).items():
            L.append(f"  - {name}: {'ok' if isinstance(v, dict) and v.get('ok') else v}")
        L.append(f"- torch.compile: {t.get('torch_compile')}")
    L += [f"- DataLoader 2 workers ({dl.get('start_method')}): {'ok' if dl.get('ok') else dl}", "",
          "### DICOM decoders", "",
          "- " + ", ".join(f"{k} {d.get(k)}" for k in ("pydicom", "pylibjpeg", "openjpeg", "libjpeg", "gdcm")),
          f"- Test source: {d.get('source')} DICOM, re-encoded into each syntax; decoded with each plugin, compared bit-for-bit.", "",
          "| syntax | auto | pylibjpeg | gdcm |", "|---|---|---|---|"]
    for name, v in (d.get("syntaxes") or {}).items():
        if "encode_error" in v:
            L.append(f"| {name} | encode failed: {v['encode_error'][:60]} | | |")
            continue
        cells = []
        for plug in ("auto", "pylibjpeg", "gdcm"):
            c = v.get(plug, {})
            cells.append(f"{'exact' if c.get('bit_exact') else 'MISMATCH'} {c.get('ms')} ms" if "bit_exact" in c
                         else f"n/a ({c.get('error', '')[:50]})")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "### Kaggle CLI", "", "```", res.get("kaggle_cli", "")[:800], "```", END]
    return "\n".join(L)


def write_env_md(md: str, path: Path) -> None:
    text = path.read_text() if path.exists() else "# Environment\n\nHand-written notes go outside the generated block.\n\n"
    if BEGIN in text and END in text:
        text = text[: text.index(BEGIN)] + md + text[text.index(END) + len(END):]
    else:
        text = text.rstrip() + "\n\n" + md + "\n"
    path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dicom", type=Path, help="an uncompressed DICOM to transcode (default: first train file, else synthetic)")
    ap.add_argument("--md", type=Path, default=REPO_ROOT / "docs" / "env.md")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    dicom = a.dicom
    json_out = a.json
    try:
        p = load_paths()
        json_out = json_out or p.audit_out / "env_check.json"
        if dicom is None and (p.data_root / "train_series").exists():
            dicom = next((p.data_root / "train_series").glob("*/*/*.dcm"), None)
    except KeyError:
        pass

    res = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    res["system"] = _guard(system_info)
    print("system done", flush=True)
    res["torch"] = _guard(torch_checks)
    print("torch done", flush=True)
    res["dataloader"] = _guard(dataloader_check)
    res["decoders"] = _guard(lambda: decoder_check(dicom))
    res["kaggle_cli"] = _run(["kaggle", "competitions", "list", "-s", "knee"]) if shutil.which("kaggle") else "kaggle not on PATH"

    md = to_markdown(res)
    write_env_md(md, a.md)
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(res, indent=2, default=str))
    print(md)


if __name__ == "__main__":
    main()
