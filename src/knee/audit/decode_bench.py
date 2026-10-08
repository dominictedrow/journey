"""S01 step B6: transfer-syntax coverage and decode speed.

Two measurements:
1. **Synthetic syntaxes:** take one real file and re-encode it as Implicit VR LE, JPEG Lossless (Process 14 SV1,
   via gdcm), JPEG 2000 lossless and RLE (via pydicom), then decode each with every available plugin and check the
   result is bit-identical to the original. This proves the decoders work, even though the train data seems to be
   all uncompressed (then the real corpus can't exercise them).
2. **Real corpus throughput:** decode + modality LUT + float32 on random files from the header scan, with
   1..N worker processes. Each configuration reads a *different* random set of files, so the OS page cache
   doesn't flatter later runs (the first touch of a file is a real disk read).

    python -m knee.audit.decode_bench --split train --files-per-config 600 --workers 1 4 8 16
Output: <audit_out>/decode_bench.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_modality_lut, pixel_array
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian, JPEG2000Lossless, RLELossless

from knee.paths import load_paths

PLUGINS = ["", "pylibjpeg", "gdcm"]  # "" = pydicom's automatic choice


def make_syntax_variants(src: Path, out_dir: Path) -> dict[str, dict]:
    """Re-encode one uncompressed DICOM into the other syntaxes. Returns {name: {"path"|"error": ...}}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, dict] = {"explicit_vr_le": {"path": str(src)}}

    try:
        d = pydicom.dcmread(src)
        d.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        p = out_dir / "implicit_vr_le.dcm"
        d.save_as(p, implicit_vr=True, little_endian=True)
        res["implicit_vr_le"] = {"path": str(p)}
    except Exception as e:  # noqa: BLE001
        res["implicit_vr_le"] = {"error": f"{type(e).__name__}: {e}"[:200]}

    for name, uid in (("jpeg2000_lossless", JPEG2000Lossless), ("rle_lossless", RLELossless)):
        try:
            d = pydicom.dcmread(src)
            d.compress(uid)
            p = out_dir / f"{name}.dcm"
            d.save_as(p)
            res[name] = {"path": str(p)}
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": f"{type(e).__name__}: {e}"[:200]}

    # pydicom has no JPEG Lossless encoder; gdcm does.
    try:
        import gdcm

        r = gdcm.ImageReader()
        r.SetFileName(str(src))
        if not r.Read():
            raise RuntimeError("gdcm could not read the source")
        ch = gdcm.ImageChangeTransferSyntax()
        ch.SetTransferSyntax(gdcm.TransferSyntax(gdcm.TransferSyntax.JPEGLosslessProcess14_1))
        ch.SetInput(r.GetImage())
        if not ch.Change():
            raise RuntimeError("gdcm transcode failed")
        p = out_dir / "jpeg_lossless_sv1.dcm"
        w = gdcm.ImageWriter()
        w.SetFileName(str(p))
        w.SetFile(r.GetFile())
        w.SetImage(ch.GetOutput())
        if not w.Write():
            raise RuntimeError("gdcm write failed")
        res["jpeg_lossless_sv1"] = {"path": str(p)}
    except Exception as e:  # noqa: BLE001
        res["jpeg_lossless_sv1"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return res


def check_variants(variants: dict[str, dict], repeats: int = 5) -> dict:
    """Decode each variant with each plugin: bit-exact vs the original? median ms per slice (warm)."""
    ref = pixel_array(variants["explicit_vr_le"]["path"])
    out = {}
    for name, v in variants.items():
        if "path" not in v:
            out[name] = {"encode_error": v["error"]}
            continue
        ts = str(pydicom.dcmread(v["path"], stop_before_pixels=True).file_meta.TransferSyntaxUID)
        out[name] = {"transfer_syntax": ts}
        for plug in PLUGINS:
            key = plug or "auto"
            try:
                times = []
                for _ in range(repeats):
                    t = time.perf_counter()
                    a = pixel_array(v["path"], decoding_plugin=plug) if plug else pixel_array(v["path"])
                    times.append(time.perf_counter() - t)
                out[name][key] = {"bit_exact": bool(np.array_equal(a, ref)), "ms": round(1000 * float(np.median(times)), 3)}
            except Exception as e:  # noqa: BLE001
                out[name][key] = {"error": f"{type(e).__name__}: {str(e).splitlines()[0]}"[:160]}
    return out


def _noop(_: int) -> int:
    return os.getpid()


def _decode_one(path: str) -> tuple[int, str | None]:
    """The work a cache builder does per slice: read, decode, rescale, float32. Returns bytes read."""
    try:
        ds = pydicom.dcmread(path)
        arr = apply_modality_lut(ds.pixel_array, ds).astype(np.float32, copy=False)
        return os.path.getsize(path), None if arr.size else "empty"
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"[:160]


def bench_corpus(root: Path, files: list[str], workers: list[int], per_config: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    pool = list(rng.permutation(files))
    results = []
    for n in workers:
        batch, pool = pool[:per_config], pool[per_config:]
        if len(batch) < per_config:
            print(f"not enough unseen files for workers={n}; stopping")
            break
        paths = [str(root / f) for f in batch]
        if n == 1:
            t = time.perf_counter()
            out = [_decode_one(p) for p in paths]
            dt = time.perf_counter() - t
        else:
            with ProcessPoolExecutor(max_workers=n) as ex:
                list(ex.map(_noop, range(n * 4)))  # start the workers before timing (spawn is slow on Windows)
                t = time.perf_counter()
                out = list(ex.map(_decode_one, paths, chunksize=8))
                dt = time.perf_counter() - t
        nbytes = sum(b for b, _ in out)
        errs = [e for _, e in out if e]
        r = {"workers": n, "files": len(paths), "seconds": round(dt, 3), "slices_per_s": round(len(paths) / dt, 1),
             "MB_per_s": round(nbytes / dt / 1e6, 1), "ms_per_slice_per_worker": round(1000 * dt * n / len(paths), 2),
             "errors": len(errs), "error_examples": errs[:3]}
        print(r, flush=True)
        results.append(r)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--files-per-config", type=int, default=600)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    a = ap.parse_args()
    if a.data_root is None or a.out is None:
        p = load_paths()
        a.data_root = a.data_root or p.data_root
        a.out = a.out or p.audit_out

    scan = pd.concat([pd.read_parquet(f, columns=["path", "transfer_syntax", "error"])
                      for f in glob.glob(str(a.out / f"header_scan_{a.split}" / "part-*.parquet"))])
    scan = scan[scan.error.isna()]
    root = a.data_root / f"{a.split}_series"
    res = {"platform": platform.platform(), "cpu_count_logical": os.cpu_count(),
           "pydicom": pydicom.__version__, "split": a.split,
           "transfer_syntax_counts": {k: int(v) for k, v in scan.transfer_syntax.value_counts().items()}}

    src = root / scan[scan.transfer_syntax == str(ExplicitVRLittleEndian)].path.iloc[0]
    variants = make_syntax_variants(src, a.out / "syntax_variants")
    res["synthetic_syntaxes"] = check_variants(variants)
    print(json.dumps(res["synthetic_syntaxes"], indent=1))

    # Per-syntax warm decode speed on real files (only syntaxes that actually occur).
    per_ts = {}
    for ts, g in scan.groupby("transfer_syntax"):
        paths = [str(root / f) for f in g.path.sample(min(50, len(g)), random_state=0)]
        for p in paths:  # warm the cache so this measures CPU decode, not disk
            _decode_one(p)
        t = time.perf_counter()
        for p in paths:
            _decode_one(p)
        per_ts[ts] = {"n": len(paths), "ms_per_slice_warm": round(1000 * (time.perf_counter() - t) / len(paths), 2)}
    res["real_warm_decode_by_syntax"] = per_ts

    res["real_cold_throughput"] = bench_corpus(root, scan.path.tolist(), a.workers, a.files_per_config)
    (a.out / "decode_bench.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
