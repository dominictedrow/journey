"""DICOM file access shared by the cache builder (S02) and the Kaggle inference notebook (S06).

Each file is read once (header + raw pixel bytes); pixels are decoded only for the slices we keep. Every function
is robust to bad files: it returns None and logs instead of raising, so one corrupt slice never kills a study.

I/O concurrency (docs/plan/03_performance_plan.md, A1): `read_slices` reads a whole series with a per-process thread
pool. One reader is latency-bound on Kaggle's NFS mount (≈ 62 files/s at 1 worker vs 325 at 16, docs/kaggle_env.md §4),
and file reads release the GIL, so threads overlap the waits even inside a single DataLoader worker. With pixels, each
file is read in one go (whole bytes, parsed from memory) instead of header now + a second open for the deferred pixel
bytes later. The parsed result is the same as `read_slice`. KNEE_IO_THREADS sets the pool size (default 16; ≤ 1 = serial).
"""
from __future__ import annotations

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.pixels import apply_modality_lut

from knee.data import geometry as G

log = logging.getLogger(__name__)

# Header fields the pipeline needs per slice. Multi-valued ones are parsed with geometry.as_vec where used.
HEADER_FIELDS = [
    "SOPInstanceUID", "InstanceNumber", "AcquisitionNumber", "EchoNumbers", "ImagePositionPatient",
    "ImageOrientationPatient", "PixelSpacing", "SliceThickness", "Rows", "Columns", "NumberOfFrames",
    "PhotometricInterpretation", "Manufacturer", "ManufacturerModelName", "SoftwareVersions", "MagneticFieldStrength",
    "SeriesDescription", "SeriesNumber", "Laterality", "RepetitionTime", "EchoTime", "InversionTime",
    "ScanningSequence", "ScanOptions", "MRAcquisitionType", "ImageType",
]
DEFER_SIZE = "64 KB"
IO_THREADS = int(os.environ.get("KNEE_IO_THREADS", "16"))

_pool: ThreadPoolExecutor | None = None
_pool_pid: int | None = None


def _io_pool() -> ThreadPoolExecutor | None:
    """One pool per process (a forked DataLoader worker never reuses its parent's pool threads)."""
    global _pool, _pool_pid
    if IO_THREADS <= 1:
        return None
    if _pool is None or _pool_pid != os.getpid():
        _pool, _pool_pid = ThreadPoolExecutor(IO_THREADS, thread_name_prefix="dicom-io"), os.getpid()
    return _pool


def _val(ds, name):
    if name not in ds:
        return None
    v = ds[name].value
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        return "\\".join(str(x) for x in v)
    return None if v is None or v == "" else v


@dataclass
class SliceFile:
    path: str
    header: dict
    ds: pydicom.Dataset | None = field(default=None, repr=False)  # kept only while the series is processed


def read_slice(path: str | os.PathLike, with_pixels: bool = True, whole_file: bool = False) -> SliceFile | None:
    """Read one DICOM. with_pixels=False skips the pixel bytes (fast header pass). With pixels, large elements are
    deferred (read from disk only when .pixel_array is used), so holding a whole series of datasets stays cheap;
    whole_file=True instead reads the file in one go and parses it from memory (one open, used by read_slices).
    None if unreadable."""
    try:
        if with_pixels and whole_file:
            with open(path, "rb") as fh:
                ds = pydicom.dcmread(io.BytesIO(fh.read()))
        elif with_pixels:
            ds = pydicom.dcmread(path, defer_size=DEFER_SIZE)
        else:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:  # noqa: BLE001 - corrupt files are logged and skipped, never fatal
        log.warning("unreadable DICOM %s: %s: %s", path, type(e).__name__, e)
        return None
    hdr = {f: _val(ds, f) for f in HEADER_FIELDS}
    return SliceFile(str(path), hdr, ds if with_pixels else None)


def read_slices(paths: list[str], with_pixels: bool = True) -> list[SliceFile | None]:
    """read_slice over many files with overlapping reads (see the module docstring). Same order as `paths`."""
    pool = _io_pool()
    if pool is None or len(paths) < 2:
        return [read_slice(p, with_pixels, whole_file=with_pixels) for p in paths]
    return list(pool.map(lambda p: read_slice(p, with_pixels, whole_file=with_pixels), paths))


def list_dicoms(series_dir: str | os.PathLike) -> list[str]:
    return sorted(e.path for e in os.scandir(series_dir) if e.is_file() and e.name.lower().endswith(".dcm"))


def read_series_headers(series_dir: str | os.PathLike) -> pd.DataFrame:
    """One row per readable file: path + HEADER_FIELDS. Unreadable files are logged and dropped."""
    rows = []
    for s in read_slices(list_dicoms(series_dir), with_pixels=False):
        if s is not None:
            rows.append({"path": s.path, **s.header})
    return pd.DataFrame(rows, columns=["path", *HEADER_FIELDS])


def decode_pixels(ds: pydicom.Dataset, path: str = "?") -> np.ndarray | None:
    """float32 (rows, cols) after the modality LUT (RescaleSlope/Intercept). MONOCHROME1 is negated so that bright
    always means high signal (none seen in the corpus [KD]; kept as insurance). Negation, not max − x: a per-slice
    max would shift each slice differently, and the per-series window takes care of the offset. None if it can't
    decode.

    A linear rescale is applied in float32: pydicom's apply_modality_lut returns float64 (2× the memory of the slice
    for every Philips slice; review 2026-09-30). A real Modality LUT Sequence still goes through pydicom."""
    try:
        arr = ds.pixel_array
        if arr.ndim != 2:
            log.warning("unsupported pixel shape %s in %s (multi-frame/colour); slice skipped", arr.shape, path)
            return None
        if "ModalityLUTSequence" in ds:
            img = apply_modality_lut(arr, ds).astype(np.float32, copy=False)
        else:
            img = arr.astype(np.float32)
            slope, icpt = ds.get("RescaleSlope"), ds.get("RescaleIntercept")
            slope = 1.0 if slope is None else float(slope)
            icpt = 0.0 if icpt is None else float(icpt)
            if slope != 1.0:
                img *= np.float32(slope)
            if icpt != 0.0:
                img += np.float32(icpt)
        if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            img = -img
        return img
    except Exception as e:  # noqa: BLE001
        log.warning("pixel decode failed %s: %s: %s", path, type(e).__name__, e)
        return None


def slice_geometry(hdr: dict) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """(ipp[3], iop[6], pixel_spacing[2]) parsed from a header dict; any may be None."""
    return (G.as_vec(hdr.get("ImagePositionPatient"), 3), G.as_vec(hdr.get("ImageOrientationPatient"), 6),
            G.as_vec(hdr.get("PixelSpacing"), 2))


def study_series_dirs(study_dir: str | os.PathLike) -> dict[str, Path]:
    """SeriesInstanceUID (folder name) → folder, for one study folder."""
    return {e.name: Path(e.path) for e in os.scandir(study_dir) if e.is_dir()}
