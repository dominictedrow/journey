"""DICOM file access shared by the cache builder (S02) and the Kaggle inference notebook (S06).

Each file is read once (header + raw pixel bytes); pixels are decoded only for the slices we keep. Every function
is robust to bad files: it returns None and logs instead of raising, so one corrupt slice never kills a study.
"""
from __future__ import annotations

import logging
import os
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


def read_slice(path: str | os.PathLike, with_pixels: bool = True) -> SliceFile | None:
    """Read one DICOM. with_pixels=False skips the pixel bytes (fast header pass). With pixels, large elements are
    deferred (read from disk only when .pixel_array is used), so holding a whole series of datasets stays cheap.
    None if unreadable."""
    try:
        if with_pixels:
            ds = pydicom.dcmread(path, defer_size=DEFER_SIZE)
        else:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:  # noqa: BLE001 - corrupt files are logged and skipped, never fatal
        log.warning("unreadable DICOM %s: %s: %s", path, type(e).__name__, e)
        return None
    hdr = {f: _val(ds, f) for f in HEADER_FIELDS}
    return SliceFile(str(path), hdr, ds if with_pixels else None)


def list_dicoms(series_dir: str | os.PathLike) -> list[str]:
    return sorted(e.path for e in os.scandir(series_dir) if e.is_file() and e.name.lower().endswith(".dcm"))


def read_series_headers(series_dir: str | os.PathLike) -> pd.DataFrame:
    """One row per readable file: path + HEADER_FIELDS. Unreadable files are logged and dropped."""
    rows = []
    for p in list_dicoms(series_dir):
        s = read_slice(p, with_pixels=False)
        if s is not None:
            rows.append({"path": s.path, **s.header})
    return pd.DataFrame(rows, columns=["path", *HEADER_FIELDS])


def decode_pixels(ds: pydicom.Dataset, path: str = "?") -> np.ndarray | None:
    """float32 (rows, cols) after the modality LUT (RescaleSlope/Intercept). MONOCHROME1 is inverted so that
    bright always means high signal (none seen in the corpus [KD]; kept as insurance). None if it can't decode."""
    try:
        arr = ds.pixel_array
        if arr.ndim != 2:
            log.warning("unsupported pixel shape %s in %s (multi-frame/colour); slice skipped", arr.shape, path)
            return None
        img = apply_modality_lut(arr, ds).astype(np.float32, copy=False)
        if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            img = img.max() - img
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
