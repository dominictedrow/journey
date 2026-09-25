"""DICOM patient-space geometry: slice ordering, plane, in-plane axes, laterality.

Pure numpy, no file I/O, so the audit (S01), the cache builder (S02) and the Kaggle inference notebook
all use exactly the same rules.

Conventions (DICOM PS3.3 C.7.6.2): patient coordinates are LPS, i.e. +x = patient Left, +y = Posterior,
+z = Superior (head). ImageOrientationPatient (IOP) = [row direction (3), column direction (3)]: the first
triplet is the direction in which the column index increases along a row (left→right on screen), the second
is the direction in which the row index increases down a column (top→bottom). PixelSpacing = [spacing
between rows (vertical), spacing between columns (horizontal)] in mm.
"""
from __future__ import annotations

import numpy as np

_AXIS_LETTERS = {0: ("L", "R"), 1: ("P", "A"), 2: ("S", "I")}  # (+, -) for x, y, z in LPS
_PLANE_BY_NORMAL_AXIS = {0: "Sagittal", 1: "Coronal", 2: "Axial"}


def as_vec(value, n: int) -> np.ndarray | None:
    """Parse a multi-valued DICOM number (list, tuple, or 'a\\b\\c' string) into a float array of length n."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace(",", "\\").split("\\") if p.strip()]
    else:
        try:
            parts = list(value)
        except TypeError:
            return None
    if len(parts) != n:
        return None
    try:
        v = np.asarray([float(p) for p in parts], dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return v if np.all(np.isfinite(v)) else None


def slice_normal(iop: np.ndarray) -> np.ndarray:
    """Unit normal of the image plane: row_dir × col_dir."""
    n = np.cross(iop[:3], iop[3:6])
    norm = np.linalg.norm(n)
    return n / norm if norm > 0 else n


def slice_positions(ipp: np.ndarray, iop: np.ndarray) -> np.ndarray:
    """Position of each slice along the plane normal (mm). ipp: (N, 3). Sort by this, never by filename."""
    return np.asarray(ipp, dtype=np.float64) @ slice_normal(iop)


def order_slices(ipp: np.ndarray, iop: np.ndarray) -> np.ndarray:
    """Indices that sort slices by physical position along the normal (ascending)."""
    return np.argsort(slice_positions(ipp, iop), kind="stable")


def axis_letter(vec: np.ndarray) -> str:
    """Dominant LPS direction of a unit vector, e.g. [0.99, 0.01, -0.1] -> '+L' (patient left)."""
    i = int(np.argmax(np.abs(vec)))
    pos, neg = _AXIS_LETTERS[i]
    return pos if vec[i] >= 0 else neg


def plane_from_iop(iop: np.ndarray) -> str:
    """Anatomical plane implied by the orientation: normal along x → sagittal, y → coronal, z → axial."""
    n = slice_normal(iop)
    return _PLANE_BY_NORMAL_AXIS[int(np.argmax(np.abs(n)))]


def obliquity_deg(iop: np.ndarray) -> float:
    """Angle (degrees) between the plane normal and its nearest patient axis. 0 = perfectly orthogonal plane."""
    n = np.abs(slice_normal(iop))
    return float(np.degrees(np.arccos(np.clip(n.max(), -1.0, 1.0))))


def image_center(ipp: np.ndarray, iop: np.ndarray, rows: int, cols: int, pixel_spacing: np.ndarray) -> np.ndarray:
    """Patient-space position (mm) of the centre of one slice. ipp = position of the first (top-left) pixel."""
    row_dir, col_dir = iop[:3], iop[3:6]
    return (np.asarray(ipp, dtype=np.float64)
            + row_dir * (cols - 1) / 2.0 * pixel_spacing[1]
            + col_dir * (rows - 1) / 2.0 * pixel_spacing[0])


def laterality_from_center_x(center_x: float, dead_zone_mm: float = 0.0) -> str:
    """Left/right knee from where the imaged volume sits relative to the scanner isocentre.

    In LPS, x < 0 is the patient's right. A knee coil is placed off-centre on the scanned side, so a right knee
    tends to have x < 0. This is a heuristic ([101] reports 92.9% agreement with the Laterality tag; S01
    re-measures it) and it assumes the table/isocentre is near the patient midline. UNVERIFIED as a rule;
    use only where the tag is missing, and report the measured agreement.
    """
    if center_x is None or not np.isfinite(center_x) or abs(center_x) <= dead_zone_mm:
        return "U"
    return "R" if center_x < 0 else "L"


def spacing_stats(positions: np.ndarray) -> dict:
    """Spacing between consecutive sorted slice positions: median, relative spread, duplicates, gaps."""
    p = np.sort(np.asarray(positions, dtype=np.float64))
    if p.size < 2:
        return {"n": int(p.size), "median_mm": np.nan, "cv": np.nan, "n_duplicate": 0, "max_gap_ratio": np.nan}
    d = np.diff(p)
    dup = int(np.sum(d < 1e-3))
    d_pos = d[d >= 1e-3]
    med = float(np.median(d_pos)) if d_pos.size else 0.0
    return {
        "n": int(p.size),
        "median_mm": med,
        "cv": float(np.std(d_pos) / med) if med > 0 else np.nan,
        "n_duplicate": dup,  # >0 means several stacks / echoes share positions (must be split before stacking)
        "max_gap_ratio": float(d_pos.max() / med) if med > 0 else np.nan,  # >1.5 hints at a missing slice
    }
