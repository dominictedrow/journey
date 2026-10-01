"""Per-series intensity normalization (S02 step 4, checklist #3).

MRI intensities have no physical unit and differ by scanner/coil/sequence, so each series gets its own window:
the lo/hi percentiles of its pixels map to 0/255. Percentiles are computed once over the whole series (not per
slice), so relative brightness between slices of one series is preserved.
"""
from __future__ import annotations

import numpy as np

PCT_LO, PCT_HI = 1.0, 99.0  # HP: public arms use 1–99 or 2–98 [KT]


def series_window(slices: list[np.ndarray], lo_pct: float = PCT_LO, hi_pct: float = PCT_HI,
                  max_samples: int = 2_000_000) -> tuple[float, float]:
    """(lo, hi) intensity window from the given float slices. Deterministic subsampling keeps it fast: every k-th
    pixel of the concatenated series. Each slice contributes its share of that stride directly, so the full series is
    never concatenated in float32 (~190 MB for 320 slices at 384 px; review 2026-09-30). Same pixels as
    concatenate-then-stride, so the same window."""
    slices = [s for s in slices if s is not None and s.size]
    total = sum(s.size for s in slices)
    if total == 0:
        return 0.0, 1.0
    k = int(np.ceil(total / max_samples)) if total > max_samples else 1
    parts, off = [], 0
    for s in slices:
        parts.append(s.ravel()[(-off) % k::k])  # global indices 0, k, 2k, ... that fall in this slice
        off += s.size
    flat = np.concatenate(parts)
    lo, hi = np.percentile(flat, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(flat)), float(np.max(flat))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def to_uint8(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = (img.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)
