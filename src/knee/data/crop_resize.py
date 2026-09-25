"""Fixed-millimetre centre crop + resize to a fixed pixel grid (S02 step 5, checklist #5).

Every output pixel covers the same physical size (crop_mm / out_px), whatever the scanner's matrix and FOV. A series
whose FOV is smaller than crop_mm is padded (not stretched) so the mm scale never changes; padding is black (0 after
windowing). The intensity window is computed on the unpadded pixels only.
"""
from __future__ import annotations

import cv2
import numpy as np

CROP_MM = 130.0  # HP: 130–140 mm; 130 fits inside ~99.6–100% of series FOVs [KD]
OUT_PX = 384     # HP: store at the largest size any track needs (Track A's CoAtNet runs at 384); downsample at load


def crop_mm_native(img: np.ndarray, pixel_spacing: np.ndarray, crop_mm: float = CROP_MM) -> np.ndarray:
    """Centre crop of up to crop_mm × crop_mm in native pixels (clipped to the image). pixel_spacing = (row, col) mm."""
    h, w = img.shape
    ch = min(h, int(round(crop_mm / pixel_spacing[0])))
    cw = min(w, int(round(crop_mm / pixel_spacing[1])))
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return img[r0:r0 + ch, c0:c0 + cw]


def resize_crop(crop: np.ndarray, pixel_spacing: np.ndarray, crop_mm: float = CROP_MM,
                out_px: int = OUT_PX) -> np.ndarray:
    """Resample a native-pixel crop to the output scale (crop_mm / out_px mm per pixel). The result is at most
    out_px × out_px; it is smaller when the series' FOV is smaller than crop_mm (pad it with pad_center)."""
    out_mm_px = crop_mm / out_px
    th = min(out_px, max(1, int(round(crop.shape[0] * pixel_spacing[0] / out_mm_px))))
    tw = min(out_px, max(1, int(round(crop.shape[1] * pixel_spacing[1] / out_mm_px))))
    shrinking = th * tw < crop.shape[0] * crop.shape[1]
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(crop.astype(np.float32, copy=False), (tw, th), interpolation=interp)


def pad_center(img: np.ndarray, out_px: int = OUT_PX, pad_value: float = 0) -> np.ndarray:
    """Place img in the centre of an out_px × out_px canvas filled with pad_value (same dtype as img)."""
    th, tw = img.shape
    if th == out_px and tw == out_px:
        return img
    out = np.full((out_px, out_px), pad_value, dtype=img.dtype)
    r0, c0 = (out_px - th) // 2, (out_px - tw) // 2
    out[r0:r0 + th, c0:c0 + tw] = img
    return out
