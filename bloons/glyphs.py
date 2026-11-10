"""Reads BTD6's outlined HUD numbers (white fill, black outline) by glyph templates.

Generic OCR (Windows OCR, RapidOCR) misreads this font: "1/40" comes back as "1140" or
nothing (RESEARCH 2026-11-09). The font is fixed, so each white-fill blob is matched to
labeled masks in data/glyphs/ (<name>_<n>.png, name = a char or one of NAMES' keys).

Learn new glyphs from a crop whose text you know: `python -m bloons.glyphs learn crop.png "1/40"`.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

DIR = Path(__file__).parent / "data" / "glyphs"
SIZE = 32               # glyphs are scaled to this height, padded to a SIZE x SIZE square
MIN_SCORE = 0.6         # IoU below this reads as '?'
NAMES = {"slash": "/", "dollar": "$"}
_templates = None


def fill_mask(bgr, v_min=150, s_max=70):
    """Bright, unsaturated pixels: the white fill of the HUD font."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)).astype(np.uint8)


def segment(bgr):
    """White-fill blobs left to right, as cropped 0/1 masks. Drops specks, blobs touching the
    left/right edge (neighbouring UI) and short ones (commas, dots: < 60% of the tallest)."""
    m = fill_mask(bgr)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    w_img = m.shape[1]
    blobs = [(x, y, w, h, i) for i, (x, y, w, h, a) in enumerate(st) if i and a >= 30
             and x > 0 and x + w < w_img and w >= 4]
    if not blobs:
        return []
    tall = max(b[3] for b in blobs)
    return [(lab[y:y + h, x:x + w] == i).astype(np.uint8)
            for x, y, w, h, i in sorted(blobs) if h >= 0.6 * tall]


def norm(mask):
    """Scale to SIZE high keeping aspect, centre on a SIZE x SIZE square."""
    h, w = mask.shape
    nw = max(1, min(SIZE, round(w * SIZE / h)))
    g = cv2.resize(mask, (nw, SIZE), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((SIZE, SIZE), np.uint8)
    x0 = (SIZE - nw) // 2
    out[:, x0:x0 + nw] = g
    return out


def templates():
    global _templates
    if _templates is None:
        _templates = []
        for p in sorted(DIR.glob("*.png")):
            name = p.stem.rsplit("_", 1)[0]
            _templates.append((NAMES.get(name, name), cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0))
    return _templates


def classify(mask):
    g = norm(mask) > 0
    best, score = "?", 0.0
    for ch, t in templates():
        iou = (g & t).sum() / max(1, (g | t).sum())
        if iou > score:
            best, score = ch, iou
    return best if score >= MIN_SCORE else "?"


def read(bgr):
    """Text of an HUD crop ('' if nothing): digits, '/', '$'. Commas are dropped."""
    return "".join(classify(g) for g in segment(bgr))


def learn(bgr, text):
    """Save each glyph of a crop whose text (commas left out) is known. False if the blob
    count doesn't match the text."""
    text = text.replace(",", "")
    gl = segment(bgr)
    if len(gl) != len(text):
        return False
    DIR.mkdir(parents=True, exist_ok=True)
    inv = {v: k for k, v in NAMES.items()}
    for ch, g in zip(text, gl):
        name = inv.get(ch, ch)
        n = len(list(DIR.glob(f"{name}_*.png")))
        cv2.imwrite(str(DIR / f"{name}_{n}.png"), norm(g) * 255)
    global _templates
    _templates = None
    return True


if __name__ == "__main__":
    if sys.argv[1:2] == ["learn"]:
        print(learn(cv2.imread(sys.argv[2]), sys.argv[3]))
    else:
        print(read(cv2.imread(sys.argv[1])))
