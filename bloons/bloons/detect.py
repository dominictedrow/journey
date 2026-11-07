"""Bloon detector (PLAN 5.2): background difference inside the path band + HSV colour classes.
Portable, no OS calls.

1. Work on the content frame resized to WORK_W px wide (like vision.py).
2. A bloon pixel differs from the empty-track background by more than DIFF_T (max channel),
   lies in the sprite band (the path polyline shifted up by the map's sprite_dy, +- BAND_UNITS)
   and isn't under one of our towers (their sprites turn to aim). Pixels that move between
   background frames (swaying grass tufts) need their own jitter + NOISE_MARGIN instead.
3. Close + open, connected components, blobs over MIN_AREA (in 2x px, so the numbers match
   RESEARCH) are kept: falling leaves are far smaller than any bloon.
4. Each blob gets an HSV feature vector (features()) and the nearest class centroid from
   data/bloon_classes.json (tools/train_bloons.py). Non-bloon classes (Halloween bats on Monkey
   Meadow) are dropped. count = blob area / that type's single-bloon area, so a bunch of
   overlapping bloons counts as several.

Positions are track positions (the blob centroid moved down by sprite_dy), normalized.
The background is the map's empty track, stored as data (tools/bloon_bg.py, from Sandbox:
Standard games animate entrance arrows / exit hearts on the track before round 1, which
would read as bloons once they vanish). Our towers are masked out at run time.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .placement import PathKeepout

DATA = Path(__file__).parent / "data"
WORK_W = 735
DIFF_T = 30               # max-channel |frame - background| for a bloon pixel
BAND_UNITS = 9.0          # sprite band half-width around the shifted path (a bloon is ~5-6 units across... up to 6.5 for ceramics)
MIN_AREA = 1000           # 2x px: black / white bloons ~2,200-2,700, a leaf < 600
NOISE_DILATE = 3          # the background jitter map is spread this many px (work size)
NOISE_MARGIN = 25         # a jittery pixel (swaying grass) needs its own jitter + this to count
FRAGMENT = 0.75          # a blob under this x its class's single area may be part of a cut bloon
FRAGMENT_REACH = 12.0    # units: how far apart the parts of one cut bloon can be (centroids)
MERGE_AT = 1.6           # a blob counts as several bloons from this many times a single one's area
DECOMPOSE_DIST = 1.5     # a blob this far from its centroid (or past MERGE_AT x a single area) is
                         # split by per-pixel colour (merged streams of mixed bloons)
LEFTOVER = 0.6           # after merging fragments, a blob under this x a single bloon is dropped
COLOUR_CLASSES = ("red", "blue", "green", "yellow", "pink", "purple", "ceramic")
FULL_W = 2940             # content width at 2x: areas are reported in these px


_BANDS = {}                 # (map path, work shape) -> band mask: built once per process


@dataclass(frozen=True)
class Bloon:
    x: float                # track position, normalized
    y: float
    kind: str               # 'red', 'blue', ..., 'ceramic'
    count: int              # bloons in this blob (overlapping bloons merge)
    area: int               # 2x px
    dist: float             # distance to the class centroid (lower = surer)


def work(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (WORK_W, round(h * WORK_W / w)), interpolation=cv2.INTER_AREA)


def features(bgr, mask, area):
    """HSV feature vector of the pixels under `mask`: an 18-bin hue histogram weighted by
    saturation (x3), then the fractions dark / white / grey / saturated, mean S, mean V, and
    log(area / 3000)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0][mask].astype(float)
    s = hsv[..., 1][mask] / 255.0
    v = hsv[..., 2][mask] / 255.0
    n = max(len(h), 1)
    hist = np.bincount((h // 10).astype(int), weights=s, minlength=18)[:18] / n
    return np.concatenate([hist * 3, [np.mean(v < 0.25), np.mean((s < 0.18) & (v > 0.75)),
                                      np.mean((s < 0.18) & (v >= 0.25) & (v <= 0.75)), np.mean(s > 0.5),
                                      np.mean(s), np.mean(v), np.log(max(area, 1) / 3000)]])


def pixel_bins(bgr, mask):
    """Hue x saturation bin (18 x 4 = 72) of each coloured pixel under `mask` (S >= 0.35,
    V >= 0.3), and those pixels' (row, col). Grey / dark / white pixels are left out."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sel = mask & (hsv[..., 1] >= 90) & (hsv[..., 2] >= 77)
    rr, cc = np.nonzero(sel)
    h, sat = hsv[rr, cc, 0].astype(int), hsv[rr, cc, 1].astype(int)
    return np.minimum(h // 10, 17) * 4 + np.minimum(sat // 64, 3), rr, cc


def load_classes(path=None):
    d = json.loads(Path(path or DATA / "bloon_classes.json").read_text())
    return d


class BloonDetector:
    def __init__(self, map_json, classes=None):
        self.classes = classes or load_classes()
        self.names = list(self.classes["centroids"])
        self.C = np.array([self.classes["centroids"][k] for k in self.names])
        self.non_bloon = set(self.classes.get("non_bloon", []))
        self.single = self.classes["single_area"]
        ph = self.classes.get("pixel_hist") or {}
        self.pix_names = [k for k in COLOUR_CLASSES if k in ph]
        self.pix_logp = np.log(np.array([ph[k] for k in self.pix_names]) + 1e-4) if self.pix_names else None
        self.pix_area = self.classes.get("pixel_area") or {}
        self.unit = tuple(map_json["scale"]["unit_norm"])
        p = map_json["path"]
        self.dy = p.get("sprite_dy_units", 0.0) * self.unit[1]
        poly = np.array(p["polyline"], float)
        poly[:, 1] -= self.dy
        self._ko = PathKeepout(self.unit, polyline=poly, half_width=0.0)
        self._key = (poly.tobytes(), self.unit)
        self.band = None            # bool (H, W) at work size, built on the first frame
        self.bg = None
        self.noise = None
        self.towers = ()            # [(x, y, radius_units)] to leave out

    def _shape(self, img):
        if self.band is None or self.band.shape != img.shape[:2]:
            key = (self._key, img.shape[:2])
            if key not in _BANDS:
                H, W = img.shape[:2]
                # distances on a 4x coarser grid, interpolated up (the field is smooth)
                h, w = -(-H // 4) + 1, -(-W // 4) + 1
                yy, xx = np.mgrid[0:h, 0:w] * 4.0
                d = self._ko.distances((xx + 0.5) / W, (yy + 0.5) / H).reshape(h, w).astype(np.float32)
                full = cv2.resize(d, ((w - 1) * 4 + 1, (h - 1) * 4 + 1), interpolation=cv2.INTER_LINEAR)
                _BANDS[key] = full[:H, :W] < BAND_UNITS
            self.band = _BANDS[key]

    def set_background(self, frames):
        """Content frames (BGR, any size) of the track with no bloons on it (and no track
        markers: the arrows / hearts a Standard game animates before round 1 are not empty
        track; use Sandbox)."""
        bg = background_from_frames(frames)
        return False if bg is None else self.set_background_arrays(*bg)

    def set_background_arrays(self, bg, dev):
        """A stored background (work-size BGR median + per-pixel jitter, background_from_frames)."""
        self._shape(bg)
        self.bg = bg.astype(np.int16)
        dev = cv2.dilate(dev.astype(np.uint8), np.ones((NOISE_DILATE, NOISE_DILATE), np.uint8)).astype(np.int16)
        self.noise = np.maximum(DIFF_T, dev + NOISE_MARGIN)    # per-pixel diff threshold
        return True

    def set_towers(self, towers):
        """Our towers as [(x, y, radius_units)]: their sprites are left out of the band."""
        self.towers = tuple(towers)

    def _tower_mask(self, shape):
        m = np.zeros(shape, bool)
        if not self.towers:
            return m
        H, W = shape
        yy, xx = np.mgrid[0:H, 0:W]
        for x, y, r in self.towers:
            m |= np.hypot(((xx + 0.5) / W - x) / self.unit[0], ((yy + 0.5) / H - y) / self.unit[1]) < r
        return m

    def pixels(self, frame):
        """Bloon-pixel mask at work size (after the band / noise / tower masks), or None."""
        if self.bg is None or frame is None:
            return None, None
        img = work(frame)
        self._shape(img)
        if img.shape[:2] != self.bg.shape[:2]:
            return None, None
        px = (np.abs(img.astype(np.int16) - self.bg).max(2) > self.noise) & self.band
        if self.towers:
            px &= ~self._tower_mask(px.shape)
        return img, px

    def detect(self, frame):
        """[Bloon] in a content frame (BGR, any size). [] without a background."""
        img, px = self.pixels(frame)
        if img is None:
            return []
        d = cv2.morphologyEx(px.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(d)
        H, W = px.shape
        s = (W / FULL_W) ** 2
        raw = []                                   # [x, y, kind, area, dist, count or None]
        for i in range(1, n):
            area = st[i, 4] / s
            if area < MIN_AREA:
                continue
            m = (lab == i) & px
            if not m.any():
                continue
            f = features(img, m, area)
            dist = ((self.C - f) ** 2).sum(1)
            k = int(np.argmin(dist))
            kind, dk = self.names[k], float(np.sqrt(dist[k]))
            if kind in self.non_bloon:
                continue
            if dk > DECOMPOSE_DIST or area > MERGE_AT * self.single.get(kind, area):
                parts = self.decompose(img, m, s)
                if parts:
                    raw += [[x, y, c, a, dk, cnt] for x, y, c, a, cnt in parts]
                    continue
            raw.append([cen[i][0] / W, cen[i][1] / H, kind, area, dk, None])
        raw = self._merge_fragments(raw)
        raw = [b for b in raw if b[5] is not None or b[3] >= LEFTOVER * self.single.get(b[2], b[3])]
        return [Bloon(round(float(x), 4), round(float(y + self.dy), 4), kind,
                      cnt if cnt is not None else blob_count(area, self.single.get(kind, area)),
                      int(area), round(float(dist), 3)) for x, y, kind, area, dist, cnt in raw]

    def assign_pixels(self, img, m):
        """Each coloured pixel under `m` -> its most likely colour class (pixel_hist).
        -> (class index per pixel, rows, cols), or None without a pixel model."""
        if self.pix_logp is None:
            return None
        b, rr, cc = pixel_bins(img, m)
        if not len(b):
            return None
        return np.argmax(self.pix_logp[:, b], 0), rr, cc

    def decompose(self, img, m, s):
        """Split a blob by per-pixel colour: [(x, y, class, area, count)] for each colour class
        with at least half a bloon's worth of pixels (pixel_area, 2x px). [] if nothing fits
        (e.g. a blob of grey / white / black bloons)."""
        r = self.assign_pixels(img, m)
        if r is None:
            return []
        cls, rr, cc = r
        H, W = m.shape
        out = []
        for j, name in enumerate(self.pix_names):
            sel = cls == j
            area = sel.sum() / s
            single = self.pix_area.get(name)
            if not single or area < 0.5 * single:
                continue
            out.append((cc[sel].mean() / W, rr[sel].mean() / H, name, area, max(1, int(round(area / single)))))
        return out

    def _merge_fragments(self, raw):
        """A bloon cut in two (by masked grass tufts) gives a fragment below FRAGMENT x its
        single area next to the rest. Merge each fragment into the nearest blob within
        FRAGMENT_REACH units when the two together still fit one bloon; the larger part
        keeps its class. Smallest fragments first."""
        done = [b for b in raw if b[5] is not None]          # decomposed parts: already counted
        raw = sorted((b for b in raw if b[5] is None), key=lambda b: b[3])
        i = 0
        while i < len(raw):
            x, y, kind, area, dist, _ = raw[i]
            single = self.single.get(kind, area)
            best = None
            if area < FRAGMENT * single:
                for j, o in enumerate(raw):
                    if j == i:
                        continue
                    d = np.hypot((o[0] - x) / self.unit[0], (o[1] - y) / self.unit[1])
                    big = o if o[3] >= area else raw[i]
                    if d < FRAGMENT_REACH and area + o[3] <= MERGE_AT * self.single.get(big[2], big[3]) \
                            and (best is None or d < best[0]):
                        best = (d, j)
            if best is None:
                i += 1
                continue
            o = raw[best[1]]
            big = o if o[3] >= area else raw[i]
            tot = area + o[3]
            merged = [(x * area + o[0] * o[3]) / tot, (y * area + o[1] * o[3]) / tot, big[2], tot, big[4], None]
            raw = [b for k, b in enumerate(raw) if k not in (i, best[1])] + [merged]
            raw.sort(key=lambda b: b[3])
            i = 0
        return raw + done


def background_from_frames(frames):
    """(median BGR uint8, jitter uint8) at work size from empty-track frames, or None."""
    ws = [work(f).astype(np.int16) for f in frames if f is not None]
    if not ws:
        return None
    bg = np.median(np.stack(ws), 0).astype(np.int16)
    dev = np.max([np.abs(w - bg).max(2) for w in ws], 0)
    return bg.astype(np.uint8), np.clip(dev, 0, 255).astype(np.uint8)


def bg_paths(map_name):
    d = DATA / "maps"
    return d / f"{map_name}.bg.png", d / f"{map_name}.bgnoise.png", d / f"{map_name}.bg.json"


def save_background(map_name, bg, dev, meta):
    """Store a map's empty-track background (data/maps/<map>.bg.png + .bgnoise.png + .bg.json)."""
    p_bg, p_dev, p_meta = bg_paths(map_name)
    cv2.imwrite(str(p_bg), bg, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    cv2.imwrite(str(p_dev), dev, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    p_meta.write_text(json.dumps(meta, indent=1) + "\n")


def load_background(map_name):
    """(bg, dev) stored for a map, or None."""
    p_bg, p_dev, _ = bg_paths(map_name)
    if not (p_bg.exists() and p_dev.exists()):
        return None
    return cv2.imread(str(p_bg)), cv2.imread(str(p_dev), cv2.IMREAD_GRAYSCALE)


def blob_count(area, single):
    """Bloons in a blob of `area`: 1 up to MERGE_AT x a single bloon's area (a lone bloon's
    blob grows a little with outline / shadow), then rounded."""
    r = area / single
    return 1 if r < MERGE_AT else int(round(r))


def summarize(bloons):
    """{kind: count} over a detect() result."""
    out = {}
    for b in bloons:
        out[b.kind] = out.get(b.kind, 0) + b.count
    return out
