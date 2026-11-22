"""Bloon path from motion (portable; PLAN 3.3). Feed frames of round 1 on an empty map;
every new bloon pixel is a bloon passing. That gives:
- a heat map (how many frames each pixel showed a bloon) -> the track band;
- an arrival map (when a bloon first reached each pixel) -> the path polyline, ordered entry
  -> exit by time. Arrival time also keeps crossings apart on maps where the track crosses.

Frames are BGR content arrays (title bar cut off), all the same size. Downscale before
feeding (the tool uses 1x); the result is in normalized content coords.

Detection (`detect`):
- 'red' (default): round 1 of Standard is 20 red bloons, so a bloon pixel is a saturated red
  pixel where the empty map had none. Seen live on Monkey Meadow (2026-11-05): plain
  differencing also catches drifting cloud shadows, falling leaves and a seasonal bat all
  over the map; none of them are red. Tree Stump (autumn) has red falling leaves, so red
  blobs smaller than a bloon (`min_area`, BLOON_MIN_AREA) are dropped before anything else
  sees them; without that a drifting leaf often won the lead-bloon pick.
- 'diff': any max-channel change > DIFF_MIN vs the empty map (maps with no animation).

Bloons are narrower than the track, so the trail is a core line; PathKeepout's half-width
widens it to the track edge (fitted against a swept mask, `fit_half_width`).
"""
import numpy as np

from .placement import Grid, Mask

DIFF_MIN = 40             # max-channel |frame - empty| that counts as a bloon pixel
NOISE_DIFF = 25           # pixels that move this much between empty frames are ignored (animation)
MIN_COUNT = 2             # frames a pixel must show a bloon to be track
CELL_FILL = 0.25          # fraction of a grid cell's pixels that must be track
MIN_BLOB = 40             # px: drop smaller track blobs (pop effects, stray noise)
DEFAULT_HALF_WIDTH = 0.5   # units: Monkey Meadow fit (2026-11-05), for maps without a sweep
SPRITE_DY_UNITS = 4.0      # a red bloon's sprite is drawn this far above its track position
                           # (Meadow fit: false blocks pile up just above every horizontal stretch)
LEAD_MAX_STEP = 0.4       # normalized x per second: red bloons on Easy move ~0.05/s; a jump past this isn't the lead bloon
EXIT_EDGE = 0.02          # normalized x: a lead centre this close to the map edge is entering or leaving
LEAD_MIN_NEW = 5          # px of never-seen track a blob must reach to count as the lead bloon
LEAD_MIN_AREA = 60        # px: a whole bloon at 1x is ~ 600-900 px; smaller = clipped at an edge
BLOON_MIN_AREA = 300      # px at 1x ('red' mode): smaller red blobs are leaves / specks. Tree Stump
                          # 2026-11-05: leaves <= ~200 px, bloons 680-900, bunched bloons 1,700+
RED = (170, 90, 90)       # bloon red at 1x: R > 170, G < 90, B < 90 (body ~ (230, 30, 30) RGB)


def is_red(bgr):
    a = np.asarray(bgr, np.int16)
    return (a[..., 2] > RED[0]) & (a[..., 1] < RED[1]) & (a[..., 0] < RED[2])


def _boxes_mask(shape, boxes):
    h, w = shape[:2]
    m = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        m[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)] = True
    return m


class PathRecorder:
    def __init__(self, empty_frames, map_box=(0.0, 0.0, 1.0, 1.0), exclude=(), detect="red",
                 min_area=BLOON_MIN_AREA):
        """`empty_frames`: a few frames of the map before any bloon (more = better noise
        mask). `map_box`: only look inside it. `exclude`: normalized boxes to ignore (HUD).
        `min_area`: in 'red' mode, red blobs smaller than this (px) are ignored."""
        import cv2
        if detect not in ("red", "diff"):
            raise ValueError(detect)
        self.detect = detect
        self.min_area = min_area
        self.map_box = tuple(map_box)
        stack = np.stack([np.asarray(f, np.int16) for f in empty_frames])
        static_red = np.any([is_red(f) for f in empty_frames], axis=0).astype(np.uint8)
        self.static_red = cv2.dilate(static_red, np.ones((7, 7), np.uint8)) > 0
        self.base = np.median(stack, axis=0).astype(np.int16)
        h, w = self.base.shape[:2]
        # The animation mask is for 'diff' only. In 'red' mode it would also cut out the track
        # under animated entry/exit chevrons (Tree Stump, 2026-11-05), and they aren't red.
        noise = (stack.max(axis=0) - stack.min(axis=0)).max(axis=2) > NOISE_DIFF \
            if len(stack) > 1 and detect == "diff" else np.zeros((h, w), bool)
        if noise.any():
            noise = cv2.dilate(noise.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        self.valid = _boxes_mask((h, w), [map_box]) & ~_boxes_mask((h, w), exclude) & ~noise
        self.counts = np.zeros((h, w), np.int32)
        self.arrival = np.full((h, w), np.inf)
        self.frames = 0
        self.lead = []            # (t, x, y): the lead bloon's centre (normalized) per frame

    def feed(self, frame, t):
        """Add one frame taken at time t (s). Returns the number of bloon pixels in it."""
        if self.detect == "red":
            d = self._bloon_sized(is_red(frame) & ~self.static_red & self.valid)
        else:
            d = (np.abs(np.asarray(frame, np.int16) - self.base).max(axis=2) > DIFF_MIN) & self.valid
        self.counts += d
        new = d & np.isinf(self.arrival)
        self.arrival[new] = t
        self.frames += 1
        if new.sum() >= LEAD_MIN_NEW:
            self._track_lead(d, new, t)
        return int(d.sum())

    def _bloon_sized(self, d):
        import cv2
        if not self.min_area or not d.any():
            return d
        n, lab, stats, _ = cv2.connectedComponentsWithStats(d.astype(np.uint8), connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= self.min_area
        keep[0] = False
        return keep[lab]

    def _track_lead(self, d, new, t):
        """The lead bloon is the blob reaching the most never-seen pixels; log its centre."""
        import cv2
        n, lab, stats, cent = cv2.connectedComponentsWithStats(d.astype(np.uint8), connectivity=8)
        if n < 2:
            return
        hits = np.bincount(lab[new], minlength=n)
        hits[0] = 0
        k = int(np.argmax(hits))
        if hits[k] >= LEAD_MIN_NEW and stats[k, cv2.CC_STAT_AREA] >= LEAD_MIN_AREA:
            h, w = d.shape
            self.lead.append((t, cent[k][0] / w, cent[k][1] / h))

    def band_px(self, min_count=MIN_COUNT, min_blob=MIN_BLOB):
        """Track pixels: seen often enough, speckle removed, bloon-sized gaps closed."""
        import cv2
        m = (self.counts >= min_count).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros(n, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_blob
        return keep[lab]

    def band_mask(self, grid: Grid, band=None, fill=CELL_FILL):
        """The band rasterized onto a mask grid: a cell is track if >= `fill` of it is."""
        band = self.band_px() if band is None else band
        h, w = band.shape
        ok = np.zeros((grid.ny, grid.nx), bool)
        for j in range(grid.ny):
            for i in range(grid.nx):
                x, y = grid.centre(i, j)
                x0, x1 = round((x - grid.dx / 2) * w), round((x + grid.dx / 2) * w)
                y0, y1 = round((y - grid.dy / 2) * h), round((y + grid.dy / 2) * h)
                cell = band[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)]
                ok[j, i] = cell.size > 0 and cell.mean() >= fill
        return Mask(grid, ok)

    def polyline(self, band=None, bins=120, simplify=0.003, sprite_dy=0.0):
        """Ordered entry -> exit centre line (normalized), Douglas-Peucker with `simplify`
        (normalized x). From the lead bloon's track when there is one (dense frames); else
        the centroid of the track pixels first reached in each slice of time. `sprite_dy`
        (normalized y) moves the line down from the sprites to the track; the ends are
        extended to the map box edge."""
        band = self.band_px() if band is None else band
        h, w = band.shape
        if len(self.lead) >= 10:
            pts = [(x, y + sprite_dy) for x, y in lead_path(self.lead, aspect=h / w, box=self.map_box)]
            return extend_to_box(simplify_polyline(pts, simplify, aspect=h / w), self.map_box)
        sel = band & np.isfinite(self.arrival)
        if sel.sum() < bins:
            return []
        yy, xx = np.nonzero(sel)
        t = self.arrival[yy, xx]
        order = np.argsort(t, kind="stable")
        pts = []
        for chunk in np.array_split(order, bins):
            if len(chunk):
                pts.append((xx[chunk].mean() / w, yy[chunk].mean() / h))
        return simplify_polyline(pts, simplify, aspect=h / w)


def lead_path(lead, aspect=1.0, max_step=LEAD_MAX_STEP, edge=EXIT_EDGE, box=(0.0, 0.0, 1.0, 1.0)):
    """Lead-bloon centres in time order, cleaned up. Once the lead bloon leaves, the next bloon
    behind it becomes the 'lead' and the track would snap back (seen live, 2026-11-05), so:
    - the path ends when it reaches the edge of `box` again after leaving it (the exit);
    - a jump of more than `max_step` x the time since the last point (capped at 0.5 s) is
      dropped (normalized x, y scaled by `aspect`)."""
    out, last_t, left_edge = [], None, False
    x0, y0, x1, y1 = box
    for t, x, y in lead:
        if out:
            px, py = out[-1]
            if np.hypot(x - px, (y - py) * aspect) > max_step * min(max(t - last_t, 0.1), 0.5):
                continue
        out.append((x, y))
        last_t = t
        near = min(x - x0, x1 - x, (y - y0) * aspect, (y1 - y) * aspect) < edge
        if not near:
            left_edge = True
        elif left_edge and len(out) > 10:
            break
    return out


def simplify_polyline(pts, eps, aspect=1.0):
    """Douglas-Peucker in isotropic units (y scaled by `aspect` = content h / w)."""
    if len(pts) < 3:
        return [tuple(p) for p in pts]
    a = np.asarray(pts, float)
    iso = a * [1.0, aspect]
    keep = np.zeros(len(a), bool)
    keep[[0, -1]] = True
    stack = [(0, len(a) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        p, q = iso[i], iso[j]
        seg = q - p
        L = np.hypot(*seg)
        r = iso[i + 1:j] - p
        d = np.abs(seg[0] * r[:, 1] - seg[1] * r[:, 0]) / L if L > 0 else np.hypot(r[:, 0], r[:, 1])
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return [(round(float(x), 5), round(float(y), 5)) for x, y in a[keep]]


def keepout_agreement(keepout, mask, radius_units):
    """How well the path keep-out alone predicts a swept mask: per cell, keep-out says
    blocked vs the sweep. Returns counts and rates (the plan's 'how much the path explains')."""
    g = mask.grid
    jj, ii = np.mgrid[0:g.ny, 0:g.nx]
    pred = (keepout.distances(g.x0 + ii * g.dx, g.y0 + jj * g.dy) < radius_units).reshape(g.ny, g.nx)
    swept_blocked = ~mask.ok
    both = int((pred & swept_blocked).sum())
    false_block = int((pred & mask.ok).sum())          # keep-out blocks a valid spot (loses options)
    missed = int((~pred & swept_blocked).sum())        # blocked for another reason (edge, tree, ...)
    n = mask.ok.size
    return {"cells": n, "agree": round(float((pred == swept_blocked).mean()), 4),
            "blocked_explained": round(both / max(int(swept_blocked.sum()), 1), 4),
            "false_block_of_ok": round(false_block / max(int(mask.ok.sum()), 1), 4),
            "missed": missed, "false_block": false_block}


def extend_to_box(poly, box, max_ext=0.12):
    """Extend the first and last segments to the edge of `box` (normalized), if within
    `max_ext`: the lead bloon is first seen a little inside the map, and the track runs on
    (Tree Stump: first whole bloon at 0.09 below the top edge, under the entry chevrons).
    Points outside the box are clamped onto it first (the sprite offset can push an end just
    past the edge: Town Center's entry came out at y = 1.009)."""
    if len(poly) < 2:
        return poly
    x0, y0, x1, y1 = box
    out = [[min(max(float(x), x0), x1), min(max(float(y), y0), y1)] for x, y in poly]
    for end, prev in ((0, 1), (-1, -2)):
        p, q = np.array(out[end], float), np.array(out[prev], float)
        d = p - q
        n = np.hypot(*d)
        if n < 1e-9:
            continue
        d /= n
        ts = [t for t in ((x0 - p[0]) / d[0] if d[0] < 0 else (x1 - p[0]) / d[0] if d[0] > 0 else np.inf,
                          (y0 - p[1]) / d[1] if d[1] < 0 else (y1 - p[1]) / d[1] if d[1] > 0 else np.inf) if t >= 0]
        t = min(ts) if ts else np.inf
        if t <= max_ext:
            out[end] = [round(float(v), 5) for v in p + t * d]
    return [tuple(p) for p in out]


def fit_half_width(polyline, unit, masks, max_false=0.01, widths=np.arange(0, 12.01, 0.25)):
    """Track half-width (units) around the centre line: the largest one whose keep-out blocks
    at most `max_false` of the ok cells of every swept land_rN mask in `masks` ({class: Mask}).
    Returns (half_width or None, {class: report at it}, table)."""
    from .placement import PathKeepout
    classes = {c: int(c[6:]) for c in masks if c.startswith("land_r") and c[6:].isdigit()}
    table, best = [], (None, None)
    for hw in widths:
        ko = PathKeepout(unit, polyline=polyline, half_width=float(hw))
        reps = {c: keepout_agreement(ko, masks[c], r) for c, r in classes.items()}
        table.append((float(hw), reps))
        if reps and max(v["false_block_of_ok"] for v in reps.values()) <= max_false:
            best = (float(hw), reps)
    return best[0], best[1], table
