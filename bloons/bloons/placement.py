"""Placeable masks (portable): the sweep grid, the held-tower tint classifier, and the mask
format stored in data/maps/<map>.json.

While a tower is held, the game draws it at the cursor with its range circle. On a spot
where it can't go, the sprite gets a red outline and the circle turns red; otherwise the
outline is white and the circle blue-grey. `held_tint()` compares the probe crop with the
same pixels before the tower was picked up and uses two signals:
- the outline, drawn above everything (trees included): new pure-red vs new white pixels.
  Blocked if red > 0.6 x white. Valid spots measured <= 0.21 (Super's red cape), blocked
  >= 1.3 (Heli's white H), across Dart, Bomb, Spike, Super, Heli, Ace, Sub.
- the circle in a ring around the cursor: blocked if its median red-vs-green shift is
  >= 40 (blocked 86-103, valid -3..3). Tree canopies are drawn over the circle and hide
  it (median ~0 there), so this signal alone misses blocked spots under trees.

Placement checks are layered (docs/PLACEMENT-PLAN.md; bottom of this file): `can_place` rules
out spots from our own towers' footprints, the bloon-path keep-out (`PathKeepout`) and the
learned spots (`SpotMemory`, data/maps/<map>.spots.json); the tint is the exact check when
placing, and `record_placement` feeds it back into the memory.

Masks are a regular grid of cell centres in normalized content coords. `rows[j][i]` is
'1' if a tower of that footprint class can be placed with its centre at (x0 + i*dx,
y0 + j*dy), '0' if not.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .result import ActionResult

# Around the cursor, normalized half-size (~162 x 159 px at 2x): holds the whole ring.
PROBE_HALF = (0.055, 0.1)
RING_PX = (80, 150)           # at 2x: past most sprites, inside the smallest circle (Heli ~155)
RED_SHIFT_MIN = 40            # median d(R - G) in the ring: blocked 86-103, ok -3..3
OUTLINE_RED_RATIO = 0.6       # new red / new white: ok <= 0.21, blocked >= 1.3
OUTLINE_MIN = 0.005           # new red + white fraction below this = no tower drawn


def is_pure_red(bgr):
    a = bgr.astype(np.int16)
    return (a[..., 2] > 200) & (a[..., 1] < 70) & (a[..., 0] < 70)


def probe_box(x, y):
    hx, hy = PROBE_HALF
    return [x - hx, y - hy, x + hx, y + hy]


def ring_mask(shape, centre, ring=RING_PX):
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.hypot(xx - centre[0], yy - centre[1])
    return (d >= ring[0]) & (d <= ring[1])


def is_white(bgr):
    return bgr.min(axis=2) > 235


def held_tint(held, base, centre=None):
    """Classify a probe crop (BGR, same region of the frame while holding / before holding)
    around the cursor at pixel `centre` (x, y) in the crop (default: its middle):
    'blocked', 'ok', or 'none' (no tower drawn there, e.g. over UI)."""
    if held.shape != base.shape or held.size == 0:
        return "none"
    if centre is None:
        centre = (held.shape[1] / 2, held.shape[0] / 2)
    red = float((is_pure_red(held) & ~is_pure_red(base)).mean())
    white = float((is_white(held) & ~is_white(base)).mean())
    if red + white < OUTLINE_MIN:
        return "none"
    if red > OUTLINE_RED_RATIO * white:
        return "blocked"
    m = ring_mask(held.shape, centre)
    if m.any():
        d = held.astype(np.int16)[m] - base.astype(np.int16)[m]
        if np.median(d[:, 2] - d[:, 1]) >= RED_SHIFT_MIN:
            return "blocked"
    return "ok"


def tint_at(base, held, x, y):
    """held_tint() at normalized (x, y) from two content-area BGR frames (title bar cut off,
    same size): `base` grabbed before the tower was picked up, `held` while holding it there."""
    if base is None or held is None or base.shape != held.shape:
        return "none"
    h, w = held.shape[:2]
    x0, y0, x1, y1 = probe_box(x, y)
    px0, py0 = max(round(x0 * w), 0), max(round(y0 * h), 0)
    px1, py1 = min(round(x1 * w), w), min(round(y1 * h), h)
    return held_tint(held[py0:py1, px0:px1], base[py0:py1, px0:px1], (x * w - px0, y * h - py0))


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int

    @classmethod
    def over(cls, box, dx, dy):
        """Cell centres covering box [x0, y0, x1, y1], half a step in from each edge."""
        x0, y0, x1, y1 = box
        nx, ny = int((x1 - x0) / dx + 1e-9), int((y1 - y0) / dy + 1e-9)
        return cls(round(x0 + dx / 2, 5), round(y0 + dy / 2, 5), dx, dy, nx, ny)

    def centre(self, i, j):
        return self.x0 + i * self.dx, self.y0 + j * self.dy

    def cell(self, x, y):
        """Nearest cell (i, j) to a normalized point, or None outside the grid."""
        i, j = round((x - self.x0) / self.dx), round((y - self.y0) / self.dy)
        return (i, j) if 0 <= i < self.nx and 0 <= j < self.ny else None

    def to_json(self):
        return {"x0": self.x0, "y0": self.y0, "dx": self.dx, "dy": self.dy, "nx": self.nx, "ny": self.ny}


@dataclass
class Mask:
    grid: Grid
    ok: np.ndarray            # bool, shape (ny, nx)

    def placeable(self, x, y):
        c = self.grid.cell(x, y)
        return bool(c and self.ok[c[1], c[0]])

    def to_json(self):
        return {"grid": self.grid.to_json(), "rows": ["".join("1" if v else "0" for v in r) for r in self.ok]}

    @classmethod
    def from_json(cls, d):
        g = Grid(**d["grid"])
        ok = np.array([[c == "1" for c in r] for r in d["rows"]], dtype=bool)
        assert ok.shape == (g.ny, g.nx)
        return cls(g, ok)


# --- which mask a tower uses -----------------------------------------------------------------
# Rect footprints have their own swept classes. Banana Farm and Skywarden have no footprint in
# towers.json yet (no wiki dump): assumed, to be measured.
RECT_CLASSES = {(27, 27): "heli", (28, 18): "ace"}
ASSUMED = {"banana_farm": {"placement": {"land": True, "water": False}, "footprint": {"radius": 11}},
           "skywarden": {"placement": {"land": True, "water": False}, "footprint": {"radius": 8}}}


def mask_classes(name, shape):
    """Mask class names a tower can use, e.g. ['land_r6'] or ['land_r6', 'water_r6'] (Ice).
    `shape` is rules.tower_shape(tower, tiers), or None for towers without stats."""
    shape = shape or ASSUMED.get(name)
    if shape is None:
        raise KeyError(f"no footprint for {name}")
    fp = shape["footprint"]
    if "radius" in fp:
        key = f"r{fp['radius']}"
    else:
        key = RECT_CLASSES.get(tuple(fp["rect"]), "rect{}x{}".format(*fp["rect"]))
    return [f"{kind}_{key}" for kind in ("land", "water") if shape["placement"][kind]]


def placeable(masks, classes, x, y):
    """Is (x, y) a valid centre for any of `classes`, given a map's {class: Mask}? A land class
    the map lacks falls back to the next larger swept land radius (conservative); a missing
    water class counts as no water on the map."""
    for c in classes:
        m = masks.get(c) or (_larger_land(masks, c) if c.startswith("land_") else None)
        if m is not None and m.placeable(x, y):
            return True
    return False


def _larger_land(masks, cls):
    radii = sorted(int(k[6:]) for k in masks if k.startswith("land_r") and k[6:].isdigit())
    want = int(cls[6:]) if cls[6:].isdigit() else 99
    bigger = [r for r in radii if r >= want]
    return masks[f"land_r{bigger[0]}"] if bigger else None


def load_masks(map_json):
    """{class: Mask} from a data/maps/<map>.json dict."""
    return {c: Mask.from_json(d) for c, d in map_json["masks"].items()}


# === Layered placement checks (docs/PLACEMENT-PLAN.md) ========================================
# can_place() runs, cheapest first: our own towers' footprints -> the bloon-path keep-out ->
# the learned-spots memory. Whatever passes is "not ruled out"; the held-tower tint
# (held_tint) is the exact check at the moment of placing, and its result goes back into
# the memory (record_placement).
#
# Distances are in game units. A map's "scale" block gives unit_norm = (ux, uy), the
# normalized content width / height of one game unit (Monkey Meadow: 7.95 px per unit at 2x).

DISC_CELLS = 1.0          # a blocked reading also blocks this many cells around it
EXACT_EPS = 1e-4          # normalized: two readings this close are the same spot
OK_SNAP_CELLS = 0.5       # an ok reading vouches for spots within half a cell (one sweep cell)


def unit_norm(map_json):
    return tuple(map_json["scale"]["unit_norm"])


def footprint_radius(shape):
    """A footprint as one radius in game units (a rect counts as its larger half-side)."""
    fp = shape["footprint"]
    return fp["radius"] if "radius" in fp else max(fp["rect"]) / 2


def _dist_units(x0, y0, x1, y1, unit):
    return np.hypot((np.asarray(x1) - x0) / unit[0], (np.asarray(y1) - y0) / unit[1])


def _seg_dist(p, a, b):
    """Distance from point p to segment ab (all in the same isotropic units)."""
    ab, ap = b - a, p - a
    t = np.clip((ap @ ab) / max(ab @ ab, 1e-12), 0.0, 1.0)
    return float(np.hypot(*(ap - t * ab)))


class PathKeepout:
    """Where bloons travel: a centre polyline with the track's half-width (preferred: exact
    distances), or else a band of grid cells (from the round-1 heat map, coarse: +-half a cell). `blocked(x, y, r)` is True if a footprint of radius r units
    centred at (x, y) would touch the band. Tunnels show no motion, so they stay open, which
    matches the game (towers can go over tunnels)."""

    def __init__(self, unit, band: Optional[Mask] = None, polyline=None, half_width=None, pad_units=0.0):
        self.unit = unit
        self.pad = pad_units          # band only: the trail is narrower than the track by this much
        self.band = band
        self.polyline = np.asarray(polyline, float) if polyline is not None and len(polyline) else None
        self.half_width = half_width
        if self.polyline is not None and half_width is not None:
            self.band = None          # the centre line + half-width is the precise model; the band is for display
        elif band is not None:
            jj, ii = np.nonzero(band.ok)
            g = band.grid
            self._pts = np.stack([g.x0 + ii * g.dx, g.y0 + jj * g.dy], axis=1)
            self._half = (0.5 * g.dx, 0.5 * g.dy)
        else:
            raise ValueError("need a band or a polyline + half_width")
        self._band_json = band.to_json() if band is not None else None

    def distance(self, x, y):
        """Game units from (x, y) to the band (0 inside a band cell, < 0 inside the polyline band)."""
        if self.band is not None:
            if not len(self._pts):
                return float("inf")
            # distance to each track cell's rectangle (0 inside it)
            ex = np.maximum(np.abs(self._pts[:, 0] - x) - self._half[0], 0) / self.unit[0]
            ey = np.maximum(np.abs(self._pts[:, 1] - y) - self._half[1], 0) / self.unit[1]
            return float(np.hypot(ex, ey).min()) - self.pad
        u = np.asarray(self.unit)
        p, pl = np.array([x, y]) / u, self.polyline / u
        return min(_seg_dist(p, pl[k], pl[k + 1]) for k in range(len(pl) - 1)) - self.half_width

    def blocked(self, x, y, radius_units):
        return self.distance(x, y) < radius_units

    def distances(self, xs, ys):
        """distance() for many points at once (vectorized; same values)."""
        xs, ys = np.asarray(xs, float).ravel(), np.asarray(ys, float).ravel()
        if self.band is not None:
            return np.array([self.distance(x, y) for x, y in zip(xs, ys)])
        uu = np.asarray(self.unit)
        P = np.stack([xs, ys], axis=1) / uu                     # (N, 2) in units
        pl = self.polyline / uu
        A, B = pl[:-1], pl[1:]                                  # (S, 2)
        AB = B - A
        L2 = np.maximum((AB ** 2).sum(1), 1e-12)
        AP = P[:, None, :] - A[None]                            # (N, S, 2)
        t = np.clip((AP * AB[None]).sum(2) / L2[None], 0, 1)
        D = np.hypot(*(AP - t[..., None] * AB[None]).transpose(2, 0, 1))
        return D.min(axis=1) - self.half_width

    def to_json(self):
        d = {}
        if self._band_json is not None:
            d["band"] = self._band_json
        if self.polyline is not None:
            d["polyline"] = [[round(float(a), 5), round(float(b), 5)] for a, b in self.polyline]
        if self.half_width is not None:
            d["half_width_units"] = self.half_width
        if self.pad:
            d["pad_units"] = self.pad
        return d

    @classmethod
    def from_json(cls, d, unit):
        band = Mask.from_json(d["band"]) if "band" in d else None
        return cls(unit, band, d.get("polyline"), d.get("half_width_units"), d.get("pad_units", 0.0))


def _land_radius(cls):
    return int(cls[6:]) if cls.startswith("land_r") and cls[6:].isdigit() else None


class SpotMemory:
    """Placement readings per footprint class: each spot is ok or blocked. Kept in
    data/maps/<map>.spots.json, separate from the hand-calibrated <map>.json.

    lookup(cls, x, y) -> 'ok' | 'blocked' | 'unknown':
    1. a reading at the exact spot wins;
    2. a blocked reading within DISC_CELLS cells blocks the spot (don't retry next to a miss);
    3. an ok reading within OK_SNAP_CELLS vouches for it.
    Land radii nest: blocked for land_r6 means blocked for every larger land_rN, and ok for
    land_r11 means ok for every smaller one, so those readings are shared."""

    def __init__(self, cell=(0.01, 0.01844), path=None, meta=None, disc_cells=DISC_CELLS):
        self.cell = tuple(cell)
        self.path = Path(path) if path else None
        self.meta = dict(meta or {})
        self.disc_cells = disc_cells
        self._pts = {}      # cls -> (N, 2) float
        self._ok = {}       # cls -> (N,) bool

    # --- storage ---
    def classes(self):
        return list(self._pts)

    def points(self, cls):
        return self._pts.get(cls, np.zeros((0, 2))), self._ok.get(cls, np.zeros(0, bool))

    def _find(self, cls, x, y):
        p = self._pts.get(cls)
        if p is None or not len(p):
            return None
        k = np.nonzero((np.abs(p[:, 0] - x) <= EXACT_EPS) & (np.abs(p[:, 1] - y) <= EXACT_EPS))[0]
        return int(k[0]) if len(k) else None

    def record(self, cls, x, y, ok, overwrite=True):
        """Store a reading. Returns True if anything changed."""
        k = self._find(cls, x, y)
        if k is not None:
            if not overwrite or self._ok[cls][k] == bool(ok):
                return False
            self._ok[cls][k] = bool(ok)
            return True
        self._pts[cls] = np.vstack([self._pts.get(cls, np.zeros((0, 2))), [[x, y]]])
        self._ok[cls] = np.append(self._ok.get(cls, np.zeros(0, bool)), bool(ok))
        return True

    def record_many(self, cls, xy, ok, overwrite=False):
        """Bulk insert (e.g. a sweep). Existing readings are kept unless overwrite."""
        xy, ok = np.asarray(xy, float).reshape(-1, 2), np.asarray(ok, bool).reshape(-1)
        n = 0
        if cls not in self._pts or not len(self._pts[cls]):
            self._pts[cls], self._ok[cls] = xy.copy(), ok.copy()
            return len(xy)
        for (x, y), v in zip(xy, ok):
            n += self.record(cls, float(x), float(y), bool(v), overwrite)
        return n

    # --- queries ---
    def _related(self, cls, want):
        """Classes whose `want` ('ok' / 'blocked') readings also apply to cls."""
        r = _land_radius(cls)
        if r is None:
            return [cls] if cls in self._pts else []
        out = []
        for c in self._pts:
            rc = _land_radius(c)
            if rc is None:
                continue
            if c == cls or (want == "blocked" and rc <= r) or (want == "ok" and rc >= r):
                out.append(c)
        return out

    def lookup(self, cls, x, y):
        k = self._find(cls, x, y)
        if k is not None:
            return "ok" if self._ok[cls][k] else "blocked"
        cx, cy = self.cell
        for c in self._related(cls, "blocked"):
            p, ok = self._pts[c], self._ok[c]
            if c != cls:
                kk = self._find(c, x, y)
                if kk is not None and not ok[kk]:
                    return "blocked"
            d = np.hypot((p[:, 0] - x) / cx, (p[:, 1] - y) / cy)
            if np.any(~ok & (d <= self.disc_cells)):
                return "blocked"
        for c in self._related(cls, "ok"):
            p, ok = self._pts[c], self._ok[c]
            d = np.hypot((p[:, 0] - x) / cx, (p[:, 1] - y) / cy)
            if np.any(ok & (d <= OK_SNAP_CELLS + 1e-9)):
                return "ok"
        return "unknown"

    def lookup_any(self, classes, x, y):
        """For a tower that can use several classes (Ice: land or water): ok if any is ok,
        blocked if all are blocked, else unknown."""
        r = [self.lookup(c, x, y) for c in classes]
        if "ok" in r:
            return "ok"
        return "blocked" if r and all(v == "blocked" for v in r) else "unknown"

    # --- file ---
    def to_json(self):
        cls = {}
        for c in sorted(self._pts):
            p, ok = self._pts[c], self._ok[c]
            cls[c] = [[round(float(x), 5), round(float(y), 5), int(v)] for (x, y), v in zip(p, ok)]
        meta = dict(self.meta)
        meta.update({"cell": list(self.cell), "disc_cells": self.disc_cells,
                     "what": "placement readings per footprint class: [x, y, ok] in normalized content coords "
                             "(bloons/placement.py SpotMemory). Written at runtime."})
        return {"_meta": meta, "classes": cls}

    def save(self, path=None):
        path = Path(path or self.path)
        d = self.to_json()
        # One line per class keeps the file small and diffable.
        body = ",\n".join(f" {json.dumps(c)}: {json.dumps(v, separators=(',', ':'))}" for c, v in d["classes"].items())
        text = '{\n "_meta": ' + json.dumps(d["_meta"]) + ',\n "classes": {\n' + body + "\n }\n}\n"
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, cell=None):
        path = Path(path)
        if not path.exists():
            return cls(cell or (0.01, 0.01844), path)
        d = json.loads(path.read_text())
        m = d["_meta"]
        mem = cls(cell or m.get("cell", (0.01, 0.01844)), path, m, m.get("disc_cells", DISC_CELLS))
        for c, rows in d["classes"].items():
            a = np.asarray(rows, float).reshape(-1, 3)
            mem._pts[c], mem._ok[c] = a[:, :2].copy(), a[:, 2] > 0.5
        return mem

    @classmethod
    def from_masks(cls, masks, path=None, meta=None):
        """Seed from swept masks ({class: Mask}): every cell centre becomes a reading."""
        g0 = next(iter(masks.values())).grid
        mem = cls((g0.dx, g0.dy), path, meta)
        for c, m in masks.items():
            mem.import_mask(c, m)
        return mem

    def import_mask(self, cls, mask, overwrite=False):
        g = mask.grid
        jj, ii = np.mgrid[0:g.ny, 0:g.nx]
        xy = np.stack([g.x0 + ii.ravel() * g.dx, g.y0 + jj.ravel() * g.dy], axis=1)
        return self.record_many(cls, xy, mask.ok.ravel(), overwrite)


# --- can_place -------------------------------------------------------------------------------

@dataclass
class Placed:
    """A tower on the map, as the harness registry knows it (heroes too: kind = hero key)."""
    kind: str
    xy: tuple
    tiers: tuple = (0, 0, 0)
    tid: Optional[int] = None


@dataclass
class PlaceContext:
    towers: dict                                   # towers.json "towers"
    unit: tuple                                    # map scale: normalized size of 1 game unit
    placed: list = field(default_factory=list)     # [Placed]
    keepout: Optional[PathKeepout] = None
    memory: Optional[SpotMemory] = None
    heroes: Optional[dict] = None                  # heroes.json "heroes"
    map_box: tuple = (0.03, 0.0, 0.84, 1.0)        # placeable screen area (calibrate.MAP_BOX)


HERO_SHAPE = {"placement": {"land": True, "water": False}, "footprint": {"radius": 7}}   # dumped heroes: r7, land


def shape_of(kind, tiers, ctx):
    """{'placement', 'footprint', ...} for a tower or hero key, or None if unknown."""
    from .rules import tower_shape
    if kind in ctx.towers:
        return tower_shape(ctx.towers[kind], tiers) or ASSUMED.get(kind)
    if ctx.heroes is not None and kind in ctx.heroes:
        return HERO_SHAPE
    return None


def can_place(kind, x, y, ctx: PlaceContext, tiers=(0, 0, 0)):
    """Can a `kind` tower (or hero) go at (x, y)? Checks only what's known without the game:
    the screen area, our own towers' footprints, the path keep-out and the learned spots.
    ok=True means 'not ruled out' (reason 'known_ok' if the memory has an ok reading,
    'unknown' otherwise); the held-tower tint has the final word."""
    x0, y0, x1, y1 = ctx.map_box
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return ActionResult(False, "off_map")
    shape = shape_of(kind, tiers, ctx)
    if shape is None:
        return ActionResult(False, f"no_footprint:{kind}")
    r = footprint_radius(shape)
    for t in ctx.placed:
        ts = shape_of(t.kind, t.tiers, ctx)
        rt = footprint_radius(ts) if ts else 6
        if _dist_units(x, y, t.xy[0], t.xy[1], ctx.unit) < r + rt:
            return ActionResult(False, f"overlaps_tower:{t.tid if t.tid is not None else t.kind}")
    if ctx.keepout is not None and not shape["placement"].get("track", False) and ctx.keepout.blocked(x, y, r):
        return ActionResult(False, "path")
    if ctx.memory is not None:
        classes = mask_classes(kind, shape) if kind in ctx.towers else ["land_r7"]
        v = ctx.memory.lookup_any(classes, x, y)
        if v == "blocked":
            return ActionResult(False, "memory_blocked")
        if v == "ok":
            return ActionResult(True, "known_ok")
    return ActionResult(True, "unknown")


def record_placement(ctx: PlaceContext, kind, x, y, ok, tiers=(0, 0, 0), save=True):
    """Feed a tint / placement outcome back into the memory. A blocked reading next to one of
    our own towers isn't saved: that block goes away when the tower is sold. A tower with
    several classes (Ice) gets an ok reading only when it has a single class (we don't know
    which class the spot was ok for), but a blocked reading for all of them."""
    if ctx.memory is None:
        return False
    shape = shape_of(kind, tiers, ctx)
    if shape is None:
        return False
    classes = mask_classes(kind, shape) if kind in ctx.towers else ["land_r7"]
    if not ok:
        r = footprint_radius(shape)
        for t in ctx.placed:
            ts = shape_of(t.kind, t.tiers, ctx)
            rt = footprint_radius(ts) if ts else 6
            if _dist_units(x, y, t.xy[0], t.xy[1], ctx.unit) < r + rt + 2 * ctx.memory.cell[0] / ctx.unit[0]:
                return False
    elif len(classes) > 1:
        return False
    changed = False
    for c in classes:
        changed |= ctx.memory.record(c, x, y, ok)
    if changed and save and ctx.memory.path:
        ctx.memory.save()
    return changed
