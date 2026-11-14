"""Range circle measurement (PLAN 5.5, portable): a selected tower's range from pixels.

Selecting a tower draws its range circle (a translucent disc with an outline) around it, and
opens the upgrade panel. The difference between a selected and a deselected frame is the
disc (solid), the panel and whatever else moved in between (swaying foliage, leaves, flowers:
scattered). So the radius is read along rays from the tower's centre: on each ray, the
changed pixels are grouped into runs (gaps under `margin` px merged), and the disc is the
first run at least `min_run` long that starts before `start_r` (the tower's sprite may
change only along its outline, which makes short runs); its end is the edge. A ray counts
only if `margin` px past the edge still lie inside the usable area (not the frame edge, the panel or the other
boxes passed in), which drops rays cut short. Each ray's edge is turned into game units
through the ground scale (an ellipse on screen: the map is seen at a tilt, y px per unit =
0.866 x px per unit), and the range is the median of the densest group of edges
(`_cluster`: rays stopped by a sprite or foliage scatter, the real edge repeats).

Parameters live in data/ui.json `range_circle`. The ground unit is the map's
`scale.unit_norm` with y times `ground_y_scale` (`ground_unit`).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ui import UI


@dataclass(frozen=True)
class RangeRead:
    units: Optional[float]       # median edge distance in game units, None if too few rays
    rays: int                    # rays counted
    spread: float                # interquartile range of the counted edges, in units


def _cfg():
    return UI["range_circle"]


def changed_mask(a, b, thr=None):
    """Pixels whose largest channel difference between two BGR frames is >= thr."""
    thr = _cfg()["diff_min"] if thr is None else thr
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) >= thr


def measure(selected, deselected, center, unit, exclude=()):
    """RangeRead for the circle around `center` (normalized content coords) from two
    same-size content frames. `unit`: (ux, uy) normalized size of one game unit along x and
    y on the ground (the circle is an ellipse on screen: uy/ux in px = 0.866, RESEARCH 5.5).
    `exclude`: normalized boxes (x0, y0, x1, y1) to ignore."""
    c = _cfg()
    H, W = selected.shape[:2]
    mask = changed_mask(selected, deselected)
    usable = np.ones((H, W), bool)
    for x0, y0, x1, y1 in exclude:
        usable[max(int(y0 * H), 0):int(np.ceil(y1 * H)), max(int(x0 * W), 0):int(np.ceil(x1 * W))] = False
    cx, cy = center[0] * W, center[1] * H
    upx, upy = unit[0] * W, unit[1] * H
    r_min, r_start, margin, n = int(c["min_r"] * W), int(c["start_r"] * W), int(c["margin"] * W), c["rays"]
    min_run = int(c["min_run"] * W)
    rs = np.arange(r_min, int(c["max_r"] * W) + margin + 1)
    edges = []
    for k in range(n):
        ang = 2 * np.pi * k / n
        dx, dy = np.cos(ang), np.sin(ang)
        xs = np.round(cx + rs * dx).astype(int)
        ys = np.round(cy + rs * dy).astype(int)
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        ok[ok] = usable[ys[ok], xs[ok]]
        stop = len(rs) if ok.all() else int(np.argmin(ok))     # the usable stretch from r_min
        on = np.flatnonzero(mask[ys[:stop], xs[:stop]])
        if not len(on):
            continue
        # changed runs along the ray, gaps shorter than `margin` merged
        cut = np.flatnonzero(np.diff(on) > margin)
        starts, ends = on[np.r_[0, cut + 1]], on[np.r_[cut, len(on) - 1]]
        # the disc: the first long run that starts near the tower (a sprite outline is short)
        disc = [(a, b) for a, b in zip(starts, ends) if a < r_start - r_min and b - a >= min_run]
        if not disc or disc[0][1] + margin >= stop:   # none, or cut short by the usable area
            continue
        r = rs[disc[0][1]] + c["edge_bias_px"]
        edges.append(float(np.hypot(r * dx / upx, r * dy / upy)))
    return _cluster(edges, c)


def _cluster(edges, c):
    """The densest group of ray edges: the fullest `bin` -unit bin, then every edge within
    `keep` units of it. Rays stopped inside a sprite or by foliage scatter; the real edge
    repeats to ~0.1 unit. RangeRead.rays counts the group."""
    if len(edges) < c["min_rays"]:
        return RangeRead(None, len(edges), 0.0)
    e = np.asarray(edges)
    b = np.floor(e / c["cluster_bin"]).astype(int)
    vals, counts = np.unique(b, return_counts=True)
    centre = (vals[int(np.argmax(counts))] + 0.5) * c["cluster_bin"]
    g = e[np.abs(e - centre) <= c["cluster_keep"]]
    if len(g) < c["min_rays"]:
        return RangeRead(None, len(g), 0.0)
    q1, med, q3 = np.percentile(g, (25, 50, 75))
    return RangeRead(round(float(med), 2), len(g), round(float(q3 - q1), 2))


def ground_unit(unit_norm):
    """A map's scale.unit_norm (ux, uy) -> one game unit on the ground (ux, uy * 0.866)."""
    return unit_norm[0], unit_norm[1] * _cfg()["ground_y_scale"]
