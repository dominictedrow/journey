"""GameState -> fixed-size numbers for a model (PLAN 5.6). Portable, no OS calls.

  vector(state, spec)  -> float32 (D,)        GameState.to_vector()
  grid(state, spec)    -> float32 (C, ny, nx) GameState.grid_image()
  layout(spec)         -> [(name, start, stop)] what each part of the vector is
  grid_channels(spec)  -> [name] per grid channel
  cell_centre(i, j)    -> normalized content (x, y) of grid cell (column i, row j), e.g. for
                          a PLACE(kind, cell) action

The sizes, one-hot orders and scales live in data/obs.json (`load_spec`). Every value is a
plain transform of a GameState field (one-hot, 0/1 flag, log-scaled amount, fraction), with a
`*_known` flag next to anything observe() can leave as None, so nothing here decides what
matters: that's the model's job (Dom's, Phase 7).

Tower slots: towers sit in `state.slots` order (the harness gives each tower the first free
slot when placed and frees it when sold, so a tower keeps its slot for its whole life).
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .placement import _larger_land

DATA = Path(__file__).parent / "data"
UPGRADE_LOCKS = ("max_tier", "crosspath", "tier5_taken")
HAS_FLAGS = ("camo", "regrow", "fortified", "lead", "moab_class")


@lru_cache(maxsize=None)
def _load(path):
    return json.loads(Path(path).read_text())


def load_spec(path=None):
    """data/obs.json (or another file with the same keys) as a dict."""
    return _load(str(path or DATA / "obs.json"))


def _log(v, scale):
    return float(np.log1p(max(0.0, float(v))) / np.log1p(scale))


def _onehot(value, names):
    out = [0.0] * len(names)
    if value in names:
        out[names.index(value)] = 1.0
    return out


class _Builder:
    """Appends named segments; the same calls with build=False give just the layout."""

    def __init__(self):
        self.vals, self.parts = [], []

    def add(self, name, values):
        values = list(values)
        self.parts.append((name, len(self.vals), len(self.vals) + len(values)))
        self.vals.extend(float(v) for v in values)


def _slot_size(spec):
    return 1 + len(spec["tower_kinds"]) + 2 + 3 + 2 + 1 + len(spec["targeting"]) + 3 * 4


def _encode(state, spec):
    """The whole vector, segment by segment. `state` None = zeros (for layout())."""
    s, sc = state, spec["scale"]
    b = _Builder()
    get = (lambda name, default=None: getattr(s, name, default)) if s is not None else (lambda n, d=None: d)

    # --- global ---
    money, lives = get("money"), get("lives")
    b.add("money", [_log(money, sc["money"]) if money is not None else 0.0, money is not None])
    b.add("lives", [lives / sc["lives"] if lives is not None else 0.0, lives is not None])
    nxt, mx = get("next_round", 0), get("max_round")
    b.add("round", [nxt / sc["round"], (mx or 0) / sc["round"], mx is not None])
    ff = get("fast_forward")
    b.add("flags", [bool(get("round_active")), bool(get("round_ended")), get("screen") == "game",
                    bool(ff), ff is not None])
    b.add("outcome", [get("outcome") == "win", get("outcome") == "loss"])
    b.add("map", _onehot(get("map_name"), spec["maps"]))
    b.add("difficulty", _onehot(get("difficulty"), spec["difficulties"]))
    b.add("mode", _onehot(get("mode"), spec["modes"]))

    # --- next round's composition (rounds.json; Standard only) ---
    comp = get("round_composition")
    types = spec["bloon_types"]
    counts = (comp or {}).get("counts") or {}
    has = (comp or {}).get("has") or {}
    b.add("composition_counts", [_log(counts.get(t, 0), sc["count"]) for t in types])
    b.add("composition_has", [bool(has.get(k)) for k in HAS_FLAGS])
    b.add("composition_size", [_log((comp or {}).get("rbe", 0), sc["count"] * 100),
                               (comp or {}).get("duration", 0) / sc["duration_s"], comp is not None])

    # --- bloons seen on the track this frame (detect.Bloon) ---
    bloons = get("bloons")
    seen = {}
    for bl in bloons or []:
        seen[bl.kind] = seen.get(bl.kind, 0) + bl.count
    lead = path_progress(get("path"), [(bl.x, bl.y) for bl in bloons or []])
    b.add("bloons_counts", [_log(seen.get(t, 0), sc["count"]) for t in types])
    b.add("bloons_lead", [max(lead) if lead else 0.0, _log(sum(seen.values()), sc["count"]),
                          bloons is not None])

    # --- tower slots ---
    towers = {t.tid: t for t in get("towers") or []}
    options = get("upgrade_options") or {}
    slots = list(get("slots") or [])
    n = spec["tower_slots"]
    slots = (slots + [None] * n)[:n]
    for k, tid in enumerate(slots):
        t = towers.get(tid)
        if t is None:
            b.add(f"slot{k}", [0.0] * _slot_size(spec))
            continue
        units = t.range_measured if t.range_measured is not None else t.range_units
        tgt = t.targeting if t.targeting is not None else "default"
        tgt = tgt if tgt in spec["targeting"] else "other"
        up = []
        for o in (options.get(tid) or [None] * 3)[:3]:
            if o is None:
                up += [0.0, 0.0, 0.0, 0.0]
            else:
                up += [_log(o.cost, sc["money"]) if o.cost is not None else 0.0, o.affordable,
                       o.locked in UPGRADE_LOCKS, o.locked == "max_tier"]
        b.add(f"slot{k}", [1.0] + _onehot(t.kind, spec["tower_kinds"]) + [t.xy[0], t.xy[1]]
              + [v / 5 for v in t.tiers] + [(units or 0) / sc["range_units"], units is not None]
              + [_log(t.sell_value, sc["money"])] + _onehot(tgt, spec["targeting"]) + up)

    # --- hero ---
    h = get("hero")
    if h is None:
        b.add("hero", [0.0] * (1 + len(spec["heroes"]) + 4))
    else:
        b.add("hero", [1.0] + _onehot(h.name, spec["heroes"]) + [h.xy[0], h.xy[1],
              h.level / sc["hero_level"], _log(h.sell_value, sc["money"])])

    # --- ability bar slots ---
    abil = get("abilities") or []
    for k in range(spec["ability_slots"]):
        a = abil[k] if k < len(abil) else None
        if a is None:
            b.add(f"ability{k}", [0.0] * 6)
            continue
        owner = slots.index(a.tid) + 1 if a.tid in slots else 0
        b.add(f"ability{k}", [1.0, a.ready, a.cooldown_frac, a.between_rounds,
                              owner / n, a.tid == "hero"])
    return b


def vector(state, spec=None):
    """float32 (D,) for one GameState; layout(spec) says what each index is."""
    return np.asarray(_encode(state, spec or load_spec()).vals, np.float32)


def layout(spec=None):
    """[(segment name, start, stop)] of vector(), in order."""
    return _encode(None, spec or load_spec()).parts


def path_progress(path, points):
    """Fraction of the way along `path` (entry -> exit polyline, normalized) for each point:
    its nearest point on the line, as arc length / total length. [] if no path."""
    if path is None or len(path) < 2 or not points:
        return []
    pl = np.asarray(path, float)
    A, B = pl[:-1], pl[1:]
    AB = B - A
    L = np.hypot(*AB.T)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    P = np.asarray(points, float)
    AP = P[:, None, :] - A[None]
    t = np.clip((AP * AB[None]).sum(2) / np.maximum(L ** 2, 1e-12)[None], 0, 1)
    D = np.hypot(*(AP - t[..., None] * AB[None]).transpose(2, 0, 1))
    k = D.argmin(axis=1)
    along = cum[k] + t[np.arange(len(P)), k] * L[k]
    return [float(v) for v in along / max(cum[-1], 1e-12)]


# --- grid image ---------------------------------------------------------------------------
def grid_channels(spec=None):
    g = (spec or load_spec())["grid"]
    return ["path"] + [f"placeable_{c}" for c in g["placeable_classes"]] + ["towers", "ranges", "bloons"]


def cell_centre(i, j, spec=None):
    """Normalized content (x, y) at the centre of grid cell column i, row j."""
    g = (spec or load_spec())["grid"]
    x0, y0, x1, y1 = g["box"]
    return x0 + (i + 0.5) * (x1 - x0) / g["nx"], y0 + (j + 0.5) * (y1 - y0) / g["ny"]


def _samples(g):
    """Sample points (sub x sub per cell) -> xs, ys of shape (ny, nx, sub*sub)."""
    x0, y0, x1, y1 = g["box"]
    nx, ny, sub = g["nx"], g["ny"], g["sub"]
    fx = (np.arange(nx * sub) + 0.5) / (nx * sub)
    fy = (np.arange(ny * sub) + 0.5) / (ny * sub)
    X, Y = np.meshgrid(x0 + fx * (x1 - x0), y0 + fy * (y1 - y0))      # (ny*sub, nx*sub)

    def cells(a):
        return a.reshape(ny, sub, nx, sub).transpose(0, 2, 1, 3).reshape(ny, nx, sub * sub)
    return cells(X), cells(Y)


def _mask_at(mask, xs, ys):
    gr = mask.grid
    i = np.rint((xs - gr.x0) / gr.dx).astype(int)
    j = np.rint((ys - gr.y0) / gr.dy).astype(int)
    inside = (i >= 0) & (i < gr.nx) & (j >= 0) & (j < gr.ny)
    out = np.zeros(xs.shape, bool)
    out[inside] = mask.ok[j[inside], i[inside]]
    return out


def grid(state, spec=None):
    """float32 (C, ny, nx) image of the play area; grid_channels(spec) names the channels."""
    spec = spec or load_spec()
    g = spec["grid"]
    xs, ys = _samples(g)
    shape = xs.shape[:2]
    chans = []
    # path: share of the cell within the bloons' lane (centre line +- grid path_half_width_units;
    # state.path_half_width_units is the placement keep-out pad, far narrower)
    path, hw = state.path, g["path_half_width_units"]
    ux, uy = state.unit
    if path is not None and len(path) >= 2:
        pl = np.asarray(path, float) / (ux, uy)
        P = np.stack([xs.ravel() / ux, ys.ravel() / uy], 1)
        A, B = pl[:-1], pl[1:]
        AB = B - A
        t = np.clip(((P[:, None] - A[None]) * AB[None]).sum(2) / np.maximum((AB ** 2).sum(1), 1e-12)[None], 0, 1)
        D = np.hypot(*(P[:, None] - A[None] - t[..., None] * AB[None]).transpose(2, 0, 1)).min(1)
        chans.append((D < hw).reshape(xs.shape).mean(2))
    else:
        chans.append(np.zeros(shape))
    # placeable, one channel per class (missing land class -> next larger swept one)
    masks = state.placeable or {}
    for c in g["placeable_classes"]:
        m = masks.get(c) or (_larger_land(masks, c) if c.startswith("land_") and masks else None)
        chans.append(_mask_at(m, xs, ys).mean(2) if m is not None else np.zeros(shape))
    # towers: footprints; ranges: ellipses covering the cell
    foot = np.zeros(xs.shape, bool)
    cover = np.zeros(shape)
    for t in state.towers:
        r = t.footprint_units if t.footprint_units is not None else g["default_footprint_units"]
        foot |= np.hypot((xs - t.xy[0]) / ux, (ys - t.xy[1]) / uy) < r
        if t.range_norm is not None:
            rx, ry = t.range_norm
            cover += (np.hypot((xs - t.xy[0]) / rx, (ys - t.xy[1]) / ry) <= 1).mean(2)
    if state.hero is not None:
        foot |= np.hypot((xs - state.hero.xy[0]) / ux, (ys - state.hero.xy[1]) / uy) < g["hero_footprint_units"]
    chans.append(foot.mean(2))
    chans.append(np.minimum(cover / g["range_cap"], 1.0))
    # bloons: per-cell count
    dens = np.zeros(shape)
    x0, y0, x1, y1 = g["box"]
    for bl in state.bloons or []:
        i = int((bl.x - x0) / (x1 - x0) * g["nx"])
        j = int((bl.y - y0) / (y1 - y0) * g["ny"])
        if 0 <= i < g["nx"] and 0 <= j < g["ny"]:
            dens[j, i] += bl.count
    chans.append(np.log1p(dens) / np.log1p(g["bloon_cap"]))
    return np.stack(chans).astype(np.float32)
