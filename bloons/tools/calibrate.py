"""Map calibration tools. PLAN Phase 3.

  python tools/calibrate.py mask <map> <class> <key> [--step 0.01] [--box x0 y0 x1 y1]   (Mac, game running)
  python tools/calibrate.py show <map> <class|path> <screenshot.png> <out.png>             (anywhere)
  python tools/calibrate.py path <map> [--debug heat.png]                                   (Mac, game running)
  python tools/calibrate.py path-edit <map> <screenshot.png>                                (anywhere with a display)
  python tools/calibrate.py keepout <map> [--cls land_r6]                                   (anywhere)
  python tools/calibrate.py spots <map>                                                     (anywhere)

`mask`: hold the tower bound to <key> (e.g. q = Dart for land_r6) and move the cursor over a
grid on the map, classifying the held tower's tint at every cell (bloons/placement.py). The
result goes into data/maps/<map>.json under masks[<class>]. Use an empty map (a fresh
sandbox) with nothing selected and the tower shop showing. Nothing is clicked except the
shop's cancel button at the end. The real cursor moves (HID), so don't use the mouse
meanwhile.

`show`: draw a stored mask over a full-window 2x screenshot (green = placeable, red = not).
With `path` as the class, draws the path band (red) and polyline (yellow) instead.

`path`: PLAN 3.3. Start in a fresh sandbox: empty map, the bloon panel showing (where a new
sandbox starts), "Set round" at 1, nothing selected. Grabs a few empty-map frames, clicks the
panel's send-round button, and follows the new red bloons until they've gone
(bloons/pathmap.py). Writes the band (on the mask grid, for display), the lead bloon's
entry -> exit centre line and the track half-width to data/maps/<map>.json under `path`. The
half-width is fitted against the swept land_r6 mask if the map has one (keep-out blocks <= 1 %
of its ok cells), otherwise pathmap.DEFAULT_HALF_WIDTH.

`path-edit`: hand-correct the polyline over a screenshot. Drag a point to move it, left-click
empty space to insert a point into the nearest segment, right-click a point to delete it,
`s` saves, `q` / Esc quits.

`keepout`: the agreement report on its own (stored path vs a stored mask).

`spots`: seed data/maps/<map>.spots.json (the learned-spots memory) from the swept masks.
Readings already in the file are kept.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloons.coords import box_to_pixels                       # noqa: E402
from bloons.placement import (Grid, Mask, PathKeepout, SpotMemory, held_tint,  # noqa: E402
                              load_masks, probe_box, unit_norm)
from bloons.ui import UI                                       # noqa: E402

MAPS = ROOT / "bloons/data/maps"
MAP_BOX = [0.03, 0.0, 0.84, 1.0]     # between the wooden left border (the cursor there drops a held tower) and the shop
SETTLE_S = 0.05                      # tint follows the cursor within ~20 ms


def clip(box):
    return [min(max(v, 0.0), 1.0) for v in box]


def crop_px(full_bgr, frame, box):
    x, y, w, h = box_to_pixels(clip(box), frame.width, frame.height, frame.title_bar_px)
    return full_bgr[y:y + h, x:x + w]


def classify(g, x, y, base, base_frame):
    import Quartz as Q
    from bloons.platform_mac import cgimage_bgr
    f = g.grab()
    if f is None:
        raise RuntimeError("capture failed (screen locked or game window off screen?)")
    box = probe_box(x, y)
    px = box_to_pixels(clip(box), f.width, f.height, f.title_bar_px)
    held = cgimage_bgr(Q.CGImageCreateWithImageInRect(f.image, Q.CGRectMake(*px)))
    cx = x * f.width - px[0]
    cy = f.title_bar_px + y * (f.height - f.title_bar_px) - px[1]
    return held_tint(held, crop_px(base, base_frame, box), (cx, cy))


def sweep(g, grid, key, log_every=200):
    from bloons.platform_mac import cgimage_bgr
    g.move(0.92, 0.5)
    time.sleep(0.3)
    base_frame = g.grab()
    base = cgimage_bgr(base_frame.image)
    g.key(key)
    time.sleep(0.6)
    labels = np.full((grid.ny, grid.nx), "", dtype=object)
    none_run, retries, t0 = 0, 0, time.monotonic()
    for j in range(grid.ny):
        cols = range(grid.nx) if j % 2 == 0 else range(grid.nx - 1, -1, -1)   # serpentine
        for i in cols:
            x, y = grid.centre(i, j)
            if not g.move(x, y):
                raise RuntimeError("game not frontmost")
            time.sleep(SETTLE_S)
            lab = classify(g, x, y, base, base_frame)
            if lab == "none":                 # hold lost (or over UI): pick the tower up again
                g.key(key)
                time.sleep(0.6)
                lab = classify(g, x, y, base, base_frame)
                retries += 1
            labels[j, i] = lab
            none_run = none_run + 1 if lab == "none" else 0
            if none_run > 60:
                raise RuntimeError(f"no held tower for 60 cells (at {x:.3f}, {y:.3f}); did the key register?")
        if (j + 1) % max(1, log_every // grid.nx) == 0:
            done = (j + 1) * grid.nx
            print(f"  {done}/{grid.nx * grid.ny} cells, {time.monotonic() - t0:.0f} s", flush=True)
    g.click(*UI["buttons"]["cancel_placement"])
    print(f"  re-picked the tower {retries} times")
    return labels


def load_map(name):
    return json.loads((MAPS / f"{name}.json").read_text())


def save_map(name, data):
    (MAPS / f"{name}.json").write_text(json.dumps(data, indent=1) + "\n")


def title_bar_px(img):
    return round(UI["window"]["title_bar_pt"] * img.shape[1] / UI["window"]["expected_pt"][0])


def draw_cells(img, m, tb, colour_ok, colour_no, only=None):
    """Fill mask cells on a full-window image (title bar `tb` px). `only`: True/False to draw
    just the ok / not-ok cells."""
    import cv2
    h, w = img.shape[0] - tb, img.shape[1]
    g = m.grid
    for j in range(g.ny):
        for i in range(g.nx):
            if only is not None and bool(m.ok[j, i]) != only:
                continue
            x, y = g.centre(i, j)
            p0 = (round((x - g.dx / 2) * w), tb + round((y - g.dy / 2) * h))
            p1 = (round((x + g.dx / 2) * w), tb + round((y + g.dy / 2) * h))
            cv2.rectangle(img, p0, p1, colour_ok if m.ok[j, i] else colour_no, -1)


def draw_polyline(img, pts, tb, colour=(0, 230, 255)):
    import cv2
    h, w = img.shape[0] - tb, img.shape[1]
    px = [(round(x * w), tb + round(y * h)) for x, y in pts]
    for a, b in zip(px, px[1:]):
        cv2.line(img, a, b, colour, 3)
    for k, p in enumerate(px):
        cv2.circle(img, p, 6, (0, 255, 0) if k == 0 else (255, 0, 255) if k == len(px) - 1 else colour, -1)


def show(a):
    import cv2
    data = load_map(a.map)
    img = cv2.imread(a.screenshot)
    tb = title_bar_px(img)
    over = img.copy()
    if a.cls == "path":
        p = data["path"]
        if "band" in p:
            draw_cells(over, Mask.from_json(p["band"]), tb, (0, 0, 230), None, only=True)
        out = cv2.addWeighted(over, 0.45, img, 0.55, 0)
        draw_polyline(out, p.get("polyline", []), tb)
    else:
        draw_cells(over, Mask.from_json(data["masks"][a.cls]), tb, (0, 200, 0), (0, 0, 230))
        out = cv2.addWeighted(over, 0.45, img, 0.55, 0)
    cv2.imwrite(a.out, out)
    print(f"wrote {a.out}")


def keepout_report(data, cls="land_r6", radius=None):
    from bloons.pathmap import keepout_agreement
    if cls not in data.get("masks", {}) or "path" not in data:
        return None
    ko = PathKeepout.from_json(data["path"], unit_norm(data))
    r = radius if radius is not None else (int(cls[6:]) if cls[6:].isdigit() else 6)
    return keepout_agreement(ko, Mask.from_json(data["masks"][cls]), r)


def keepout(a):
    rep = keepout_report(load_map(a.map), a.cls, a.radius)
    print(json.dumps(rep, indent=1) if rep else f"need both a recorded path and mask {a.cls} (run `path` first)")


def spots(a):
    data = load_map(a.map)
    path = MAPS / f"{a.map}.spots.json"
    masks = load_masks(data)
    mem = SpotMemory.load(path, cell=(next(iter(masks.values())).grid.dx, next(iter(masks.values())).grid.dy))
    mem.meta.update({"map": a.map, "game_version": UI["_meta"]["game_version"]})
    added = {c: mem.import_mask(c, m) for c, m in masks.items()}
    mem.meta["seeded_from"] = f"{a.map}.json masks ({time.strftime('%Y-%m-%d')})"
    mem.save(path)
    print(f"wrote {path}: added {added}")


# --- path (PLAN 3.3) ---------------------------------------------------------------------------
PATH_EMPTY_FRAMES = 6
PATH_MAX_S = 150
PATH_QUIET_S = 4.0        # stop after this long with no bloon pixels, once some were seen
PATH_START_TIMEOUT_S = 10
PATH_MIN_PX = 30          # bloon pixels in a frame (1x) that count as "something moving"


def content_1x(g):
    f = g.grab()
    if f is None:
        raise RuntimeError("capture failed (screen locked or game window off screen?)")
    img = f.content_bgr()
    step = max(1, round(f.scale))
    return img[::step, ::step], f


def record_path(g, start="sandbox", max_s=PATH_MAX_S, log_every=5.0):
    """start: 'sandbox' clicks the bloon panel's send-round button (sandbox needs it); 'space'
    taps Space, which starts round 1 in a Standard game (sandbox is locked on most maps)."""
    from bloons.pathmap import PathRecorder
    g.ensure_front()
    g.move(0.92, 0.5)
    time.sleep(0.4)
    empty = []
    for _ in range(PATH_EMPTY_FRAMES):
        empty.append(content_1x(g)[0])
        time.sleep(0.15)
    rec = PathRecorder(empty, MAP_BOX, exclude=list(UI["hud"].values()))
    sent = g.key("space") if start == "space" else g.click(*UI["buttons"]["sandbox_send_round"])
    if not sent:
        raise RuntimeError("game not frontmost")
    g.move(0.92, 0.5)
    t0 = time.monotonic()
    seen_at, last_move, last_log = None, None, t0
    while True:
        img, f = content_1x(g)
        t = f.t - t0
        n = rec.feed(img, t)
        if n >= PATH_MIN_PX:
            seen_at = seen_at if seen_at is not None else t
            last_move = t
        if seen_at is None and t > PATH_START_TIMEOUT_S:
            raise RuntimeError("no bloons seen: did round 1 start? (fresh sandbox, bloon panel showing, Set round 1)")
        if last_move is not None and t - last_move > PATH_QUIET_S:
            break
        if t > max_s:
            print("  stopped at the time limit")
            break
        if time.monotonic() - last_log > log_every:
            last_log = time.monotonic()
            print(f"  {t:.0f} s, {rec.frames} frames, {n} bloon px", flush=True)
    print(f"  {rec.frames} frames over {t:.1f} s (first bloon at {seen_at:.1f} s)")
    return rec


def path_cmd(a):
    from bloons.platform_mac import Game
    data = load_map(a.map)
    g = Game()
    rec = record_path(g, a.start, a.max_s)
    grid = next(iter(load_masks(data).values())).grid if data.get("masks") else \
        Grid.over(MAP_BOX, 0.01, round(0.01 * g.bounds[2] / (g.bounds[3] - g.title_bar_pt), 5))
    from bloons.pathmap import DEFAULT_HALF_WIDTH, SPRITE_DY_UNITS, fit_half_width
    band = rec.band_px()
    mask = rec.band_mask(grid, band)
    unit = unit_norm(data)
    poly = rec.polyline(band, sprite_dy=SPRITE_DY_UNITS * unit[1])
    print(f"  band: {int(mask.ok.sum())} of {mask.ok.size} cells; polyline: {len(poly)} points "
          f"from {len(rec.lead)} lead-bloon positions")
    masks = load_masks(data) if data.get("masks") else {}
    hw, fit = None, None
    if any(c.startswith("land_r") for c in masks) and len(poly) >= 2:
        hw, rep, _ = fit_half_width(poly, unit, masks)
        if hw is not None:
            fit = {"max_false_block": 0.01, "reports": rep}
            print(f"  fitted half-width {hw} units:")
            for cls, r in rep.items():
                print(f"    {cls}: explains {r['blocked_explained']:.1%} of blocked cells, "
                      f"false-blocks {r['false_block_of_ok']:.2%} of ok cells")
        else:
            print("  no half-width keeps false blocks <= 1 %: check the line (show / path-edit)")
    if hw is None:
        hw = DEFAULT_HALF_WIDTH
        print(f"  no land_r6 sweep: half-width {hw} units (default)")
    if a.debug:
        import cv2
        heat = np.clip(rec.counts * (255.0 / max(int(rec.counts.max()), 1)), 0, 255).astype(np.uint8)
        cv2.imwrite(a.debug, cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO))
        Path(a.debug + ".lead.json").write_text(json.dumps([[round(v, 5) for v in r] for r in rec.lead]))
        print(f"  heat map: {a.debug} (+ raw lead-bloon track {a.debug}.lead.json)")
    data["path"] = {"polyline": [list(p) for p in poly], "half_width_units": hw, "half_width_fit": fit,
                    "sprite_dy_units": SPRITE_DY_UNITS,
                    "band": mask.to_json(), "captured": time.strftime("%Y-%m-%d"), "frames": rec.frames,
                    "how": "round-1 red bloons vs the empty map (tools/calibrate.py path)"}
    data["_meta"]["path"] = ("entry -> exit centre line + track half-width (keep-out); band = cells bloons "
                             "covered, for display (bloons/pathmap.py)")
    save_map(a.map, data)
    print(f"wrote {MAPS / (a.map + '.json')}")


def path_edit(a):
    import cv2
    data = load_map(a.map)
    pts = [list(p) for p in data.get("path", {}).get("polyline", [])]
    base = cv2.imread(a.screenshot)
    tb = title_bar_px(base)
    h, w = base.shape[0] - tb, base.shape[1]
    scale = min(1.0, 1400 / w)
    st = {"drag": None, "dirty": False}

    def to_px(p):
        return p[0] * w * scale, (tb + p[1] * h) * scale

    def nearest(x, y):
        if not pts:
            return None, 1e9
        d = [np.hypot(*(np.subtract(to_px(p), (x, y)))) for p in pts]
        k = int(np.argmin(d))
        return k, d[k]

    def on_mouse(ev, x, y, flags, _):
        nx, ny = x / scale / w, (y / scale - tb) / h
        if ev == cv2.EVENT_LBUTTONDOWN:
            k, d = nearest(x, y)
            if k is not None and d < 12:
                st["drag"] = k
            else:
                if len(pts) < 2:
                    pts.append([nx, ny])
                else:
                    from bloons.placement import _seg_dist
                    P = np.array([nx, ny * h / w])
                    segs = [_seg_dist(P, np.array([p[0], p[1] * h / w]), np.array([q[0], q[1] * h / w]))
                            for p, q in zip(pts, pts[1:])]
                    pts.insert(int(np.argmin(segs)) + 1, [nx, ny])
                st["dirty"] = True
        elif ev == cv2.EVENT_MOUSEMOVE and st["drag"] is not None:
            pts[st["drag"]] = [nx, ny]
            st["dirty"] = True
        elif ev == cv2.EVENT_LBUTTONUP:
            st["drag"] = None
        elif ev == cv2.EVENT_RBUTTONDOWN:
            k, d = nearest(x, y)
            if k is not None and d < 12:
                pts.pop(k)
                st["dirty"] = True

    win = f"path-edit {a.map}  (drag / click to insert / right-click delete / s save / q quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = base.copy()
        draw_polyline(img, pts, tb)
        cv2.imshow(win, cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("s"):
            data.setdefault("path", {})["polyline"] = [[round(x, 5), round(y, 5)] for x, y in pts]
            data["path"]["edited"] = time.strftime("%Y-%m-%d")
            save_map(a.map, data)
            st["dirty"] = False
            print(f"saved {len(pts)} points")
        elif k in (ord("q"), 27):
            if st["dirty"]:
                print("quit without saving the last changes")
            break
    cv2.destroyAllWindows()


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mask")
    m.add_argument("map")
    m.add_argument("cls", help="footprint class name, e.g. land_r6")
    m.add_argument("key", help="tower hotkey, e.g. q")
    m.add_argument("--step", type=float, default=0.01, help="x step; y step keeps cells square")
    m.add_argument("--box", type=float, nargs=4, default=MAP_BOX)
    m.add_argument("--out", default=None, help="default data/maps/<map>.json")
    sh = sub.add_parser("show")
    sh.add_argument("map")
    sh.add_argument("cls")
    sh.add_argument("screenshot")
    sh.add_argument("out")
    p = sub.add_parser("path")
    p.add_argument("map")
    p.add_argument("--debug", default=None, help="write the heat map here")
    p.add_argument("--start", choices=["sandbox", "space"], default="sandbox",
                   help="sandbox: send-round button; space: Space in a Standard game")
    p.add_argument("--max-s", type=float, default=PATH_MAX_S,
                   help="stop after this long (Auto Start chains rounds, and some maps never go quiet)")
    pe = sub.add_parser("path-edit")
    pe.add_argument("map")
    pe.add_argument("screenshot")
    ko = sub.add_parser("keepout")
    ko.add_argument("map")
    ko.add_argument("--cls", default="land_r6")
    ko.add_argument("--radius", type=float, default=None, help="footprint radius, units (default from the class)")
    sp = sub.add_parser("spots")
    sp.add_argument("map")
    a = ap.parse_args(argv)
    cmds = {"show": show, "path": path_cmd, "path-edit": path_edit, "keepout": keepout, "spots": spots}
    if a.cmd in cmds:
        return cmds[a.cmd](a)

    from bloons.platform_mac import Game
    g = Game()
    w, h = g.bounds[2], g.bounds[3] - g.title_bar_pt
    grid = Grid.over(a.box, a.step, round(a.step * w / h, 5))
    print(f"{a.map} {a.cls}: {grid.nx} x {grid.ny} cells", flush=True)
    t0 = time.monotonic()
    labels = sweep(g, grid, a.key)
    counts = {k: int((labels == k).sum()) for k in ("ok", "blocked", "none")}
    print(f"done in {time.monotonic() - t0:.0f} s: {counts}")

    out = Path(a.out) if a.out else ROOT / "bloons/data/maps" / f"{a.map}.json"
    data = json.loads(out.read_text()) if out.exists() else {"_meta": {}, "masks": {}}
    data["_meta"].update({"map": a.map, "game_version": UI["_meta"]["game_version"],
                          "window_pt": UI["window"]["expected_pt"],
                          "masks": "placeable cell centres per footprint class (bloons/placement.py)"})
    entry = Mask(grid, labels == "ok").to_json()
    entry.update({"tower_key": a.key, "swept": time.strftime("%Y-%m-%d"), "counts": counts})
    data["masks"][a.cls] = entry
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
