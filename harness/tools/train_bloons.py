"""Build data/bloon_classes.json (bloon colour centroids) from labelled Sandbox captures. PLAN 5.2.

  python tools/train_bloons.py <dir> --run a:red,blue,green --run b:purple,lead [--map monkey_meadow]
                               [--out bloons/data/bloon_classes.json] [--check]

<dir> holds, per run, background frames `empty_<run>_*.png` (the track empty) and frames
`<kind>_NN.png` taken while only that bloon type was on the track (Sandbox: one bloon button
pressed a few times; wait for the track to clear between types). Content frames at any size.

Every blob in a <kind> frame is labelled <kind>, except dark, purple-tinted blobs in batches
whose bloons aren't dark (black / zebra / lead): those are labelled 'bat' (Monkey Meadow's
Halloween bats fly over the track) and kept as a non-bloon class.

Centroids are trained on even frames; odd frames are the held-out check (printed), then the
written file uses all frames. `single_area`: the median blob area per class (2x px).
"""
import argparse
import collections
import datetime
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloons import detect   # noqa: E402

DARK_KINDS = {"black", "zebra", "lead"}


def is_bat(f):
    dark, purple = f[18], f[14:17].sum()        # V < 0.25 fraction; hue bins 140-169 (x3 weighted)
    return dark > 0.35 and purple > 0.3


def blobs(det, frame):
    """[(features, area)] for every blob, before classification."""
    img, px = det.pixels(frame)
    d = cv2.morphologyEx(px.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    d = cv2.morphologyEx(d, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(d)
    s = (px.shape[1] / detect.FULL_W) ** 2
    out = []
    for i in range(1, n):
        area = st[i, 4] / s
        m = (lab == i) & px
        if area >= detect.MIN_AREA and m.any():
            out.append((detect.features(img, m, area), area, detect.pixel_bins(img, m)[0], s))
    return out


def collect(d, runs, map_json):
    det = detect.BloonDetector(map_json, classes={"centroids": {"x": [0] * 25}, "single_area": {}})
    rows = []                                     # (label, frame index, features, area, pixel bins, px scale)
    for run, kinds in runs:
        bg = [cv2.imread(f) for f in sorted(glob.glob(f"{d}/empty_{run}_*.png"))]
        if not det.set_background(bg):
            raise SystemExit(f"no background frames empty_{run}_*.png in {d}")
        for k in kinds:
            for fi, path in enumerate(sorted(glob.glob(f"{d}/{k}_[0-9]*.png"))):
                for f, area, bins, sc in blobs(det, cv2.imread(path)):
                    label = "bat" if k not in DARK_KINDS and is_bat(f) else k
                    rows.append((label, fi, f, area, bins, sc))
    return rows


def centroids(rows):
    by = collections.defaultdict(list)
    for label, _, f, *_ in rows:
        by[label].append(f)
    return {k: np.mean(v, 0) for k, v in sorted(by.items())}


def classify(C, f):
    names = list(C)
    return names[int(np.argmin([((C[k] - f) ** 2).sum() for k in names]))]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--run", action="append", required=True, help="<run>:<kind>,<kind>,...")
    ap.add_argument("--map", default="monkey_meadow")
    ap.add_argument("--out", default=str(ROOT / "bloons/data/bloon_classes.json"))
    ap.add_argument("--check", action="store_true", help="only print the held-out check")
    a = ap.parse_args(argv)
    runs = [(r.split(":")[0], r.split(":")[1].split(",")) for r in a.run]
    map_json = json.loads((ROOT / f"bloons/data/maps/{a.map}.json").read_text())
    rows = collect(a.dir, runs, map_json)
    train = [r for r in rows if r[1] % 2 == 0]
    test = [r for r in rows if r[1] % 2 == 1]
    C = centroids(train)
    wrong = collections.Counter((r[0], classify(C, r[2])) for r in test if classify(C, r[2]) != r[0])
    acc = 1 - sum(wrong.values()) / max(len(test), 1)
    print(f"held-out: {acc:.3f} of {len(test)} blobs; errors {dict(wrong)}")
    print("blobs per class:", dict(collections.Counter(r[0] for r in rows)))
    if a.check:
        return
    C = centroids(rows)
    single = {}
    for k in C:
        areas = np.array([r[3] for r in rows if r[0] == k])
        med = np.median(areas)
        single[k] = int(np.median(areas[(areas > 0.7 * med) & (areas < 1.3 * med)]))
    # per-pixel colour model (detect.decompose): hue x sat histogram per colour class, and the
    # area a lone bloon's pixels get assigned to its own class
    hist = {}
    for k in detect.COLOUR_CLASSES:
        h = np.zeros(72)
        for r in rows:
            if r[0] == k and abs(r[3] / single.get(k, r[3]) - 1) < 0.3:
                h += np.bincount(r[4], minlength=72)
        if h.sum():
            hist[k] = h / h.sum()
    names = [k for k in detect.COLOUR_CLASSES if k in hist]
    logp = np.log(np.array([hist[k] for k in names]) + 1e-4)
    pixel_area = {}
    for j, k in enumerate(names):
        got = [np.sum(np.argmax(logp[:, r[4]], 0) == j) / r[5] for r in rows
               if r[0] == k and len(r[4]) and abs(r[3] / single[k] - 1) < 0.3]
        pixel_area[k] = int(np.median(got))
    print("pixel_area:", pixel_area)
    out = {"_meta": {"game_version": json.loads((ROOT / "bloons/data/ui.json").read_text())["_meta"]["game_version"],
                     "made": datetime.date.today().isoformat(), "map": a.map,
                     "how": "tools/train_bloons.py on Sandbox captures (one bloon type at a time, empty-track "
                            "backgrounds); features = detect.features(); nearest centroid",
                     "held_out_accuracy": round(acc, 3), "blobs": len(rows)},
           "non_bloon": ["bat"],
           "single_area": {k: v for k, v in single.items() if k != "bat"},
           "pixel_area": pixel_area,
           "pixel_hist": {k: [round(float(x), 5) for x in hist[k]] for k in names},
           "centroids": {k: [round(float(x), 4) for x in v] for k, v in C.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
