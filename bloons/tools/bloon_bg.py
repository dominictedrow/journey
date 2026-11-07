"""Store a map's empty-track background for the bloon detector (PLAN 5.2).

  python tools/bloon_bg.py <map>                       (Mac, game running: grabs 8 frames 0.5 s apart)
  python tools/bloon_bg.py <map> --from 'dir/empty_a_*.png'    (anywhere: saved content frames)

Use a Sandbox game on that map with no bloons and no towers on the track, nothing selected.
Not a Standard game before round 1: it animates entrance arrows and exit hearts on the track,
which vanish once the round starts and would then read as bloons. Writes
data/maps/<map>.bg.png (work-size median, pixels outside the path band blacked out),
<map>.bgnoise.png (per-pixel jitter: swaying grass) and <map>.bg.json (version stamp).
Redo it when the map's look changes (seasonal events, game updates).
"""
import argparse
import datetime
import glob
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloons import detect   # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map")
    ap.add_argument("--from", dest="src", help="glob of saved content frames instead of the live game")
    ap.add_argument("-n", type=int, default=8)
    a = ap.parse_args(argv)
    map_json = json.loads((ROOT / f"bloons/data/maps/{a.map}.json").read_text())
    if a.src:
        files = sorted(glob.glob(a.src))
        frames, source = [cv2.imread(f) for f in files], f"{len(files)} saved frames ({a.src})"
    else:
        from bloons.platform_mac import Game
        g = Game()
        frames = []
        for i in range(a.n):
            if i:
                time.sleep(0.5)
            f = g.grab()
            frames.append(None if f is None else f.content_bgr())
        source = f"{a.n} live frames"
    r = detect.background_from_frames(frames)
    if r is None:
        raise SystemExit("no frames")
    bg, dev = r
    det = detect.BloonDetector(map_json, classes={"centroids": {"x": [0] * 25}, "single_area": {}})
    det._shape(bg)
    bg[~det.band] = 0
    dev[~det.band] = 0
    meta = {"_meta": {"map": a.map, "game_version": json.loads((ROOT / "bloons/data/ui.json").read_text())["_meta"]["game_version"],
            "made": datetime.date.today().isoformat(), "source": source, "work_w": detect.WORK_W,
            "how": "tools/bloon_bg.py: median of empty-track Sandbox frames; outside the path band blacked out"}}
    detect.save_background(a.map, bg, dev, meta)
    print("wrote", *detect.bg_paths(a.map))


if __name__ == "__main__":
    main()
