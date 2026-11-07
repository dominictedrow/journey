"""Build the vision data from the full fixture frames (bloons/data/fixtures_full/, gitignored):

- bloons/data/templates/<name>.png: each template in data/templates.json, cut from its
  source frame at work width.
- bloons/data/fixtures/frames/<frame>.jpg: every frame in fixtures/frames.json at work
  width, with its redact boxes greyed out, as JPEG q90 (1.5 MB for all instead of 6.5 MB
  as PNG; scores and probes barely move). The offline tests run vision.py on these.
- With --scores: a table of every template's score on every frame, for picking thresholds.

    .venv/bin/python scripts/make_templates.py [--scores]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402

from bloons.vision import SPEC, Templates, cut, box_px, to_work  # noqa: E402

DATA = ROOT / "bloons/data"
FULL, FRAMES, TPL = DATA / "fixtures_full", DATA / "fixtures/frames", DATA / "templates"
UI = json.loads((DATA / "ui.json").read_text())


def load_work(path):
    """Full saved frame (title bar included, any scale) -> work image."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    tb = round(UI["window"]["title_bar_pt"] * img.shape[1] / UI["window"]["expected_pt"][0])
    return to_work(img[tb:])


def main():
    TPL.mkdir(exist_ok=True)
    FRAMES.mkdir(exist_ok=True)
    for name, t in SPEC["templates"].items():
        crop = cut(load_work(FULL / t["src"]), t["box"])
        cv2.imwrite(str(TPL / f"{name}.png"), crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"template {name:20} {crop.shape[1]}x{crop.shape[0]}")

    manifest = json.loads((DATA / "fixtures/frames.json").read_text())["frames"]
    total = 0
    for stem, f in manifest.items():
        img = load_work(FULL / f"{stem}.png").copy()
        for box in f.get("redact", []):
            x0, y0, x1, y1 = box_px(box, img)
            img[y0:y1, x0:x1] = 128
        out = FRAMES / f"{stem}.jpg"
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        total += out.stat().st_size
    print(f"frames: {len(manifest)}, {total / 1e6:.2f} MB")

    if "--scores" in sys.argv:
        tpl = Templates()
        names = tpl.names
        print(f"\n{'frame':32}" + "".join(f"{n[:9]:>10}" for n in names))
        for stem in manifest:
            img = cv2.imread(str(FRAMES / f"{stem}.jpg"))
            m = tpl.match_all(img)
            pr = tpl.probe(img, "play_button")
            print(f"{stem[:32]:32}" + "".join(f"{m[n].score:10.2f}" for n in names)
                  + f"  probe {pr.bgr} -> {pr.label} ({pr.dist:.1f})")


if __name__ == "__main__":
    main()
