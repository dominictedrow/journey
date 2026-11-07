"""Template matching and colour probes on game frames. Portable (numpy + OpenCV); the Mac
capture hands frames over via platform_mac.Frame.content_bgr().

Everything runs on a "work image": the content area (window minus title bar) as BGR,
resized to WORK_W pixels wide. The game UI scales with the window, so boxes are the same
normalized [x0, y0, x1, y1] as in ui.json, and templates are stored at work width.

- `match(img, name)`: normalized cross-correlation (TM_CCOEFF_NORMED) of one template
  within its search region (its box grown by a margin, or an explicit "search" box).
- `probe(img, name)`: mean colour of a small box, classified by nearest reference colour.
  For states that shape can't separate (fast-forward on vs off is just a lighter green).
- `button_state(img)`: the bottom-right play button: 'idle' (start round), 'ff_off',
  'ff_on' (round running), or None (not visible / dimmed by an overlay).

Template boxes, thresholds and probe colours live in data/templates.json.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DATA = Path(__file__).parent / "data"
SPEC = json.loads((DATA / "templates.json").read_text())
WORK_W = SPEC["_meta"]["work_width"]


def to_work(content_bgr):
    """Content-area BGR image (any size) -> work image (WORK_W wide, aspect kept)."""
    h, w = content_bgr.shape[:2]
    if w == WORK_W:
        return content_bgr
    return cv2.resize(content_bgr, (WORK_W, round(h * WORK_W / w)), interpolation=cv2.INTER_AREA)


def box_px(box, img):
    """Normalized box -> (x0, y0, x1, y1) pixel bounds in img, clipped to the image."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return (max(0, round(x0 * w)), max(0, round(y0 * h)), min(w, round(x1 * w)), min(h, round(y1 * h)))


def cut(img, box):
    x0, y0, x1, y1 = box_px(box, img)
    return img[y0:y1, x0:x1]


@dataclass(frozen=True)
class Match:
    name: str
    score: float                 # best correlation in the search region, -1..1
    found: bool                  # score >= the template's threshold
    center: Optional[tuple]      # normalized (x, y) of the best match


@dataclass(frozen=True)
class Probe:
    name: str
    label: Optional[str]         # nearest class, or None if no class is within max_dist
    bgr: tuple                   # measured mean colour
    dist: float                  # distance to the nearest class


class Templates:
    """Loads templates once and matches them. `spec` defaults to data/templates.json."""

    def __init__(self, spec=SPEC, root=DATA / "templates"):
        self.spec, self.root = spec, Path(root)
        self._cache = {}

    @property
    def names(self):
        return list(self.spec["templates"])

    def image(self, name):
        if name not in self._cache:
            img = cv2.imread(str(self.root / f"{name}.png"), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"template {name}: run scripts/make_templates.py")
            self._cache[name] = img
        return self._cache[name]

    def search_box(self, name):
        t = self.spec["templates"][name]
        if "search" in t:
            return t["search"]
        m = t.get("margin", self.spec["_meta"]["margin"])
        x0, y0, x1, y1 = t["box"]
        return [x0 - m, y0 - m, x1 + m, y1 + m]

    def match(self, img, name) -> Match:
        img = to_work(img)
        t, tpl = self.spec["templates"][name], self.image(name)
        sx0, sy0, sx1, sy1 = box_px(self.search_box(name), img)
        region = img[sy0:sy1, sx0:sx1]
        th, tw = tpl.shape[:2]
        if region.shape[0] < th or region.shape[1] < tw:
            return Match(name, -1.0, False, None)
        res = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (bx, by) = cv2.minMaxLoc(res)
        score = float(np.nan_to_num(score, nan=-1.0))  # flat region -> NaN
        h, w = img.shape[:2]
        center = ((sx0 + bx + tw / 2) / w, (sy0 + by + th / 2) / h)
        return Match(name, score, score >= t["threshold"], center)

    def match_all(self, img, names=None):
        img = to_work(img)
        return {n: self.match(img, n) for n in (names or self.names)}

    def probe(self, img, name) -> Probe:
        p = self.spec["probes"][name]
        bgr = cut(to_work(img), p["box"]).reshape(-1, 3).mean(0)
        dists = {k: float(np.linalg.norm(bgr - np.array(v))) for k, v in p["classes"].items()}
        label = min(dists, key=dists.get)
        d = dists[label]
        return Probe(name, label if d <= p["max_dist"] else None, tuple(round(float(c), 1) for c in bgr), d)

    def button_state(self, img):
        """'idle' / 'ff_off' / 'ff_on' / None, from the play-button templates + FF colour probe."""
        img = to_work(img)
        play, ff = self.match(img, "btn_play"), self.match(img, "btn_ff")
        if play.found and play.score >= ff.score:
            return "idle" if self.probe(img, "play_button").label == "idle" else None
        if ff.found:
            label = self.probe(img, "play_button").label
            return {"ff_off": "ff_off", "ff_on": "ff_on"}.get(label)
        return None


_default = None


def _templates():
    global _default
    if _default is None:
        _default = Templates()
    return _default


def match(img, name) -> Match:
    return _templates().match(img, name)


def match_all(img, names=None):
    return _templates().match_all(img, names)


def probe(img, name) -> Probe:
    return _templates().probe(img, name)


def button_state(img):
    return _templates().button_state(img)
