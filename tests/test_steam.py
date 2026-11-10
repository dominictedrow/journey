"""Offline tests on Steam (Windows) fixtures. Run on their own: pytest tests/test_steam.py
(the calibration is chosen at import, and tests/test_offline.py pins the Mac one)."""
import json
import os
import sys
from pathlib import Path

os.environ["BLOONS_CALIB"] = "steam"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import pytest  # noqa: E402

from bloons import calib  # noqa: E402

pytestmark = pytest.mark.skipif(calib.NAME != "steam", reason="run tests/test_steam.py on its own")
FIX = Path(__file__).resolve().parents[1] / "bloons/data/fixtures/win"


def img(name):
    return cv2.imread(str(FIX / name))


def test_screens_and_button():
    from bloons.state import read_screen
    from bloons.vision import Templates
    spec = json.loads((FIX / "screens.json").read_text())
    for name, kind in spec["screens"].items():
        assert read_screen(img(f"s_{name}.jpg")).kind == kind, name
    t = Templates()
    for name, state in spec["buttons"].items():
        assert t.button_state(img(f"s_{name}.jpg")) == state, name


def test_panel_rows():
    from bloons import glyphs, panel
    cases = {
        ("panel_bomb_mixed.png", "right"): [("xp_locked", 1000, 1), ("closed", None, 0), ("xp_locked", 600, 1)],
        ("panel_bomb_locked.png", "right"): [("xp_locked", 200, 0), ("xp_locked", 180, 0), ("xp_locked", 150, 0)],
        ("panel_right_dart.png", "left"): [("buy", 120, 0), ("buy", 85, 0), ("buy", 75, 0)],
    }
    for (name, side), want in cases.items():
        im = img(name)
        h, w = im.shape[:2]
        crop = lambda b: im[int(b[1] * h):int(b[3] * h), int(b[0] * w):int(b[2] * w)]  # noqa: E731
        rows = panel.read_paths(im, side, lambda b: glyphs.read(crop(b)))
        assert [(r.state, r.price, r.tier) for r in rows.values()] == want, name


def test_xp_screen():
    from bloons import xp
    assert xp.read(img("xp_bomb_before.jpg")) == xp.XpScreen(1465, (0, 0, 0), ("afford",) * 3)
    assert xp.read(img("xp_bomb_3.jpg")) == xp.XpScreen(935, (1, 1, 1), ("poor", "afford", "afford"))
    assert xp.read(img("xp_bomb_4.jpg")) == xp.XpScreen(285, (1, 2, 1), ("poor",) * 3)
    assert xp.unlock_cost(img("xp_bomb_4.jpg")) == 2700
