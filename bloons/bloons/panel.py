"""Upgrade panel reader (PLAN 5.3, portable): what the open panel says about each path, to
check the price table and the guards against the game.

Each path row shows, from left to right, 5 tier pips (filled = owned), the current upgrade
and the next tier's buy button. The button is:
  green, white price       -> 'buy'    (affordable)
  grey-blue, red price     -> 'poor'   (not enough money)
  grey-blue, MAX UPGRADES  -> 'max'    (tier 5, the crosspath cap, or the type's tier 5 taken)
and a path that can't be started any more (two others upgraded) shows a brown row with a
padlock and PATH CLOSED instead -> 'closed'. Anything else (an overlay, a black test frame)
reads 'unknown' and is ignored by the checks.

The state comes from the button's colour, the price from OCR (an `ocr(box) -> str`
callable, so tests can fake it), the tier from the pips. Boxes are in data/ui.json
`upgrade_panel.<side>.path_button_box / path_prices / path_pips` (+ `pip_dy`).
"""
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from . import guards, ui
from .ui import UI

# HSV bands (OpenCV: H 0-180), measured on Dart Monkey panels (Sandbox + Standard, 2026-11-05).
# Button interiors: green 0.39-0.69 of the box (the upgrade's icon covers part of it, most on
# Plasma Monkey Fan Club), grey-blue 0.65-0.84, closed rows 1.0 brown; the runner-up colour
# is <= 0.10 everywhere.
GREEN = ((35, 120, 120), (75, 255, 255))
GREY = ((95, 30, 110), (125, 140, 255))
BROWN = ((5, 80, 60), (25, 230, 230))     # the panel's wood, what a closed row shows instead
MIN_FRAC = 0.25
PIP_HUE = (25, 50)                         # filled pips: yellow-green, saturation ~254
PIP_MIN_SAT = 220                          # empty pips are brown (hue 12-16, saturation <= 186)


@dataclass(frozen=True)
class PathRow:
    state: str              # 'buy' | 'poor' | 'max' | 'closed' | 'unknown'
    price: Optional[int]    # the button's price ('buy' / 'poor' with a readable price), else None
    tier: int               # filled pips


def parse_price(text):
    """'$1,530' -> 1530; None if there's no number (e.g. 'MAX UPGRADES' or '')."""
    m = re.search(r"\$?\s*(\d[\d,.]*)", text or "")
    if not m:
        return None
    digits = re.sub(r"\D", "", m.group(1))
    return int(digits) if digits else None


def _crop(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(h * y0):max(int(h * y1), int(h * y0) + 1), int(w * x0):max(int(w * x1), int(w * x0) + 1)]


def _frac(hsv, band):
    lo, hi = band
    return float(cv2.inRange(hsv, np.array(lo), np.array(hi)).mean() / 255) if hsv.size else 0.0


def button_colour(img, box):
    """'green' / 'grey' / 'brown' / None for the inside of a path's buy button."""
    hsv = cv2.cvtColor(_crop(img, box), cv2.COLOR_BGR2HSV)
    fr = {"green": _frac(hsv, GREEN), "grey": _frac(hsv, GREY), "brown": _frac(hsv, BROWN)}
    best = max(fr, key=fr.get)
    return best if fr[best] >= MIN_FRAC else None


def pip_tier(img, point, dy):
    """How many of a path's 5 pips are filled (tier 1 at `point`, tier t at dy * (t - 1))."""
    h, w = img.shape[:2]
    r = max(1, int(0.003 * w))
    n = 0
    for t in range(5):
        x, y = int(w * point[0]), int(h * (point[1] + dy * t))
        px = cv2.cvtColor(img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1], cv2.COLOR_BGR2HSV)
        hue, sat = float(px[..., 0].mean()), float(px[..., 1].mean())
        n += PIP_HUE[0] <= hue <= PIP_HUE[1] and sat >= PIP_MIN_SAT
    return n


def read_paths(img, side, ocr, paths=(0, 1, 2)):
    """The open panel's path rows on `side` of a content image (BGR, any scale) -> {path: PathRow}.
    `ocr(box)` reads the text in a normalized box; it's only called for rows with a button
    showing a price (one call per such row). `ocr=None`: colours and pips only (no prices;
    a grey button then reads 'poor')."""
    p = ui.panel(side)
    dy = UI["upgrade_panel"]["pip_dy"]
    out = {}
    for i in paths:
        tier = pip_tier(img, p["path_pips"][i], dy)
        colour = button_colour(img, p["path_button_box"][i])
        price = None
        if colour == "green":
            state, price = "buy", (parse_price(ocr(p["path_prices"][i])) if ocr else None)
        elif colour == "grey":
            price = parse_price(ocr(p["path_prices"][i])) if ocr else 0
            state = "max" if price is None else "poor"
            price = price or None
        elif colour == "brown":
            state = "closed"
        else:
            state = "unknown"
        out[i] = PathRow(state, price, tier)
    return out


def expected(t, path, s):
    """What the panel should show for tower record `t` on `path` per the guards:
    (state class 'open' / 'max' / 'closed', price or None, tier)."""
    r = guards.can_upgrade(t.tid, path, s)
    tier = t.tiers[path]
    if r.reason in ("max_tier", "crosspath", "tier5_taken"):
        return ("closed" if tier == 0 else "max"), None, tier
    return "open", guards.upgrade_cost(t, path, s), tier


def compare(t, rows, s):
    """Disagreements between the panel rows of tower record `t` and the guards (GuardState
    `s`, with the registry) -> {path: [str, ...]} for the paths that disagree. Affordability
    isn't compared (the money moves); 'unknown' rows aren't compared at all."""
    bad = {}
    for path, row in rows.items():
        if row.state == "unknown":
            continue
        want_state, want_price, want_tier = expected(t, path, s)
        got_state = "open" if row.state in ("buy", "poor") else row.state
        msgs = []
        if got_state != want_state:
            msgs.append(f"state {row.state} != {want_state}")
        elif want_price is not None and row.price is not None and row.price != want_price:
            msgs.append(f"price {row.price} != {want_price}")
        if row.tier != want_tier:
            msgs.append(f"tier {row.tier} != {want_tier}")
        if msgs:
            bad[path] = msgs
    return bad
