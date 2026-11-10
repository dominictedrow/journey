"""The upgrade-unlock (XP) screen, Steam (docs/max.md B2). Portable: works on content images.

`read(img)` -> XpScreen: the tower's banked XP and, per path, how many tiers are unlocked
and what the next one shows ('afford' / 'poor' / None when the path is done or unreadable).
`unlock_next(h, path)` buys the next tier of a path with XP on the open screen.
Layout in ui.json `xp_screen`.
"""
import re
from dataclasses import dataclass

import cv2
import numpy as np

from . import glyphs
from .ui import UI

X = UI.get("xp_screen", {})


def ring_class(hsv_px):
    h, s, v = hsv_px
    if s >= 180 and v >= 150 and 92 <= h <= 110:
        return "blue"
    if s >= 180 and v >= 150 and (h <= 10 or h >= 170):
        return "red"
    if s >= 150 and v >= 130 and 35 <= h <= 85:
        return "green"
    return "dull"


def icon_state(img, col, row):
    """'blue' (unlocked) / 'green' (next, affordable) / 'red' (next, too few XP) / 'dull'."""
    H, W = img.shape[:2]
    cx, cy = X["cols_x"][col] * W, X["rows_y"][row] * H
    k = W / 1920
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    votes = {}
    for dx, dy in X["samples_px"]:
        x, y = int(cx + dx * k), int(cy + dy * k)
        c = ring_class(hsv[y - 3:y + 4, x - 3:x + 4].reshape(-1, 3).mean(0))
        votes[c] = votes.get(c, 0) + 1
    best = max(votes, key=votes.get)
    return best if votes[best] >= 2 else "dull"


@dataclass(frozen=True)
class XpScreen:
    xp: int | None
    unlocked: tuple          # tiers unlocked per path, e.g. (2, 1, 1)
    next_state: tuple        # per path: 'afford' / 'poor' / None (path complete or unclear)


def _crop(img, box):
    H, W = img.shape[:2]
    return img[int(box[1] * H):int(box[3] * H), int(box[0] * W):int(box[2] * W)]


def read_number(text):
    m = re.findall(r"\d+", text or "")
    return int("".join(m)) if m else None


def read(img):
    unlocked, nxt = [], []
    for row in range(3):
        states = [icon_state(img, c, row) for c in range(5)]
        n = 0
        while n < 5 and states[n] == "blue":
            n += 1
        unlocked.append(n)
        nxt.append(None if n == 5 else {"green": "afford", "red": "poor"}.get(states[n]))
    return XpScreen(read_number(glyphs.read(_crop(img, X["xp"]), "bright")), tuple(unlocked), tuple(nxt))


def unlock_cost(img):
    return read_number(glyphs.read(_crop(img, X["unlock_cost"])))


def unlock_next(h, path, sleep=None):
    """On the open XP screen: select the next tier of `path` and unlock it. -> (ok, cost)."""
    sleep = sleep or h.sleep
    before = read(h.content())
    tier = before.unlocked[path]
    if tier >= 5 or before.next_state[path] != "afford":
        return False, None
    pt = (X["cols_x"][tier], X["rows_y"][path])
    h.game.click(*pt)
    sleep(0.6)
    cost = unlock_cost(h.content())
    h.game.click(*X["unlock_button"])
    sleep(1.2)
    after = read(h.content())
    return after.unlocked[path] == tier + 1, cost


def unlock_all_affordable(h, log=print, max_steps=15):
    """Unlock tiers on the open screen, cheapest next tier first, while XP lasts. -> [(path, cost)]."""
    done = []
    for _ in range(max_steps):
        s = read(h.content())
        options = [p for p in range(3) if s.next_state[p] == "afford"]
        if not options:
            break
        costs = {}
        for p in options:                       # select each candidate to see its price
            h.game.click(X["cols_x"][s.unlocked[p]], X["rows_y"][p])
            h.sleep(0.5)
            costs[p] = unlock_cost(h.content()) or 10 ** 9
        p = min(costs, key=costs.get)
        ok, cost = unlock_next(h, p)
        log(f"unlock path {p} tier {s.unlocked[p] + 1} for {cost} XP: {'ok' if ok else 'failed'}")
        if not ok:
            break
        done.append((p, cost))
    return done
