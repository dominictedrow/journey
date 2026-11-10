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
    if s >= 230 and v >= 215 and 15 <= h <= 30:
        return "gold"                   # tier 5 once tier 4 is unlocked (affordable or not)
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
    votes.pop("dull", None)             # icon art abstains; locked icons are greyed out whole
    best = max(votes, key=votes.get, default="dull")
    return best if votes.get(best, 0) >= 2 else "dull"


@dataclass(frozen=True)
class XpScreen:
    xp: int | None
    unlocked: tuple          # tiers unlocked per path, e.g. (2, 1, 1)
    next_state: tuple        # per path: 'afford' / 'poor' / 'tier5' (gold: compare the cost with xp)
                             # / None (path complete or unclear)


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
        nxt.append(None if n == 5 else {"green": "afford", "red": "poor", "gold": "tier5"}.get(states[n]))
    return XpScreen(read_number(glyphs.read(_crop(img, X["xp"]), "bright")), tuple(unlocked), tuple(nxt))


def unlock_cost(img):
    return read_number(glyphs.read(_crop(img, X["unlock_cost"])))


def read_settled(h, tries=6):
    """read() once two reads in a row agree (icons animate for a moment after an unlock)."""
    last = read(h.content())
    for _ in range(tries):
        h.sleep(0.35)
        now = read(h.content())
        if now == last:
            return now
        last = now
    return last


def select(h, path, tier):
    """Click an icon, then close the description tooltip it may open (it covers the top row
    and swallows the next click)."""
    h.game.click(X["cols_x"][tier], X["rows_y"][path])
    h.sleep(0.4)
    h.game.click(*X["dismiss"])
    h.sleep(0.4)


def unlock_next(h, path):
    """On the open XP screen: select the next tier of `path` and unlock it. -> (ok, cost)."""
    before = read_settled(h)
    tier = before.unlocked[path]
    if tier >= 5 or before.next_state[path] not in ("afford", "tier5"):
        return False, None
    select(h, path, tier)
    cost = unlock_cost(h.content())
    h.game.click(*X["unlock_button"])
    h.sleep(1.0)
    after = read_settled(h)
    return after.unlocked[path] == tier + 1, cost


def unlock_all_affordable(h, log=print, max_steps=20):
    """Unlock tiers on the open screen, cheapest next tier first, while XP lasts. A path that
    fails twice is left for next time. -> [(path, cost)]."""
    done, fails = [], {}
    for _ in range(max_steps):
        s = read_settled(h)
        options = [p for p in range(3) if s.next_state[p] in ("afford", "tier5") and fails.get(p, 0) < 2]
        costs = {}
        for p in options:                       # select each candidate to see its price
            select(h, p, s.unlocked[p])
            costs[p] = unlock_cost(h.content()) or 10 ** 9
        costs = {p: c for p, c in costs.items() if s.xp is None or c <= s.xp}
        if not costs:
            break
        p = min(costs, key=costs.get)
        ok, cost = unlock_next(h, p)
        log(f"unlock path {p} tier {s.unlocked[p] + 1} for {cost} XP: {'ok' if ok else 'failed'}")
        if ok:
            done.append((p, cost))
        else:
            fails[p] = fails.get(p, 0) + 1
    return done
