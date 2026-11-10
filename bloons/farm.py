"""XP farm logic (docs/max.md Part B), portable: one game's play policy on top of the harness.

Tower XP from a round is split between tower types by the cash invested in each (heroes
don't count; RESEARCH 2026-11-09). So a game is: a small defence first (`defense`: kinds in
build order, the hero helps for free XP-wise), then every dollar into the `targets` (the
types whose upgrades we're unlocking): upgrade the cheapest unlocked tier on a target tower,
else place another target tower on the best free spot.

Spots: the map's placeable cells for that tower, scored by how much of the bloon path lies in
the tower's range (ranges are ellipses on screen: harness.ground_unit()).
"""
import time

import numpy as np

from . import guards, rules

PATH_STEP = 0.004          # normalized spacing of path samples for spot scores
MIN_COVER = 3              # a spot must see at least this many path samples


def path_samples(h):
    pl = np.asarray(h.map["path"]["polyline"], float)
    pts = []
    for a, b in zip(pl[:-1], pl[1:]):
        n = max(1, int(np.hypot(*(b - a)) / PATH_STEP))
        pts.extend(a + (b - a) * (k / n) for k in range(n))
    return np.array(pts)


def best_spot(h, kind, tiers=(0, 0, 0), exclude=()):
    """(x, y) of the free placeable cell for `kind` that covers the most path, or None."""
    rng = rules.tower_range(h.towers[kind], tiers)[0] if kind in h.towers else 40.0
    rng = rng or 30.0
    gu = np.asarray(h.ground_unit)
    if not hasattr(h, "_farm_path"):
        h._farm_path = path_samples(h)
    pts = h._farm_path / gu
    masks = h.placeable_masks()
    best, score = None, MIN_COVER - 1
    for cls in h.mask_classes_of(kind):
        m = masks.get(cls)
        if m is None:
            continue
        g = m.grid
        jj, ii = np.nonzero(m.ok)
        for i, j in zip(ii, jj):
            x, y = g.x0 + i * g.dx, g.y0 + j * g.dy
            if (round(x, 4), round(y, 4)) in exclude:
                continue
            d = np.hypot(*(pts - np.array([x, y]) / gu).T)
            s = int((d <= rng).sum())
            if s > score:
                best, score = (x, y), s
    return best


class Spender:
    """Spends a game's money: defence first, then the targets. Remembers what the game refused
    (XP-locked tiers, blocked spots) so it doesn't retry them every tick."""

    def __init__(self, h, targets, defense=(), log=print):
        self.h, self.targets, self.defense, self.log = h, list(targets), list(defense), log
        self.xp_locked = set()        # (kind, path, tier) the account hasn't unlocked
        self.bad_spots = set()
        self.place_fail = {}          # kind -> consecutive placement failures

    def _placed(self, kind):
        return [t for t in self.h.registry.values() if t.kind == kind]

    def _try_place(self, kind):
        if self.place_fail.get(kind, 0) >= 3 or not guards.can_buy_tower(kind, self.h.guard_state()):
            return False
        for _ in range(12):            # spots the harness rules out without touching the game
            xy = best_spot(self.h, kind, exclude=self.bad_spots)
            if xy is None:
                self.place_fail[kind] = 99
                return False
            r = self.h.place_tower(kind, *xy)
            if r.reason == "no_money":
                return False
            self.log(f"place {kind} at ({xy[0]:.3f}, {xy[1]:.3f}): {r.reason}")
            if r:
                self.place_fail[kind] = 0
                return True
            self.bad_spots.add((round(xy[0], 4), round(xy[1], 4)))
            if r.reason in ("blocked", "unconfirmed", "no_hold", "not_front"):   # the game said no
                self.place_fail[kind] = self.place_fail.get(kind, 0) + 1
                return False
        return False

    def _upgrade_options(self, kinds):
        """(cost, tid, path) for the next tier of every tower of `kinds`, cheapest first."""
        s = self.h.guard_state()
        out = []
        for t in self.h.registry.values():
            if t.kind not in kinds:
                continue
            for p in range(3):
                if (t.kind, p, t.tiers[p] + 1) in self.xp_locked:
                    continue
                g = guards.can_upgrade(t.tid, p, s)
                if g or g.reason == "no_money":
                    out.append((g.cost or 0, t.tid, p))
        return sorted(out)

    def _try_upgrade(self, kinds):
        money = self.h.money or 0
        for cost, tid, p in self._upgrade_options(kinds):
            if cost > money:
                return False
            t = self.h.registry[tid]
            r = self.h.upgrade(tid, p)
            self.log(f"upgrade {t.kind}#{tid} path {p}: {r.reason}")
            if r:
                return True
            if r.reason in ("xp_locked", "panel_locked"):
                self.xp_locked.add((t.kind, p, t.tiers[p] + 1))
                continue
            return False
        return False

    def step(self):
        """One purchase if anything is affordable. -> True if something was bought."""
        h = self.h
        if h.hero_choice and h.hero is None and guards.can_place_hero(h.guard_state()):
            xy = best_spot(h, "hero", exclude=self.bad_spots)
            if xy:
                r = h.place_hero(*xy)
                self.log(f"place hero: {r.reason}")
                if r:
                    return True
                self.bad_spots.add((round(xy[0], 4), round(xy[1], 4)))
        for kind in self.defense:                 # the defence: one of each, in order
            if not self._placed(kind) and kind in h.towers:
                if self._try_place(kind):
                    return True
                if guards.can_buy_tower(kind, h.guard_state()).reason == "no_money":
                    return False
        if self._try_upgrade(set(self.targets)):
            return True
        for kind in self.targets:
            if self._try_place(kind):
                return True
        return False

    def spend(self, max_actions=6):
        n = 0
        while n < max_actions and self.step():
            n += 1
        return n


def play_game(h, spender, log=print, timeout_s=3600, tick_s=1.0):
    """Play the current game to its end. -> 'win' | 'loss' | 'timeout'. Starts rounds (Auto
    Start then carries on), clears prompts, spends between observations."""
    t0 = time.monotonic()
    started = False
    unknown = 0
    while time.monotonic() - t0 < timeout_s:
        s = h.observe()
        if s.outcome:
            return s.outcome
        if s.screen == "game":
            unknown = 0
            spender.spend()
            if not started or (s.round_active is False and not s.round_ended):
                r = h.start_round()
                started = started or bool(r)
        elif s.screen in ("victory", "defeat"):
            continue
        else:
            done = h.handle_prompts()
            if not done:
                unknown += 1
                if unknown >= 5:
                    log(f"stuck on screen {s.screen}: Escape")
                    h.game.key("escape")
                    unknown = 0
        time.sleep(tick_s)
    return "timeout"
