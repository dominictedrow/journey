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
DEFENSE_TIERS = 3          # defence towers' tier budget at first (+1 each time lives drop, up to 7):
                           # cash in the defence is XP the targets don't get
DEFENSE_START, DEFENSE_MAX = 2, 4    # defence towers of each kind: at first, and at most


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
        self.defense_count = DEFENSE_START   # raised by play_game when lives drop
        self.defense_tiers = DEFENSE_TIERS

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
        """(cost, tid, path) for the next tier of every tower of `kinds`, cheapest first.
        Defence towers (not targets) stop at DEFENSE_TIERS tiers."""
        s = self.h.guard_state()
        out = []
        for t in self.h.registry.values():
            if t.kind not in kinds:
                continue
            if t.kind not in self.targets and sum(t.tiers) >= self.defense_tiers:
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
        for kind in self.defense:                 # the defence: defense_count of each, in order
            if len(self._placed(kind)) < self.defense_count and kind in h.towers:
                if self._try_place(kind):
                    return True
                if guards.can_buy_tower(kind, h.guard_state()).reason == "no_money":
                    return False
        if self._try_upgrade(set(self.targets) | set(self.defense)):
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
    lives = None
    while time.monotonic() - t0 < timeout_s:
        s = h.observe()
        if s.outcome:
            return s.outcome
        if s.screen == "game":
            unknown = 0
            if lives is not None and s.lives is not None and s.lives < lives:
                spender.defense_count = min(DEFENSE_MAX, spender.defense_count + 1)
                spender.defense_tiers = min(7, spender.defense_tiers + 1)
                log(f"lives {lives} -> {s.lives}: defence {spender.defense_count} of each, "
                    f"{spender.defense_tiers} tiers")
            lives = s.lives if s.lives is not None else lives
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
                if unknown == 3:          # most unknown overlays (reward splashes) go on a click
                    log(f"unknown screen {s.screen}: click")
                    h.game.click(0.5, 0.5)
                elif unknown >= 6:
                    log(f"stuck on screen {s.screen}: Escape")
                    h.game.key("escape")
                    unknown = 0
        time.sleep(tick_s)
    return "timeout"


# --- planning (docs/max.md B2) ---------------------------------------------------------------
def card_unlocked(img, cell_xy, probe_px):
    """Is a Monkeys-menu card coloured (tower unlocked) rather than grey?"""
    import cv2
    H, W = img.shape[:2]
    k = W / 1920
    x, y = cell_xy[0] * W, cell_xy[1] * H
    x0, y0, x1, y1 = probe_px
    patch = img[int(y + y0 * k):int(y + y1 * k), int(x + x0 * k):int(x + x1 * k)]
    _, s, v = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(0)
    return s >= 60 and v >= 165


def survey(h, log=print):
    """Visit the Monkeys menu: -> {kind: {'unlocked_tower': bool, 'xp', 'unlocked', 'next_state'}}.
    Only unlocked towers' XP screens are opened (a locked card may offer a paid unlock)."""
    from . import xp
    mm = h.ui["monkeys_menu"]
    out = {}
    if not h.go_home():
        return out
    h._click_ui("monkeys_menu.open")
    for tab in mm["tabs"]:
        h.game.click(*mm["tabs"][tab])
        h.sleep(1.0)
        for kind, cell in mm["cells"].items():
            if cell["tab"] != tab:
                continue
            if not card_unlocked(h.content(), cell["xy"], mm["card_probe_px"]):
                out[kind] = {"unlocked_tower": False}
                continue
            h.game.click(*cell["xy"])
            h.sleep(1.5)
            if h.screen_kind() != "xp_screen":
                log(f"survey: {kind} didn't open its XP screen")
                out[kind] = {"unlocked_tower": True}
                h.go_home()
                h._click_ui("monkeys_menu.open")
                h.game.click(*mm["tabs"][tab])
                h.sleep(1.0)
                continue
            s = xp.read_settled(h)
            out[kind] = {"unlocked_tower": True, "xp": s.xp, "unlocked": list(s.unlocked),
                         "next_state": list(s.next_state)}
            h._click_ui("xp_screen.back")
            h.sleep(0.8)
    return out


def placeable_here(h, kind):
    """Can `kind` go anywhere on this map (water towers need water masks)?"""
    masks = h.placeable_masks()
    return any(c in masks for c in h.mask_classes_of(kind))


def choose_targets(h, progress, n=2, exclude=()):
    """The `n` unlocked, placeable tower types with the most tiers still locked."""
    cand = []
    for kind, p in progress.items():
        if not p.get("unlocked_tower") or kind in exclude or "unlocked" not in p:
            continue
        left = 15 - sum(p["unlocked"])
        if left > 0 and placeable_here(h, kind):
            cand.append((-left, kind))
    return [k for _, k in sorted(cand)[:n]]
