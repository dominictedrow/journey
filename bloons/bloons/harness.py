"""BloonsHarness: the action/observation surface for a person or an RL loop (PLAN Phase 4).

Portable: `game` is anything with platform_mac.Game's methods (grab, read_hud, click, move,
key, scroll), so offline tests pass a fake. Coordinates are normalized content coords.

Step 4.1 is the skeleton the actions build on:
- data: towers / heroes / account / ui.json, the map file (scale, path keep-out) and its
  learned-spots memory (<map>.spots.json);
- the tower registry (guards.TowerRec per tid, plus the hero) and what the game has seen
  bought this episode;
- `guard_state()` / `place_context()` / `available_actions()` for guards.py and
  placement.can_place (Dom's action mask is built from `available_actions`);
- input: `press(action)` uses a confirmed hotkey (ui.json `hotkeys`, towers.json `hotkey`)
  or else clicks the UI point; `select_shop(kind)` picks a tower or the hero from the shop,
  scrolling it for the towers without a hotkey.

Actions (4.2+) call their guard first and return its refusal unchanged, without touching the
game, and return ActionResult rather than raising on game-side failures.
"""
import json
import time
from pathlib import Path

from . import abilitybar, guards, panel, rangecircle, rules, ui
from .hud import HudFilter
from .placement import (PathKeepout, PlaceContext, Placed, SpotMemory, can_place, load_masks, mask_classes,
                        footprint_radius, record_placement, shape_of, tint_at, unit_norm)
from .result import ActionResult
from .state import RoundTracker

DATA = Path(__file__).parent / "data"
SCROLL_LINES = 200        # past the shop's full length (~30 events x 5 lines, RESEARCH 3.1)
HERO_TID = "hero"         # the hero's id in place contexts and can_sell
PARK = (0.92, 0.5)        # cursor rest point over the shop: off the map, for the before-hold frame
SETTLE_S = 0.08           # after moving a held tower: the tint follows within ~20 ms (3.2)
CONFIRM_TRIES = 4         # HUD reads to see the money change after placing / selling
SYNC_FRESH_S = 1.0        # observe() this recent: one button read agreeing with the tracker will do
MONEY_FRESH_S = 0.5       # HUD money read this recent: _money_before uses the filtered value
REFUND_INCOME_SLACK = 300  # $ of pop income that can land on top of a sell refund while confirming
TOWER_SPRITE_PAD = 4.0    # units past a tower's footprint that the bloon detector leaves out


def _read_panel(frame):
    from .state import read_screen
    return None if frame is None else read_screen(frame).panel


def _read_kind(frame):
    from .state import read_screen
    return None if frame is None else read_screen(frame).kind


def _read_screen(frame):
    from .state import Screen, read_screen
    return Screen("unknown") if frame is None else read_screen(frame)


def _read_button(frame):
    from .vision import button_state
    return None if frame is None else button_state(frame)


def _read_probe(name):
    def read(frame):
        from .vision import probe
        return None if frame is None else probe(frame, name).label
    return read


def _icon_diff(a, b):
    import numpy as np
    if a is None or b is None or a.shape != b.shape or a.size == 0:
        return 0.0
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def load_data(name):
    """Parse bloons/data/<name> (JSON)."""
    return json.loads((DATA / name).read_text())


class BloonsHarness:
    def __init__(self, game, map_name, difficulty="easy", hero=None, account=True, sleep=time.sleep,
                 spots_path=None, mode="standard"):
        """`mode`: 'standard' or 'sandbox' (rounds are sent from the Sandbox bloon panel).
        `hero`: heroes.json key chosen for this game (placeable once), or None.
        `account`: apply data/account.json Monkey Knowledge to prices (False = none).
        `sleep`: injectable for tests. `spots_path`: learned-spots file (default
        data/maps/<map>.spots.json; tests and trial runs point it elsewhere)."""
        if mode not in ("standard", "sandbox"):
            raise ValueError(mode)
        self.game = game
        self.mode = mode
        self.map_name = map_name
        self.difficulty = difficulty
        self.hero_choice = hero
        self.sleep = sleep
        self.ui = ui.UI
        self.towers = load_data("towers.json")["towers"]
        self.n_slots = load_data("obs.json")["tower_slots"]   # tower slots in observe() / to_vector
        self.heroes = load_data("heroes.json")["heroes"]
        ha = load_data("hero_abilities.json")
        self.hero_abilities, self.default_cooldown = ha["heroes"], ha["_meta"]["default_cooldown_s"]
        self.clock = time.monotonic     # ability cooldown estimates; tests swap it
        self.account = load_data("account.json") if account else None
        if hero is not None and hero not in self.heroes:
            raise KeyError(hero)
        self._set_map(map_name, spots_path)
        self.hud = HudFilter()
        self.tint = tint_at       # (base, held, x, y) -> 'ok' | 'blocked' | 'none'; tests swap it
        self.read_panel = _read_panel   # content frame -> 'left' | 'right' | None; tests swap it
        self.read_kind = _read_kind     # content frame -> state.Screen kind ('game', 'pause', ...)
        self.read_screen = _read_screen  # content frame -> state.Screen (kind, button, panel)
        self.read_button = _read_button  # -> 'idle' / 'ff_off' / 'ff_on' / None
        self.read_sandbox_panel = _read_probe("sandbox_panel")    # -> 'shop' / 'bloons' / None
        self.icon_diff = _icon_diff     # (before, after) slot crops -> mean |diff|; tests swap it
        self.read_bar = abilitybar.read_bar   # (content, n_slots, panel_side) -> [SlotRead]; 5.4
        self.read_range = rangecircle.measure  # (selected, deselected, centre, unit, exclude) -> RangeRead; 5.5
        self.bar_filter = abilitybar.BarFilter()
        self.read_targeting = self._ocr_targeting   # panel side -> label text ('' if none)
        self.read_hero_price = self._ocr_hero_price  # panel side -> level-up button text
        self.read_paths = self._read_paths          # (side, paths, prices) -> {path: panel.PathRow}
        self.read_map_names = self._ocr_map_names   # map select page -> 6 thumbnail names
        self.pick_trace = []                        # last _pick_map reads / flips / clicks
        self.prompt_actions = {k: v for k, v in self.ui["prompts"].items()
                               if not k.startswith("_") and k not in ("max_steps", "after_click_s")}
        self.tracker = RoundTracker()
        self.rounds = {r["round"]: r for r in load_data("rounds.json")["rounds"]}
        self.new_episode()

    def _set_map(self, map_name, spots_path=None):
        """Load a map's data: scale, path keep-out and learned-spots memory."""
        self.map_name = map_name
        self.map = load_data(f"maps/{map_name}.json")
        self.unit = unit_norm(self.map)
        self.keepout = PathKeepout.from_json(self.map["path"], self.unit) if "path" in self.map else None
        self.masks = load_masks(self.map) if self.map.get("masks") else {}
        self._masks_cache = None
        from .detect import BloonDetector, load_background
        self.detector = BloonDetector(self.map) if "path" in self.map else None
        bg = load_background(map_name) if self.detector is not None else None
        if bg is not None:
            self.detector.set_background_arrays(*bg)
        self.memory = SpotMemory.load(spots_path or DATA / "maps" / f"{map_name}.spots.json")
        if not self.memory.meta:
            self.memory.meta = {"map": map_name, "game_version": self.map["_meta"]["game_version"]}

    # --- episode state ----------------------------------------------------------------------
    def new_episode(self):
        """Forget everything about the current game (call when a new game starts)."""
        self.registry = {}            # tid -> guards.TowerRec
        self.hero = None              # guards.HeroRec once placed
        self.bought = set()           # tower kinds bought at least once this game
        # Ability bar slots in order of first gain. The game groups identical abilities into one
        # icon (with a ready-count badge), so a slot is an ability name with its instances:
        # {name, cooldown, between_rounds, inst: {(tid, path): ready_at}}.
        self.ability_slots = []
        if hasattr(self, "bar_filter"):
            self.bar_filter.reset()
        self.round_active = False     # Standard: set by start_round; the round loop clears it
        self.fast_forward = None      # last FF state read or set
        self.locked = set()          # tower kinds not available (mode / unlocks)
        self.shop_layout = None       # 'top' / 'bottom' once we've scrolled the shop
        self._next_tid = 1
        self.slots = [None] * self.n_slots   # tid per tower slot: first free one on place, freed on sell
        self.hud.reset()
        self.tracker.reset()
        self._masks_cache = None      # (placed key, {class: Mask}) for observe()
        self.panel_checks = {}        # tid -> (tiers, {path: agreed?}) from the last panel reads
        self.price_mismatches = []    # (kind, tiers, path, panel price, table price) seen buying
        self.range_checks = {}        # tid -> (tiers, measured range in units) from measure_range
        self.range_mismatches = []    # (kind, tiers, measured, table) off by more than the tolerance
        self.rounds_done = 0          # rounds finished this game, counted by observe() (the HUD
                                      # number is ambiguous between rounds; RESEARCH step 2.x)
        self._xp_round = 0            # last round whose hero XP was credited (no double credit)
        self._hero_prev = False       # a hero was placed at the previous observe() (6: XP only
                                      # for round ends it was there for)
        self._obs_t = None            # clock() of the last observe() (tracker button freshness)
        self._money_t = None          # clock() of the last HUD read that agreed with the filter
        self._money_unsure = False    # a purchase went unconfirmed: money may have dropped unseen

    @property
    def money(self):
        """Last accepted HUD money, or None before the first good read."""
        return self.hud.current.money

    def _read_hud(self, frame=None):
        """Read the HUD once and feed the filter -> the raw Hud. Money counts as fresh
        (MONEY_FRESH_S) only when the read agrees with the filtered value it leaves."""
        r = self.game.read_hud(frame)
        self.hud.update(r)
        f = self.money
        if r.money is not None and f is not None and self.hud.agrees("money", f, r.money):
            self._money_t = self.clock()
        return r

    def refresh_hud(self, frame=None):
        """Read the HUD once and feed the filter -> the smoothed Hud."""
        self._read_hud(frame)
        return self.hud.current

    # --- registry ---------------------------------------------------------------------------
    def register(self, kind, xy, spent):
        """Record a tower we placed -> its tid."""
        if kind not in self.towers:
            raise KeyError(kind)
        tid = self._next_tid
        self._next_tid += 1
        self.registry[tid] = guards.TowerRec(tid, kind, tuple(xy), (0, 0, 0), int(spent))
        self.bought.add(kind)
        if None in self.slots:          # more towers than slots: the extra ones get none
            self.slots[self.slots.index(None)] = tid
        return tid

    def register_hero(self, xy, spent):
        if self.hero_choice is None:
            raise ValueError("no hero chosen for this game")
        self.hero = guards.HeroRec(self.hero_choice, tuple(xy), 1, 0, int(spent))
        self._gain_hero_abilities(1)
        return HERO_TID

    def record_upgrade(self, tid, path, price):
        t = self.registry[tid]
        tiers = list(t.tiers)
        tiers[path] += 1
        t.tiers, t.spent = tuple(tiers), t.spent + int(price)
        new = [a for a in (self.towers[t.kind].get("stats") or {}).get("abilities") or []
               if a["path"] == path and a["tier"] == tiers[path]]
        if new:
            # this tier's list replaces the last tier's (Turbo -> Perma Charge). A tier can have
            # several (Heli 0-4-0: Redeploy + Support Drop), and one that carries over to the
            # next tier (Redeploy at 0-5-0, Vine Crush, Take Aim) keeps its cooldown and slot
            names = {a["name"] for a in new}
            self._drop_abilities(lambda key, name: key == (tid, path) and name not in names)
        for a in new:
            kept = next((s for s in self.ability_slots if s["name"] == a["name"] and (tid, path) in s["inst"]), None)
            if kept is None:
                self._gain_ability(tid, a["name"], a.get("cooldown"), a.get("between_rounds", True), path)
            elif a.get("cooldown"):         # carried over, maybe faster (Redeploy 45 s -> 15 s):
                cd = float(a["cooldown"])       # this tower's instance only, not the slot's others
                kept["cds"][(tid, path)] = cd
                kept["inst"][(tid, path)] = min(kept["inst"][(tid, path)], self.clock() + cd)

    def unregister(self, tid):
        """Forget a sold tower (or the hero, tid 'hero')."""
        if tid == HERO_TID:
            self.hero = None
        else:
            self.registry.pop(tid)
            self.slots = [None if s == tid else s for s in self.slots]
        self._drop_abilities(lambda key, name: key[0] == tid)

    # --- contexts for guards / placement ----------------------------------------------------
    def guard_state(self):
        """guards.GuardState from what the harness knows now. Unknown money counts as $0, so
        nothing that costs money is offered until the HUD has been read."""
        return guards.GuardState(
            money=self.money or 0, difficulty=self.difficulty, towers=self.towers, heroes=self.heroes,
            account=self.account, registry=self.registry, hero=self.hero, hero_choice=self.hero_choice,
            bought=self.bought, abilities=self.abilities, round_active=self.round_active, locked=self.locked)

    # --- abilities (bar order = the order they were gained) ---------------------------------
    def _gain_ability(self, tid, name, cooldown, between_rounds=True, path=None):
        """A tower tier (`path`) or hero level (path None) added an activated ability. Adds
        only: record_upgrade drops what a later tier replaces (Turbo -> Perma Charge). New
        abilities start on cooldown (seen live, step 4.5)."""
        cd = float(cooldown) if cooldown else float(self.default_cooldown)
        slot = next((a for a in self.ability_slots if a["name"] == name), None)
        if slot is None:
            slot = {"name": name, "between_rounds": bool(between_rounds), "inst": {}, "cds": {}}
            self.ability_slots.append(slot)
        slot["inst"][(tid, path)] = self.clock() + cd
        slot["cds"][(tid, path)] = cd

    @staticmethod
    def _cd(a, key=None):
        """Cooldown of slot `a`'s instance `key` (default: the one ready soonest). Kept per
        instance: two towers' copies of one ability can be at different tiers (Heli 0-4-0
        Redeploy 45 s, 0-5-0 15 s)."""
        if key is None:
            key = min(a["inst"], key=a["inst"].get)
        return a["cds"][key]

    def _drop_abilities(self, match):
        """Drop every instance for which match((tid, path), ability name) holds."""
        for a in self.ability_slots:
            a["inst"] = {k: v for k, v in a["inst"].items() if not match(k, a["name"])}
            a["cds"] = {k: v for k, v in a["cds"].items() if k in a["inst"]}
        n = len(self.ability_slots)
        self.ability_slots = [a for a in self.ability_slots if a["inst"]]
        if len(self.ability_slots) != n:    # later slots moved down: their read history is stale
            self.bar_filter.reset()

    def _hero_round_xp(self, r):
        """End of round `r` (Standard): the placed hero gains rules.hero_xp_for_round(r) XP and
        levels up from it the way the game does, gaining each new level's abilities. The four
        maps are Beginner (XP x 1); XP Monkey Knowledge isn't modelled. An estimate: the game's
        level isn't read back from the screen."""
        h = self.hero
        if h is None or self.mode != "standard":
            return
        need = self.heroes[h.name]["level_xp"]
        h.xp_into_level += rules.hero_xp_for_round(r)
        while h.level < 20 and h.xp_into_level >= need[h.level - 1]:
            h.xp_into_level -= need[h.level - 1]
            h.level += 1
            self._gain_hero_abilities(h.level)
        if h.level >= 20:
            h.xp_into_level = 0

    def _gain_hero_abilities(self, level):
        for a in self.hero_abilities.get(self.hero_choice) or []:
            if a["level"] == level:
                self._gain_ability(HERO_TID, a["name"], a["cooldown_s"])

    @property
    def abilities(self):
        """[guards.Ability] in bar order. `ready` comes from the cooldown estimate (game seconds
        counted as wall seconds), which every bar read (`sync_abilities`: observe() and the
        ability actions) resets from the pixels (5.4)."""
        now = self.clock()
        return [guards.Ability(i, a["name"], any(now >= t for t in a["inst"].values()), a["between_rounds"],
                               next(iter(a["inst"]))[0])
                for i, a in enumerate(self.ability_slots)]

    def sync_abilities(self, content, panel_side=None):
        """Read the ability bar from `content` (5.4) and set each slot's cooldown estimate from
        it: a ring with fraction f -> the slot's earliest instance is ready in f * cooldown; no
        ring -> it's ready now. Slots the reader can't see are left alone. Returns the
        filtered reads, or None when there's nothing to read."""
        if content is None or not self.ability_slots:
            return None
        now = self.clock()
        reads = self.bar_filter.update(self.read_bar(content, len(self.ability_slots), panel_side), now,
                                       [self._cd(a) for a in self.ability_slots])
        for r, a in zip(reads, self.ability_slots):
            if not r.seen:
                continue
            key = min(a["inst"], key=a["inst"].get)
            a["inst"][key] = now + r.frac * self._cd(a, key) if r.ring else min(a["inst"][key], now)
        return reads

    def place_context(self):
        return PlaceContext(self.towers, self.unit, self._placed(), self.keepout, self.memory, self.heroes)

    def available_actions(self):
        """Legal actions right now with their costs (guards.available_actions)."""
        return guards.available_actions(self.guard_state())

    # --- input ------------------------------------------------------------------------------
    def key_for(self, action, path=None):
        """The confirmed hotkey for `action` ('hero', 'upgrade' + path, 'sell', 'play_ff', ...,
        or a towers.json key), or None if there isn't one (then click the UI point)."""
        if action in self.towers:
            k = self.towers[action].get("hotkey")
            return k.lower() if k else None
        k = self.ui["hotkeys"].get(action)
        if isinstance(k, list):
            return k[path] if path is not None and path < len(k) else None
        return k

    def press(self, action, xy=None, path=None, wait=None):
        """Hotkey if confirmed, else click `xy`. ok=False with 'no_input' if neither exists,
        'not_front' if the game didn't take the input."""
        k = self.key_for(action, path)
        if k is not None:
            sent = self.game.key(k)
        elif xy is not None:
            sent = self.game.click(*xy)
        else:
            return ActionResult(False, "no_input")
        if not sent:
            return ActionResult(False, "not_front")
        self.sleep(self.ui["timing_s"]["after_click"] if wait is None else wait)
        return ActionResult(True, action)

    def select_shop(self, kind):
        """Pick up a tower (towers.json key) or the hero ('hero') from the shop. Uses the
        hotkey when there is one; otherwise scrolls the shop to the layout showing it and
        clicks its cell."""
        if kind != "hero" and kind not in self.towers:
            return ActionResult(False, "unknown_tower")
        wait = self.ui["timing_s"]["after_tower_key"]
        if self.key_for(kind) is not None:
            return self.press(kind, wait=wait)
        if self.mode == "sandbox":
            r = self.sandbox_panel("shop")      # a new Sandbox opens on the bloon panel
            if not r:
                return r
        layout, xy = ui.shop_cell(kind)
        if layout != self.shop_layout:
            sx, sy = self.ui["shop"]["scroll_point"]
            if not self.game.scroll(sx, sy, SCROLL_LINES if layout == "top" else -SCROLL_LINES):
                return ActionResult(False, "not_front")
            self.shop_layout = layout
            self.sleep(0.5)
        if not self.game.click(*xy):
            return ActionResult(False, "not_front")
        self.sleep(wait)
        return ActionResult(True, kind)

    def content(self):
        """The current content-area frame (BGR), or None if it can't be captured."""
        f = self.game.grab()
        return None if f is None else f.content_bgr()

    def cancel_hold(self):
        """Drop a held tower with the cancel button (not Escape: with nothing held, Escape
        opens the pause menu)."""
        return self.press("cancel_placement", xy=tuple(self.ui["buttons"]["cancel_placement"]))

    def _hud_money(self, raw=False):
        """One raw HUD money read (also fed to the filter); if unreadable, the filter's last
        accepted value (may be None), or None with `raw`."""
        r = self._read_hud()
        return r.money if r.money is not None or raw else self.money

    def _money_before(self):
        """The money before an action, for _money_moved. A single raw read is used only if it
        agrees with the filtered value or with a second raw read: a lone misread ($1,357 read
        as $357) would make a real purchase look unconfirmed, and the tower would go
        unregistered. Otherwise the filtered value (may be None). A filtered value read in the
        last MONEY_FRESH_S is used as is (no extra OCR on the action's hot path), unless a
        purchase went unconfirmed since. These reads aren't fed to the filter: two quick reads
        of a still frame can repeat one misread, and the filter would take it. So two agreeing
        reads below the filtered value count only while a drop is possible (_money_unsure:
        money only falls when we spend); otherwise the filtered value stands."""
        f = self.money
        fresh = self._money_t is not None and self.clock() - self._money_t < MONEY_FRESH_S
        if f is not None and fresh and not self._money_unsure:
            return f
        a = self.game.read_hud().money
        if a is None:
            return f
        if f is None or self.hud.agrees("money", f, a):
            return a
        b = self.game.read_hud().money
        if b is not None and self.hud.agrees("money", a, b) and (b > f or self._money_unsure):
            self.hud.accept("money", b)
            self._money_unsure = False
            return b
        return f

    def _money_moved(self, before, delta, max_move=None, min_move=None):
        """Wait for the HUD money to move by about `delta` (< 0 spent, > 0 refund) from
        `before`. Pop income can come in meanwhile, so half the amount counts. A move up to the
        expected size (a spend can only look smaller, from income; a refund larger by up to
        REFUND_INCOME_SLACK) confirms on one read and is forced into the filtered money. The
        amount is only an estimate for hero levels, sells and table prices the panel didn't
        show, so a bigger move up to `max_move` (default 1.5x) confirms on two agreeing reads;
        Beyond `max_move` it's a misread ($1,357 read as $57, maybe repeated on a still frame)
        and never confirms. Likewise a smaller move down to `min_move` (default: none below
        half), for an estimate that can be too high (a hero level when XP was missed).
        -> the amount to book (`delta`'s size on the one-read path, the measured move on the
        two-read path), or 0 when unconfirmed; then _money_unsure is set, since the action
        may have gone through anyway."""
        if before is None:
            self._money_unsure = True
            return 0
        d, sign = abs(delta), (1 if delta > 0 else -1)
        hi = d + REFUND_INCOME_SLACK if delta > 0 else d * 1.1 + 5
        top = max(hi, max_move if max_move is not None else d * 1.5 + (REFUND_INCOME_SLACK if delta > 0 else 5))
        low = d / 2 if min_move is None else min(d / 2, max(1, min_move))
        other = None                            # last read outside the one-read range
        for _ in range(CONFIRM_TRIES):
            m = self._hud_money(raw=True)       # a miss isn't a read (no filtered fallback)
            moved = None if m is None else (m - before) * sign
            if moved is not None and d / 2 <= moved <= hi:
                paid = d                        # confirmed: the guards shouldn't wait for 2 more reads
            elif moved is not None and (hi < moved <= top or low <= moved < d / 2):
                if other is None or not self.hud.agrees("money", other, m):
                    other = m
                    self.sleep(0.15)
                    continue
                paid = moved                    # two reads agree: book what really moved
            else:
                self.sleep(0.15)
                continue
            self.hud.accept("money", m)
            self._money_unsure = False
            return paid
        self._money_unsure = True
        return 0

    # --- 4.2 placing and selling ------------------------------------------------------------
    def _place(self, shop_key, kind, x, y, price):
        """Hold `shop_key` from the shop over (x, y), read the tint once, and click if it's ok.
        -> ActionResult (ok = placed; the caller registers it)."""
        ctx = self.place_context()
        p = can_place(kind, x, y, ctx)
        if not p:
            return p
        before = self._money_before()
        if not self.game.move(*PARK):
            return ActionResult(False, "not_front")
        self.sleep(0.2)
        base = self.content()
        tint = "none"
        for _ in range(2):                       # a dropped hold reads 'none': pick it up again
            r = self.select_shop(shop_key)
            if not r:
                return r
            self.game.move(x, y)
            self.sleep(SETTLE_S)
            tint = self.tint(base, self.content(), x, y)
            if tint != "none":
                break
        if tint != "ok":
            self.cancel_hold()
            if tint == "blocked":
                record_placement(ctx, kind, x, y, False)
                return ActionResult(False, "blocked", price)
            return ActionResult(False, "no_hold", price)
        if not self.game.click(x, y):
            return ActionResult(False, "not_front")
        self.sleep(self.ui["timing_s"]["after_click"])
        if not self._money_moved(before, -price):
            # The money didn't drop: not placed. If the tower is still held, the spot refused it.
            still = self.tint(base, self.content(), x, y)
            if still != "none":
                self.cancel_hold()
                record_placement(ctx, kind, x, y, False)
                return ActionResult(False, "blocked", price)
            return ActionResult(False, "unconfirmed", price)
        record_placement(ctx, kind, x, y, True)
        return ActionResult(True, "placed", price)

    def place_tower(self, kind, x, y):
        """Buy a `kind` tower and place it at (x, y). ok -> value is the new tid.
        Refusals: the guard's ('no_money', 'unknown_tower'), can_place's ('off_map', 'path',
        'overlaps_tower:<tid>', 'memory_blocked', ...), or 'blocked' (the tint said no; the
        spot is remembered), 'no_hold', 'unconfirmed', 'not_front'."""
        g = guards.can_buy_tower(kind, self.guard_state())
        if not g:
            return g
        r = self._place(kind, kind, x, y, g.cost)
        if not r:
            return r
        return ActionResult(True, "placed", g.cost, self.register(kind, (x, y), g.cost))

    def place_hero(self, x, y):
        """Place the hero chosen for this game at (x, y). ok -> value 'hero'."""
        g = guards.can_place_hero(self.guard_state())
        if not g:
            return g
        r = self._place("hero", self.hero_choice, x, y, g.cost)
        if not r:
            return r
        tid = self.register_hero((x, y), g.cost)
        # placed mid-round: it earns this round's end XP even if no observe() runs before the
        # round ends; placed at idle, a round end observe() hasn't seen yet isn't its
        if self.mode == "standard" and self._button_now() in ("ff_off", "ff_on"):
            self._hero_prev = True
        return ActionResult(True, "placed", g.cost, tid)

    def select_tower(self, tid):
        """Click a placed tower (or the hero) to open its panel."""
        t = self.hero if tid == HERO_TID else self.registry.get(tid)
        if t is None:
            return ActionResult(False, "no_tower")
        if not self.game.click(*t.xy):
            return ActionResult(False, "not_front")
        self.sleep(self.ui["timing_s"]["after_click"])
        return ActionResult(True, "selected")

    def sell(self, tid):
        """Sell a tower (or the hero, tid 'hero'): select it, then the sell key. ok -> cost is
        the refund as a negative number."""
        g = guards.can_sell(tid, self.guard_state())
        if not g:
            return g
        before = self._money_before()
        r = self.select_tower(tid)
        if not r:
            return r
        r = self.press("sell", xy=tuple(self.ui["upgrade_panel"]["left"]["sell"]))
        if not r:
            return r
        if not self._money_moved(before, -g.cost):
            return ActionResult(False, "unconfirmed", g.cost)
        self.unregister(tid)
        return ActionResult(True, "sold", g.cost)

    # --- 4.3 upgrades -----------------------------------------------------------------------
    def panel_side(self, tries=3):
        """Which side the selection panel is open on ('left' / 'right'), or None."""
        for _ in range(tries):
            side = self.read_panel(self.content())
            if side:
                return side
            self.sleep(0.15)
        return None

    def close_panel(self, side):
        return self.press("close_panel", xy=tuple(ui.panel(side)["close"]))

    def _panel_buy(self, tid, guard, press, max_cost=None, min_cost=None):
        """Select `tid`, find its panel, run `press(side)`, confirm the money drop on the HUD,
        close the panel. -> ActionResult (ok = paid; cost = the amount booked, see
        _money_moved). If `press` returns a cost (the price the panel showed), that's the
        amount confirmed instead of the guard's. `max_cost` / `min_cost`: the range a price
        estimate can be off by (_money_moved)."""
        before = self._money_before()
        r = self.select_tower(tid)
        if not r:
            return r
        side = self.panel_side()
        if side is None:
            return ActionResult(False, "no_panel", guard.cost)
        r = press(side)
        self.close_panel(side)          # first: a panel on the left covers the HUD money
        if not r:
            return r
        cost = guard.cost if r.cost is None else r.cost
        # the HUD can miss the drop; a press whose result says the panel saw it ("pips") counts
        paid = self._money_moved(before, -cost, max_cost, min_cost)
        ok = bool(paid) or r.value == "pips"
        return ActionResult(ok, "bought" if ok else "unconfirmed", paid or cost)

    def upgrade(self, tid, path):
        """Buy the next tier on `path` (0-2) for tower `tid`: guard (crosspath, tier <= 5, one
        tier 5 per path per type, money), then the path's hotkey with the panel open.
        ok -> value is the tower's new tiers."""
        g = guards.can_upgrade(tid, path, self.guard_state())
        if not g:
            return g

        def buy(side):
            # 5.3: read the row first. The panel's word beats the registry's: a locked or
            # unaffordable row isn't pressed, and a different price is the one confirmed.
            row = self.read_paths(side, (path,)).get(path)
            cost = None
            if row is not None and row.state != "unknown":
                self._note_check(tid, {path: row})
                if row.state in ("max", "closed"):
                    return ActionResult(False, "panel_locked", g.cost, row)
                if row.state == "poor":
                    return ActionResult(False, "panel_no_money", g.cost, row)
                if row.price is not None and row.price != g.cost:
                    t = self.registry[tid]
                    self.price_mismatches.append((t.kind, t.tiers, path, row.price, g.cost))
                    cost = row.price
            r = self.press("upgrade", xy=tuple(ui.panel(side)["path_buttons"][path]), path=path)
            if not r:
                return r
            seen = None                     # the tier pips going up confirm it without the HUD
            if row is not None and row.state != "unknown":
                for _ in range(3):
                    after = self.read_paths(side, (path,), prices=False).get(path)
                    if after is not None and after.tier == row.tier + 1:
                        seen = "pips"
                        break
                    self.sleep(0.15)
            return ActionResult(True, r.reason, cost, seen)

        r = self._panel_buy(tid, g, buy)
        if not r:
            return r
        self.record_upgrade(tid, path, r.cost)
        return ActionResult(True, "upgraded", r.cost, self.registry[tid].tiers)

    # --- 5.3 panel check ----------------------------------------------------------------------
    def _read_paths(self, side, paths=(0, 1, 2), prices=True):
        f = self.game.grab()
        if f is None:
            return {}
        return panel.read_paths(f.content_bgr(), side, (lambda box: self.game.ocr(f, box)) if prices else None, paths)

    def _note_check(self, tid, rows):
        """Compare panel rows of `tid` with the guards and remember per path whether they
        agreed (at the tower's current tiers). -> {path: [disagreements]}."""
        t = self.registry[tid]
        bad = panel.compare(t, rows, self.guard_state())
        tiers, seen = self.panel_checks.get(tid, (None, {}))
        if tiers != t.tiers:
            seen = {}
        seen.update({p: p not in bad for p, row in rows.items() if row.state != "unknown"})
        self.panel_checks[tid] = (t.tiers, seen)
        return bad

    def check_upgrades(self, tid):
        """Open `tid`'s panel, read the 3 path rows (state, OCR'd price, tier pips), close it
        and compare with the price table + guards (panel.compare). ok -> everything readable
        agreed ('match'); else 'mismatch' / 'unreadable' / a selection failure. value is
        (rows, {path: [disagreements]}). observe() shows the result as UpgradeOption.checked."""
        if tid not in self.registry:
            return ActionResult(False, "no_tower")
        r = self.select_tower(tid)
        if not r:
            return r
        side = self.panel_side()
        if side is None:
            return ActionResult(False, "no_panel")
        rows = self.read_paths(side)
        self.close_panel(side)
        bad = self._note_check(tid, rows)
        if all(row.state == "unknown" for row in rows.values()):
            return ActionResult(False, "unreadable", None, (rows, bad))
        return ActionResult(not bad, "mismatch" if bad else "match", None, (rows, bad))

    # --- 5.5 range --------------------------------------------------------------------------
    @property
    def ground_unit(self):
        """One game unit on the ground in normalized coords (ux, uy): the map's unit_norm with
        y times range_circle.ground_y_scale (the map is seen at a tilt; RESEARCH 5.5)."""
        return rangecircle.ground_unit(self.unit)

    def measure_range(self, tid):
        """Measure tower `tid`'s range from its circle: select it, grab, close the panel, grab,
        and read the disc between the two frames (rangecircle.measure). ok -> value is the
        range in game units and it agrees with the table within range_circle.tolerance
        ('match'), or the table has no value ('measured'); 'mismatch' (recorded in
        range_mismatches) / 'unreadable' / a selection failure otherwise. The result is kept
        per tower at its current tiers (observe(): TowerState.range_measured)."""
        t = self.registry.get(tid)
        if t is None:
            return ActionResult(False, "no_tower")
        rc = self.ui["range_circle"]
        r = self.select_tower(tid)
        if not r:
            return r
        side = self.panel_side()
        if side is None:
            return ActionResult(False, "no_panel")
        self.sleep(rc["settle_s"])
        selected = self.content()
        self.close_panel(side)
        self.sleep(rc["settle_s"] + 0.2)
        deselected = self.content()
        if selected is None or deselected is None or selected.shape != deselected.shape:
            return ActionResult(False, "no_frame")
        p = ui.panel(side)
        (ax, ay), (hx, hy) = p["ability_icon"], rc["ability_icon_half"]
        exclude = [tuple(b) for b in rc["exclude_always"]] + [tuple(p["panel"]), (ax - hx, ay - hy, ax + hx, ay + hy)]
        read = self.read_range(selected, deselected, t.xy, self.ground_unit, exclude)
        if read.units is None:
            return ActionResult(False, "unreadable", None, read)
        self.range_checks[tid] = (t.tiers, read.units)
        table, _ = rules.tower_range(self.towers[t.kind], t.tiers)
        if table is None:
            return ActionResult(True, "measured", None, read.units)
        if abs(read.units - table) > rc["tolerance"]:
            self.range_mismatches.append((t.kind, t.tiers, read.units, table))
            return ActionResult(False, "mismatch", None, read.units)
        return ActionResult(True, "match", None, read.units)

    def buy_hero_level(self):
        """Buy the hero's next level with the panel's level-up button. ok -> value is the new
        level. The guard's price is $1 per XP still missing, from the XP observe() credits
        at each round end (_hero_round_xp, an estimate). The button shows the real price:
        it's read first (refused as 'panel_no_money' if it's more than the money) and is
        the amount confirmed and booked; unreadable, the HUD-measured move is booked."""
        g = guards.can_buy_hero_level(self.guard_state())
        if not g:
            return g
        # the XP estimate can be off either way: the game charges between ~$1 (XP missed)
        # and the whole level (XP over-credited)
        full = guards.hero_level_price(self.guard_state(), xp_into_level=0)

        def level_up(side):
            price = panel.parse_price(self.read_hero_price(side))
            if price is not None and not 1 <= price <= full:
                price = None                    # misread: no level costs more than a whole one
            if price is not None and self.money is not None and price > self.money:
                return ActionResult(False, "panel_no_money", price)
            r = self.press("hero_level_up", xy=tuple(ui.panel(side)["hero_level_up"]))
            return ActionResult(r.ok, r.reason, price) if r else r

        r = self._panel_buy(HERO_TID, g, level_up, max_cost=full, min_cost=1)
        if not r:
            return r
        h = self.hero
        h.level, h.xp_into_level, h.spent = h.level + 1, 0, h.spent + r.cost
        self._gain_hero_abilities(h.level)
        return ActionResult(True, "leveled", r.cost, h.level)


    # --- 4.4 rounds, fast-forward, auto start -----------------------------------------------
    def _poll(self, read, want=None, tries=4, wait=0.15):
        """Read the current frame with `read` until it gives a value (or `want(value)`
        holds), at most `tries` times -> the last value."""
        v = None
        for i in range(tries):
            v = read(self.content())
            if (want(v) if want else v is not None):
                return v
            if i < tries - 1:
                self.sleep(wait)
        return v

    def _button_now(self, content=None, tries=4, first=None):
        """The play button once `tracker.confirm` consecutive reads agree on it (the first from
        `content` if given; the same count RoundTracker uses), at most `tries` reads ->
        'idle' / 'ff_off' / 'ff_on', or None. A single frame can misread, and round_active
        decides whether Space starts a round or toggles FF. `first`: a read the caller already
        made of `content` (that frame isn't read again)."""
        need = max(1, self.tracker.confirm)
        prev, run = None, 0
        for i in range(max(tries, need)):
            if i:
                self.sleep(0.05)
            if i == 0 and first is not None:
                cur = first
            else:
                cur = self.read_button(content if (i == 0 and content is not None) else self.content())
            run = run + 1 if cur is not None and cur == prev else (1 if cur is not None else 0)
            prev = cur
            if run >= need:
                return cur
        return None

    def sandbox_panel(self, side):
        """Sandbox: show the tower shop ('shop') or the bloon-send panel ('bloons'); one
        button toggles them. ok=False 'no_panel' if neither is recognised."""
        cur = self._poll(self.read_sandbox_panel)
        if cur is None:
            return ActionResult(False, "no_panel")
        if cur != side:
            if not self.game.click(*self.ui["buttons"]["powers"]):
                return ActionResult(False, "not_front")
            self.shop_layout = None             # don't trust the shop's scroll after a toggle
            self.sleep(self.ui["timing_s"]["panel_toggle"])
            if self._poll(self.read_sandbox_panel, lambda v: v == side) != side:
                return ActionResult(False, "no_panel")
        return ActionResult(True, side)

    def start_round(self):
        """Start the next round.
        Standard: only from idle (▶); play_ff, then the button must turn to ▶▶, and fast
        forward is switched on if the round started without it (_ensure_ff).
        Sandbox: the ▶▶ button shows all the time and Space doesn't send bloons, so this
        switches to the bloon panel, clicks its send button and switches back to the shop.
        Nothing on screen confirms the send or the round's end there, so a send always goes
        (Sandbox can stack rounds) and sets round_active for good: round-only abilities stay
        allowed for the rest of the Sandbox game."""
        if self.mode == "sandbox":
            r = self.sandbox_panel("bloons")
            if not r:
                return r
            if not self.game.click(*self.ui["buttons"]["sandbox_send_round"]):
                return ActionResult(False, "not_front")
            self.sleep(self.ui["timing_s"]["after_click"])
            self.round_active = True
            r = self.sandbox_panel("shop")
            if r:
                self._ensure_ff()
            return ActionResult(True, "sent", 0) if r else r
        # The button decides, not round_active: that's only as fresh as the last observe(),
        # so a round that ended since would be refused.
        # Only on agreeing reads: on one misread 'idle' mid-round, Space would toggle FF
        # instead. Reads that never agree -> 'no_button', nothing pressed.
        b = self._button_now()
        if b is None:
            return ActionResult(False, "no_button")
        self.round_active = b != "idle"
        if b != "idle":
            self.fast_forward = b == "ff_on"
        g = guards.can_start_round(self.guard_state())
        if not g:
            return g
        r = self.press("play_ff", xy=tuple(self.ui["buttons"]["play_ff"]))
        if not r:
            return r
        b = self._poll(self.read_button, lambda v: v in ("ff_off", "ff_on"), tries=6)
        if b not in ("ff_off", "ff_on"):
            return ActionResult(False, "unconfirmed")
        self.round_active, self.fast_forward = True, b == "ff_on"
        self._ensure_ff()
        return ActionResult(True, "started", 0)

    def _ensure_ff(self):
        """Turn fast forward on: one more press of the running ▶▶ button (Space at idle starts
        a round at normal speed). FF isn't an action: the harness keeps it on so episodes run
        fast. Best effort: a press that doesn't take leaves fast_forward False / None and the
        round still counts as started. Pressed only on agreeing ff_off reads (_button_now): one
        misread frame of a round already on FF would otherwise switch it off."""
        b = self._button_now()
        if b is not None:
            self.fast_forward = b == "ff_on"
        if b != "ff_off":
            return
        if not self.press("play_ff", xy=tuple(self.ui["buttons"]["play_ff"])):
            return
        b = self._poll(self.read_button, lambda v: v == "ff_on", tries=6)
        self.fast_forward = True if b == "ff_on" else (False if b == "ff_off" else None)

    # --- 4.5 abilities ----------------------------------------------------------------------
    def slot_crop(self, frame, slot, panel_side=None):
        """The ability bar icon for `slot` cut from a content frame (bottom clipped at the
        screen edge), or None."""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        x, y = ui.ability_slot(slot, panel_side)
        hx, hy = self.ui["ability_bar"]["icon_half"]
        return frame[max(round((y - hy) * h), 0):h, max(round((x - hx) * w), 0):min(round((x + hx) * w), w)]

    def _ability_fired(self, slot, before):
        """Did the press fire the ability? A (near) full ring on the slot says yes. Where the
        bar reader can see the slot and only one instance was ready, the ring is required:
        an icon coming off cooldown changes the crop just as much (seen live, 2026-11-06).
        Otherwise (unreadable, or a ready-count badge, whose first press leaves the icon
        ready) the icon crop changing enough counts."""
        need = self.ui["ability_bar"]["used_min_diff"]
        now = self.clock()
        multi = sum(t <= now for t in self.ability_slots[slot]["inst"].values()) > 1
        for _ in range(3):
            c = self.content()
            r = self.read_bar(c, slot + 1)[slot] if c is not None else None
            if r is not None and r.seen and r.ring and r.frac >= 0.85:
                return True
            if (r is None or not r.seen or multi) and self.icon_diff(before, self.slot_crop(c, slot)) >= need:
                return True
            self.sleep(0.15)
        return False

    def _after_use(self, slot, fired, key=None):
        """Book the use on one instance: `key` (use_ability_of), else the instance that has been
        ready longest. Not fired: probably still cooling down (the estimate was early), so every
        instance backs off a little."""
        a, now = self.ability_slots[slot], self.clock()
        if not fired:
            for k, t in a["inst"].items():
                a["inst"][k] = max(t, now + 3.0)
            return ActionResult(False, "unconfirmed", 0)
        if key is None:
            key = min(a["inst"], key=a["inst"].get)
        a["inst"][key] = now + self._cd(a, key)
        return ActionResult(True, "used", 0, a["name"])

    def _sync_round(self, content):
        """Standard: set round_active from the play button in `content` when it's readable
        (the round-only check shouldn't trust the last observe()). Sandbox's button always
        shows FF, so there it's left alone."""
        if self.mode == "standard" and content is not None:
            b = self.read_button(content)
            fresh = self._obs_t is not None and self.clock() - self._obs_t < SYNC_FRESH_S
            if b is None or b != self.tracker.button or not fresh:
                b = self._button_now(first=b)
            if b is not None:
                self.round_active = b != "idle"

    def use_ability(self, slot, target=None):
        """Activate ability bar slot `slot` (0-based; one icon per ability name, in order of first gain; a press uses one instance):
        its hotkey if confirmed, else a click on the icon. `target` (x, y): click it next, for
        abilities that ask for a point. Confirmed by the icon changing. ok -> value is the
        ability's name. With no panel open (actions close theirs), the bar is in place."""
        content = self.content()
        if 0 <= slot < len(self.ability_slots) and not self.ability_slots[slot]["between_rounds"]:
            self._sync_round(content)       # only round-only abilities care whether one is running
        self.sync_abilities(content)
        g = guards.can_use_ability(slot, self.guard_state())
        if not g:
            return g
        before = self.slot_crop(content, slot)
        r = self.press("ability", xy=ui.ability_slot(slot), path=slot)
        if not r:
            return r
        if target is not None:
            if not self.game.click(*target):
                return ActionResult(False, "not_front")
            self.sleep(self.ui["timing_s"]["after_click"])
        return self._after_use(slot, self._ability_fired(slot, before))

    def place_ability_target(self, slot, x, y):
        """Activate a targeted ability and click its target point (x, y)."""
        return self.use_ability(slot, target=(x, y))

    def use_ability_of(self, tid):
        """Activate tower `tid`'s (or the hero's) ability from its panel: the big ability icon
        next to the panel. A tower with two abilities uses the first. Confirmed on the bar
        after the panel is closed."""
        found = [(i, k) for i, a in enumerate(self.ability_slots) for k in a["inst"] if k[0] == tid]
        slot, key = found[0] if found else (None, None)
        if slot is None:
            return ActionResult(False, "no_ability" if tid in self.registry or tid == HERO_TID else "no_tower")
        content = self.content()
        if not self.ability_slots[slot]["between_rounds"]:
            self._sync_round(content)
        if not self.round_active and not self.ability_slots[slot]["between_rounds"]:
            return ActionResult(False, "round_only")
        self.sync_abilities(content)
        if self.clock() < self.ability_slots[slot]["inst"][key]:
            return ActionResult(False, "cooldown")
        before = self.slot_crop(content, slot)
        r = self.select_tower(tid)
        if not r:
            return r
        side = self.panel_side()
        if side is None:
            return ActionResult(False, "no_panel")
        clicked = self.game.click(*ui.panel(side)["ability_icon"])
        self.sleep(self.ui["timing_s"]["after_click"])
        self.close_panel(side)
        if not clicked:
            return ActionResult(False, "not_front")
        return self._after_use(slot, self._ability_fired(slot, before), key)

    # --- 4.6 targeting and target points ----------------------------------------------------
    def _ocr_targeting(self, side):
        f = self.game.grab()
        return "" if f is None else self.game.ocr(f, ui.panel(side)["targeting_label"])

    def _ocr_hero_price(self, side):
        f = self.game.grab()
        return "" if f is None else self.game.ocr(f, ui.panel(side)["hero_level_price"])

    def targeting_label(self, mode):
        """The panel text for a towers.json targeting mode ('lock_in_place' -> 'LOCK IN PLACE')."""
        return self.ui["targeting"]["labels"].get(mode, mode.replace("_", " ").upper())

    @staticmethod
    def _same_text(a, b):
        """OCR-tolerant label comparison (letters only, case-insensitive, ratio >= 0.8)."""
        import difflib
        norm = lambda t: "".join(c for c in t.upper() if c.isalpha())
        a, b = norm(a), norm(b)
        return bool(a) and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8

    def _label_is(self, text, mode):
        return self._same_text(text, self.targeting_label(mode))

    def _choose_mode(self, tid, mode):
        """Select `tid` and Tab through its targeting until the label reads `mode`. Leaves the
        panel open. -> (ActionResult, side); side is None if no panel was opened."""
        t = self.registry.get(tid)
        if t is None:
            return ActionResult(False, "no_tower"), None
        r = self.select_tower(tid)
        if not r:
            return r, None
        side = self.panel_side()
        if side is None:
            return ActionResult(False, "no_panel"), None
        first = None
        for _ in range(self.ui["targeting"]["max_presses"] + 1):
            text = ""
            for _ in range(3):
                text = self.read_targeting(side)
                if text:
                    break
                self.sleep(0.15)
            if not text:
                return ActionResult(False, "no_targeting"), side
            if self._label_is(text, mode):
                return ActionResult(True, "chosen", 0, mode), side
            if first is None:
                first = text
            elif self._same_text(text, first):
                break                               # back at the first label: mode not offered
            r = self.press("targeting_next", xy=tuple(ui.panel(side)["targeting_next"]), wait=0.35)
            if not r:
                return r, side
        return ActionResult(False, "no_such_mode"), side

    def set_targeting(self, tid, mode):
        """Set tower `tid`'s targeting to a towers.json mode ('first', 'strong', 'circle', ...):
        Tab until the panel's label reads it (upgrades can add modes, so the label is read
        each time, not counted). ok -> value is the mode."""
        r, side = self._choose_mode(tid, mode)
        if side is not None:
            self.close_panel(side)
        if not r:
            return r
        self.registry[tid].targeting = mode
        return ActionResult(True, "set", 0, mode)

    def set_flight_path(self, tid, pattern):
        """Monkey Ace flight pattern ('circle', 'figure_infinite', 'figure_eight'): a
        targeting mode on the Ace's panel."""
        return self.set_targeting(tid, pattern)

    def _set_points(self, tid, mode, points):
        """Choose a point mode, click its button (the label spot), click `points` on the map,
        and confirm by the panel reopening on the same side."""
        need = self.ui["targeting"]["point_modes"].get(mode)
        if need is None:
            return ActionResult(False, "not_point_mode")
        if len(points) != need:
            return ActionResult(False, "bad_points")
        r, side = self._choose_mode(tid, mode)
        if not r:
            if side is not None:
                self.close_panel(side)
            return r
        x0, y0, x1, y1 = ui.panel(side)["targeting_label"]
        if not self.game.click((x0 + x1) / 2, (y0 + y1) / 2):
            return ActionResult(False, "not_front")
        self.sleep(self.ui["timing_s"]["after_click"])
        if self.panel_side(tries=2) is not None:      # the button should close the panel
            self.close_panel(side)
            return ActionResult(False, "no_point_button")
        for x, y in points:
            if not self.game.click(x, y):
                return ActionResult(False, "not_front")
            self.sleep(self.ui["timing_s"]["after_click"])
        back = self.panel_side()
        if back is None:
            self.cancel_hold()                         # leave point mode (cancel button)
            return ActionResult(False, "unconfirmed")
        self.close_panel(back)
        self.registry[tid].targeting = mode
        return ActionResult(True, "set", 0, [tuple(p) for p in points])

    def set_target_point(self, tid, x, y):
        """Point the tower at (x, y): Mortar's Set Target, Heli's Lock in Place (ui.json
        targeting.point_mode_of)."""
        t = self.registry.get(tid)
        if t is None:
            return ActionResult(False, "no_tower")
        mode = self.ui["targeting"]["point_mode_of"].get(t.kind)
        if mode is None:
            return ActionResult(False, "no_target_point")
        return self._set_points(tid, mode, [(x, y)])

    def set_patrol_points(self, tid, points):
        """Heli Pilot patrol between two points [(x, y), (x, y)]."""
        t = self.registry.get(tid)
        if t is None:
            return ActionResult(False, "no_tower")
        if t.kind != "heli_pilot":
            return ActionResult(False, "no_patrol")
        return self._set_points(tid, "patrol_points", list(points))

    # --- 4.8 prompts ------------------------------------------------------------------------
    def _ui_point(self, ref):
        section, key = ref.split(".")
        return tuple(self.ui[section][key])

    def handle_prompts(self, overrides=None):
        """Clear popups and overlays with their default button (ui.json `prompts`; override per
        harness via `prompt_actions` or per call: {kind: 'section.key' | None}). Repeats while
        a handled prompt shows, since one can follow another (e.g. restart confirm over pause).
        -> the screen kinds acted on, in order. Prompts mapped to None (victory, defeat) are
        left for the caller, and the loop stops there."""
        actions = dict(self.prompt_actions, **(overrides or {}))
        cfg = self.ui["prompts"]
        handled = []
        for _ in range(cfg["max_steps"]):
            kind = self._poll(self.read_kind, tries=1)
            ref = actions.get(kind)
            if ref is None:
                break
            if not self.game.click(*self._ui_point(ref)):
                break
            handled.append(kind)
            self.sleep(cfg["after_click_s"])
        return handled

    # --- 4.9 games: new / restart / home ----------------------------------------------------
    MENUS = ("map_select", "difficulty_select", "mode_select")

    def screen_kind(self):
        """The current screen kind; None if the window can't be captured (then try to find
        it again: loading a map can drop it for a moment)."""
        kind = self.read_kind(self.content())
        if kind is None and hasattr(self.game, "reattach"):
            self.game.reattach(timeout=1.0)
        return kind

    def _wait_kind(self, kinds, timeout):
        """Poll until the screen is one of `kinds` -> that kind, or None after `timeout` s."""
        deadline = self.clock() + timeout
        while True:
            kind = self.screen_kind()
            if kind in kinds:
                return kind
            if self.clock() >= deadline:
                return None
            self.sleep(0.3)

    def _click_ui(self, ref, wait=None):
        ok = self.game.click(*self._ui_point(ref))
        self.sleep(self.ui["timing_s"]["menu"] if wait is None else wait)
        return ok

    def _open_pause(self):
        """Open the pause menu with the gear. An open right-hand panel shifts the HUD left and
        puts its own close ✕ where the gear is (seen live 2026-11-09), so close any panel
        first; a second gear click covers a panel the one read missed."""
        if self.screen_kind() == "pause":
            return True
        side = self.panel_side(tries=1)
        if side:
            self.close_panel(side)
            self.sleep(self.ui["timing_s"]["panel_toggle"])
        for _ in range(2):
            self.game.click(*self.ui["buttons"]["settings"])
            self.sleep(self.ui["timing_s"]["pause_open"])
            if self._wait_kind(("pause",), 2.0) == "pause":
                return True
        return False

    def go_home(self):
        """Back to the home screen from a game (pause -> Home), a defeat screen (Home), or a
        menu (back arrow). Victory isn't handled yet (what follows Next isn't mapped).
        Forgets the episode once the game is left."""
        t = self.ui["timing_s"]
        for _ in range(6):
            kind = self.screen_kind()
            if kind == "home":
                return ActionResult(True, "home")
            if kind in ("game", "pause"):
                if not self._open_pause():
                    return ActionResult(False, "no_pause_menu")
                self._click_ui("pause.home", wait=0)
                self.new_episode()
                if self._wait_kind(("home",), t["load_max"]):
                    self.sleep(t["menu"])           # home animates in; early clicks are lost
                    return ActionResult(True, "home")
                return ActionResult(False, "not_home")
            if kind == "defeat":
                self._click_ui("menus.defeat_home", wait=0)
                self.new_episode()
                if self._wait_kind(("home",), t["load_max"]):
                    self.sleep(t["menu"])           # home animates in; early clicks are lost
                    return ActionResult(True, "home")
                return ActionResult(False, "not_home")
            if kind in ("restart_confirm", "collection_event"):
                self.handle_prompts()
            elif kind in self.MENUS:
                self._click_ui("menus.back")
            else:
                return ActionResult(False, f"unknown_screen:{kind}")
        return ActionResult(False, "not_home")

    def _ocr_map_names(self):
        ms = self.ui["map_select"]
        f = self.game.grab()
        if f is None:
            return [""] * len(ms["slots"])
        hx, hy = ms["name_half"]
        return [self.game.ocr(f, [x - hx, y + ms["name_dy"] - hy, x + hx, y + ms["name_dy"] + hy])
                for x, y in ms["slots"]]

    def _stable_names(self):
        """The 6 thumbnail names once the grid has stopped sliding (two equal reads)."""
        names = self.read_map_names()
        for _ in range(6):
            self.sleep(0.3)
            again = self.read_map_names()
            if again == names and sum(1 for n in names if n.strip()) >= 2:   # blank mid-slide reads
                break                                                        # can also be "equal"
            names = again
        return names

    def _page_of(self, names, pages):
        """Beginner page index from the names in view (majority vote over every name that
        contains a known Beginner name), or None (another category, or nothing read)."""
        votes = {}
        for n in names:
            for pg, keys in enumerate(pages):
                if any(k in n for k in keys):
                    votes[pg] = votes.get(pg, 0) + 1
        return max(votes, key=votes.get) if votes else None

    def _flip(self, step, before, norm):
        """One page flip, confirmed by the (normalized) names changing; a click lost
        mid-slide is sent again once. Returns the new page's normalized stable names."""
        ms = self.ui["map_select"]
        names = before
        for _ in range(2):
            self.game.click(*ms[step])
            self.sleep(1.0)
            names = [norm(n) for n in self._stable_names()]
            if names != before:
                break
            self.pick_trace.append(("flip_lost", step))
        return names

    def _pick_map(self, map_name):
        """On map select: click the map's thumbnail, found by its OCR'd name. The grid
        reopens on some earlier page and paging wraps around every category, so the current
        page comes from any Beginner name in view (ui.json map_select.beginner_pages) and the
        search flips toward the target's page, keeping its direction across pages it can't
        place (forward to start). On the
        target's page, a name the OCR keeps missing falls back to the data slot. Each read
        and flip is logged in self.pick_trace."""
        ms = self.ui["map_select"]
        norm = lambda t: "".join(c for c in t.upper() if c.isalpha())
        pages = [[norm(n) for n in pg] for pg in ms["beginner_pages"]]
        want = norm(map_name)
        info = ms["maps"].get(map_name)
        target = info["page"] if info else self._page_of([want], pages)
        self.pick_trace = []
        names = [norm(n) for n in self._stable_names()]
        misses, step = 0, "next_page"
        for _ in range(ms["cycle_pages"]):
            here = self._page_of(names, pages)
            self.pick_trace.append(("page", here, names))
            for i, name in enumerate(names):
                if want and want in name:
                    self.pick_trace.append(("click", i))
                    self.game.click(*ms["slots"][i])
                    return ActionResult(True, "map", 0, map_name)
            if target is not None and here == target:
                misses += 1
                if misses < 3:                      # name not read yet: look again
                    self.sleep(0.3)
                    names = [norm(n) for n in self._stable_names()]
                    continue
                if info:
                    self.pick_trace.append(("click_slot", info["slot"]))
                    self.game.click(*ms["slots"][info["slot"]])
                    return ActionResult(True, "map_by_slot", 0, map_name)
                return ActionResult(False, "map_not_found")
            if here is not None and target is not None:     # else keep going the same way
                step = "prev_page" if here > target else "next_page"
            names = self._flip(step, names, norm)
        return ActionResult(False, "map_not_found")

    def new_game(self, map_name=None, difficulty=None, mode=None, overwrite_save=False):
        """Start a game: home -> Play -> map (found by name) -> difficulty -> mode, then the
        Sandbox popup's OK. Defaults: the harness's current map / difficulty / mode. Sandbox
        is locked on maps not beaten yet ('no_game' after the timeout). If that map and
        difficulty has a saved game the game asks 'Overwrite save?': OK only with
        `overwrite_save=True`, else Cancel and 'save_exists' (the save is the player's).
        ok -> the harness is set up for the new map with a fresh episode."""
        map_name = map_name or self.map_name
        difficulty = difficulty or self.difficulty
        mode = mode or self.mode
        if not (DATA / "maps" / f"{map_name}.json").exists():
            return ActionResult(False, "unknown_map")
        if difficulty not in ("easy", "medium", "hard"):
            return ActionResult(False, "unknown_difficulty")
        if mode not in ("standard", "sandbox"):
            return ActionResult(False, "unknown_mode")
        t = self.ui["timing_s"]
        r = self.go_home()
        if not r:
            return r
        for _ in range(2):                          # a click while home is settling is lost
            self._click_ui("menus.home_play", wait=0)
            if self._wait_kind(("map_select",), 4.0):
                break
        else:
            return ActionResult(False, "no_map_select")
        self.sleep(t["menu"])                       # the grid slides in; early page clicks are lost
        r = self._pick_map(map_name)
        if not r:
            return r
        if not self._wait_kind(("difficulty_select",), 5.0):
            return ActionResult(False, "no_difficulty_select")
        self._click_ui(f"menus.difficulty_{difficulty}", wait=0)
        if not self._wait_kind(("mode_select",), 5.0):
            return ActionResult(False, "no_mode_select")
        self._click_ui(f"menus.mode_{mode}", wait=0)
        kind = self._wait_kind(("game", "overwrite_save"), t["load_max"])
        if kind == "overwrite_save":
            if not overwrite_save:
                self._click_ui("menus.overwrite_save_cancel")
                return ActionResult(False, "save_exists")
            self._click_ui("menus.overwrite_save_ok", wait=0)
            kind = self._wait_kind(("game",), t["load_max"])
        if kind != "game":
            return ActionResult(False, "no_game")
        if mode == "sandbox":
            self.sleep(1.0)
            self._click_ui("menus.sandbox_popup_ok")   # harmless on the map if it isn't there
        if map_name != self.map_name:
            self._set_map(map_name)
        self.difficulty, self.mode = difficulty, mode
        self.new_episode()
        self.refresh_hud()
        return ActionResult(True, "new_game", 0, (map_name, difficulty, mode))

    def restart(self):
        """Restart the current game: pause (or the defeat screen) -> Restart -> confirm.
        ok -> a fresh episode on the same map."""
        t = self.ui["timing_s"]
        kind = self.screen_kind()
        if kind == "defeat":
            self._click_ui("menus.defeat_restart", wait=0)
        elif kind in ("game", "pause"):
            if not self._open_pause():
                return ActionResult(False, "no_pause_menu")
            self._click_ui("pause.restart", wait=0)
        elif kind != "restart_confirm":
            return ActionResult(False, f"not_in_game:{kind}")
        if not self._wait_kind(("restart_confirm",), 3.0):
            return ActionResult(False, "no_confirm")
        self._click_ui("menus.restart_confirm_restart", wait=0)
        self.new_episode()
        if not self._wait_kind(("game",), t["load_max"]):
            return ActionResult(False, "no_game")
        if self.mode == "sandbox":
            self.sleep(1.0)
            self._click_ui("menus.sandbox_popup_ok")
        self.refresh_hud()
        return ActionResult(True, "restarted")

    # --- 5.2 bloons -------------------------------------------------------------------------
    def capture_background(self, n=8, gap=0.5):
        """Replace the bloon detector's empty-track background with n grabbed frames (for this
        session only; tools/bloon_bg.py stores one). Only with an empty track and no Standard
        pre-round markers, i.e. in Sandbox. -> True if the detector took it."""
        if self.detector is None:
            return False
        frames = []
        for i in range(n):
            if i:
                self.sleep(gap)
            frames.append(self.content())
        return self.detector.set_background(frames)

    def _detector_towers(self):
        """Our towers' sprite areas for the detector: footprint + TOWER_SPRITE_PAD units."""
        ctx = self.place_context()
        out = []
        for p in self._placed():
            shape = shape_of(p.kind, p.tiers, ctx)
            out.append((p.xy[0], p.xy[1], (footprint_radius(shape) if shape else 6) + TOWER_SPRITE_PAD))
        return out

    def detect_bloons(self, frame=None):
        """[detect.Bloon] on a content frame (grabbed if None); [] without a background."""
        if self.detector is None:
            return []
        self.detector.set_towers(self._detector_towers())
        return self.detector.detect(self.content() if frame is None else frame)

    # --- 5.1 observation --------------------------------------------------------------------
    def _placed(self):
        placed = [Placed(t.kind, t.xy, t.tiers, t.tid) for t in self.registry.values()]
        if self.hero is not None:
            placed.append(Placed(self.hero.name, self.hero.xy, tid=HERO_TID))
        return placed

    def placeable_masks(self):
        """{mask class: placement.Mask}: the map's swept masks minus the path keep-out and our
        towers' footprints. Cached until a tower is placed, sold or upgraded."""
        from .observe import placeable_masks
        placed = self._placed()
        key = tuple((p.kind, p.xy, p.tiers) for p in placed)
        if self._masks_cache is None or self._masks_cache[0] != key:
            self._masks_cache = (key, placeable_masks(self.masks, self.unit, placed, self.keepout,
                                                      self.place_context()))
        return self._masks_cache[1]

    def mask_classes_of(self, kind, tiers=(0, 0, 0)):
        """The mask classes a tower kind (or the hero, 'land_r7') places on."""
        if kind in self.towers:
            return mask_classes(kind, shape_of(kind, tiers, self.place_context()))
        return ["land_r7"]

    def observe(self):
        """One GameState from one frame: screen kind and play button (state.read_screen), the
        HUD (in game only; filtered), round state (state.RoundTracker: round_ended fires once
        per finished round, outcome needs 2 frames), plus the registry, hero, upgrade options
        (price table; `checked` from the last panel read, 5.3), abilities (bar read, 5.4), ranges
        (table, or `range_measured` from measure_range, 5.5), path and placeable masks. Also syncs
        `round_active` from the play button and counts `rounds_done` (the HUD round reads the
        same before and after a round with auto-start off, so it's only trusted mid-round).
        `bloons` (5.2): detect.Bloon list, using the map's stored empty-track background
        (data/maps/<map>.bg.png); [] on idle Standard frames; None off the game screen or
        for a map without a stored background. Call it often: the round edge and outcome are
        worked out across calls."""
        from .observe import AbilityState, GameState, HeroState, TowerState, UpgradeOption
        f = self.game.grab()
        content = None if f is None else f.content_bgr()
        screen = self.read_screen(content)
        hud = self.refresh_hud(f) if screen.kind == "game" else self.hud.current
        was_outcome = self.tracker.outcome
        rs = self.tracker.update(screen, hud)
        if self.mode == "standard":     # Sandbox: the button always shows FF (rounds are sent
                                        # from the bloon panel), so it says nothing about rounds
            if rs.round_active is not None:
                self.round_active = rs.round_active
            if rs.round_ended:          # the tracker knows which rounds, unless the HUD never read
                ended = rs.ended_rounds or (self.rounds_done + 1,)
                for r in ended:
                    # each round's XP once, and only for a hero that was already placed at the
                    # previous observe() (placed after the round ended: it didn't earn it)
                    if r > self._xp_round and self._hero_prev:
                        self._hero_round_xp(r)
                    self._xp_round = max(self._xp_round, r)
                self.rounds_done = ended[-1]
            elif rs.outcome == "win" and was_outcome is None:
                self.rounds_done += 1
            run = self.tracker.running_round        # lag-corrected, unlike the raw HUD round
            if rs.round_active and run:
                self.rounds_done = run - 1
            self._hero_prev = self.hero is not None
        self._obs_t = self.clock()
        gs = self.guard_state()
        towers, options = [], {}
        for t in self.registry.values():
            rng, exact = rules.tower_range(self.towers[t.kind], t.tiers)
            seen_tiers, measured = self.range_checks.get(t.tid, (None, None))
            measured = measured if seen_tiers == t.tiers else None
            use = measured if measured is not None else rng
            ux, uy = self.ground_unit
            shape = shape_of(t.kind, t.tiers, self.place_context())
            towers.append(TowerState(t.tid, t.kind, t.xy, t.tiers, rng, exact or measured is not None,
                                     (use * ux, use * uy) if use is not None else None,
                                     t.targeting, t.spent, guards.sell_value(t.spent), measured,
                                     footprint_radius(shape) if shape else None,
                                     self.slots.index(t.tid) if t.tid in self.slots else None))
            opts = []
            for path in range(3):
                r = guards.can_upgrade(t.tid, path, gs)
                locked = r.reason if r.reason in ("max_tier", "crosspath", "tier5_taken") else None
                cost = None if r.reason == "max_tier" else guards.upgrade_cost(t, path, gs)
                tiers, seen = self.panel_checks.get(t.tid, (None, {}))
                opts.append(UpgradeOption(cost, bool(r), locked, seen.get(path) if tiers == t.tiers else None))
            options[t.tid] = opts
        hero = None
        if self.hero is not None:
            hero = HeroState(self.hero.name, self.hero.xy, self.hero.level, self.hero.xp_into_level,
                             guards.sell_value(self.hero.spent))
        now = self.clock()
        seen = set()
        if screen.kind == "game":
            reads = self.sync_abilities(content, screen.panel) or []
            seen = {r.slot for r in reads if r.seen}
            now = self.clock()
        abilities = []
        for a in self.ability_slots:
            left = min(t - now for t in a["inst"].values())
            frac = 0.0 if left <= 0 else min(1.0, left / self._cd(a))
            abilities.append(AbilityState(len(abilities), a["name"], next(iter(a["inst"]))[0], left <= 0, frac,
                                          a["between_rounds"], len(abilities) in seen))
        bloons = None
        if screen.kind == "game" and content is not None and self.detector is not None:
            if (self.mode == "standard" and screen.button == "idle" and rs.round_active is False
                    and not self.round_active):
                bloons = []                     # idle between rounds: no bloons (and the
                                                # pre-round track markers would read as some)
            elif self.detector.bg is not None:
                bloons = self.detect_bloons(content)
        cur = self.hud.current
        path = self.map.get("path") or {}
        return GameState(
            t=now, screen=screen.kind, map_name=self.map_name, difficulty=self.difficulty, mode=self.mode,
            money=cur.money, lives=cur.lives, round=cur.round, max_round=cur.rounds_total,
            hud_misses=dict(self.hud.misses),
            round_active=bool(self.round_active), round_ended=rs.round_ended and self.mode == "standard",
            fast_forward=rs.fast_forward,
            outcome=rs.outcome, towers=towers, hero=hero, upgrade_options=options, abilities=abilities,
            path=self.keepout.polyline if self.keepout is not None else None,
            path_half_width_units=path.get("half_width_units"), unit=self.unit,
            placeable=self.placeable_masks(),
            rounds_done=self.rounds_done, next_round=self.rounds_done + 1,
            round_composition=self.rounds.get(self.rounds_done + 1) if self.mode == "standard" else None,
            bloons=bloons, slots=list(self.slots))
