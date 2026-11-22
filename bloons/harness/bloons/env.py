"""The one import for a model: state, every action (flagged if it's legal now), and act(name).

    import sys; sys.path.insert(0, "../harness")
    from bloons.env import ACTIONS, Bot

    bot = Bot("monkey_meadow", "medium", hero="quincy")   # Steam BTD6 (Windows), game open on that map
    bot = Bot(..., platform="mac")                        # BTD6+ on the Mac instead
    s = bot.state()            # GameState (s.to_vector() / s.grid_image() for the network)
    bot.reward                 # reward since the last state(): +3 clean round, -10 leaky round, -11 defeat
    acts = bot.actions()       # {name: available} in ACTIONS order; list(acts.values()) is the mask
    r = bot.act("upgrade 3 2") # or bot.act(index into ACTIONS) -> ActionResult(ok, reason, cost, value)

Action names (fixed list, so ACTIONS[i] is model output i):
    start_round | hero_level | sell hero
    place <tower> <i> <j>   tower from obs.json tower_kinds, at grid cell column i, row j
    hero <i> <j>            place the hero chosen in Bot(hero=...)
    upgrade <slot> <path>   tower slot 0-29 (GameState.slots), path 0 top / 1 mid / 2 bottom
    sell <slot>
    ability <slot>          ability bar slot

"Available" is what the harness knows without touching the game (money, crosspath rules,
cooldowns, the placeable masks); the game can still refuse a spot, and act() then returns
ok=False with the reason.
"""
from . import calib, guards, obsvec
from .placement import placeable

SPEC = obsvec.load_spec()
_G = SPEC["grid"]
CELLS = [(i, j) for j in range(_G["ny"]) for i in range(_G["nx"])]
_XY = {c: obsvec.cell_centre(*c, SPEC) for c in CELLS}
ACTIONS = (["start_round", "hero_level", "sell hero"]
           + [f"place {k} {i} {j}" for k in SPEC["tower_kinds"] for i, j in CELLS]
           + [f"hero {i} {j}" for i, j in CELLS]
           + [f"upgrade {s} {p}" for s in range(SPEC["tower_slots"]) for p in range(3)]
           + [f"sell {s}" for s in range(SPEC["tower_slots"])]
           + [f"ability {a}" for a in range(SPEC["ability_slots"])])


class Bot:
    def __init__(self, map_name="monkey_meadow", difficulty="easy", hero=None, game=None, platform=None, **kw):
        """`platform`: 'win' (Steam BTD6) or 'mac' (BTD6+); None keeps BLOONS_PLATFORM, else win.
        It picks the data files and the game backend, so it must be the first thing in the process
        to load harness data: a later Bot with a different platform raises RuntimeError.
        `game`: default the real one (platform.Game); tests pass a fake. `kw` go to BloonsHarness."""
        self.platform = calib.use(platform)
        from .harness import HERO_TID, BloonsHarness   # after use(): it loads the calibrated data
        self.hero_tid = HERO_TID
        if game is None:
            from .platform import Game
            game = Game()
        self.h = h = BloonsHarness(game, map_name, difficulty, hero=hero, **kw)
        self.last, self.reward, self.lives0 = None, 0, None
        cell = lambda i, j: _XY[int(i), int(j)]  # noqa: E731
        self.do = {
            "start_round": h.start_round,
            "hero_level": h.buy_hero_level,
            "place": lambda k, i, j: h.place_tower(k, *cell(i, j)),
            "hero": lambda i, j: h.place_hero(*cell(i, j)),
            "upgrade": lambda s, p: h.upgrade(self._tid(s), int(p)),
            "sell": lambda s: h.sell(self.hero_tid if s == "hero" else self._tid(s)),
            "ability": lambda a: h.use_ability(int(a)),
        }

    def _tid(self, slot):
        return self.h.slots[int(slot)]          # None (empty slot) -> the guard says 'no_tower'

    def state(self):
        """Read the screen: a fresh GameState (also kept as self.last). Call it every step.
        Also sets self.reward, which is the reward earned since the previous state() call."""
        prev, self.last = self.last, self.h.observe()
        self.reward = self._reward(prev, self.last)
        return self.last

    def _reward(self, prev, s):
        """Round end: +3 with no lives lost, -10 with lives lost (+1 survive -11 leak).
        Defeat: -11. Money is never part of it."""
        new_outcome = s.outcome if s.outcome != getattr(prev, "outcome", None) else None
        if new_outcome == "loss":
            self.lives0 = None                   # Continue resets lives; re-read the baseline
            return -11
        r = 0
        if s.round_ended or new_outcome == "win":    # the final round ends as 'win', not round_ended
            # ponytail: unreadable lives count as no lives lost; the HUD filter makes that rare
            leaked = None not in (s.lives, self.lives0) and s.lives < self.lives0
            r = -10 if leaked else 3
        if s.lives is not None and (s.round_ended or self.lives0 is None):
            self.lives0 = s.lives                # baseline = lives at the last round end
        return r

    def actions(self):
        """{name: available} for every name in ACTIONS, from the last state()."""
        s = self.last or self.state()
        h = self.h
        ok = dict.fromkeys(ACTIONS, False)
        slot = {t: i for i, t in enumerate(h.slots) if t is not None} | {self.hero_tid: "hero"}
        for a in guards.available_actions(h.guard_state()):
            kind = a["action"]
            if kind in ("buy_tower", "place_hero"):
                k = a.get("kind")
                classes = h.mask_classes_of(k or h.hero_choice)
                for (i, j), xy in _XY.items():
                    if placeable(s.placeable, classes, *xy):
                        ok[f"place {k} {i} {j}" if k else f"hero {i} {j}"] = True
            elif kind == "upgrade":
                ok[f"upgrade {slot.get(a['tid'])} {a['path']}"] = True
            elif kind == "sell":
                ok[f"sell {slot.get(a['tid'])}"] = True
            elif kind == "use_ability":
                ok[f"ability {a['slot']}"] = True
            else:                                # start_round, buy_hero_level
                ok[{"buy_hero_level": "hero_level"}.get(kind, kind)] = True
        return {n: ok[n] for n in ACTIONS}       # drops past-the-list names (21st tower, 9th ability)

    def act(self, name):
        """Run one action by name ('place dart_monkey 5 7') or ACTIONS index -> ActionResult."""
        if isinstance(name, int):
            name = ACTIONS[name]
        verb, *args = name.split()
        return self.do[verb](*args)
