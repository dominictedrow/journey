"""Action guardrails (portable, pure): refuse actions the harness knows are illegal before
touching the game, and list the actions that are legal right now.

Every `can_*` returns ActionResult(ok, reason, cost). Harness actions call their guard first
and return its refusal unchanged (PLAN 4.2 / 4.3 / 4.5). `available_actions()` lists the legal
actions with their costs; the env's action mask is built from it (that part is Dom's).

Placement spots aren't enumerable, so `available_actions` lists which towers can be bought
and `placement.can_place` checks a given spot.

Refusal reasons: 'no_money', 'unknown_tower', 'unknown_hero', 'no_tower', 'max_tier',
'crosspath', 'tier5_taken', 'no_hero', 'hero_placed', 'hero_max_level', 'no_ability',
'cooldown', 'round_only', 'round_active'.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

from . import rules
from .result import ActionResult

SELL_RATE = 0.7           # sell value / money spent, seen in game (2.5); Easy, no MK sell bonus


@dataclass
class TowerRec:
    """A registry entry. `spent`: everything paid for it (base + upgrades)."""
    tid: int
    kind: str
    xy: tuple
    tiers: tuple = (0, 0, 0)
    spent: int = 0
    targeting: Optional[str] = None   # last mode set through the harness (None = game default)


@dataclass
class HeroRec:
    name: str                   # heroes.json key
    xy: tuple
    level: int = 1
    xp_into_level: int = 0
    spent: int = 0


@dataclass
class Ability:
    """An icon in the ability bar, as read off the screen."""
    slot: int
    name: str
    ready: bool
    between_rounds: bool = True     # usable while no round runs (towers.json abilities flag)
    tid: Any = None                 # the tower id it belongs to ('hero' for the hero), if known


@dataclass
class GuardState:
    money: int
    difficulty: str
    towers: dict                                   # towers.json "towers"
    heroes: dict = field(default_factory=dict)     # heroes.json "heroes"
    account: Optional[dict] = None                 # account.json (Monkey Knowledge)
    registry: dict = field(default_factory=dict)   # tid -> TowerRec
    hero: Optional[HeroRec] = None                 # the placed hero
    hero_choice: Optional[str] = None              # hero selected for this game (placeable once)
    bought: set = field(default_factory=set)       # tower kinds bought at least once this game
    abilities: list = field(default_factory=list)  # [Ability]
    round_active: bool = False
    locked: set = field(default_factory=set)       # tower kinds not available (mode / unlocks)


def _money(price, s, what):
    if price > s.money:
        return ActionResult(False, "no_money", price)
    return ActionResult(True, what, price)


def tower_price(kind, s):
    return rules.shop_price(kind, s.towers[kind]["cost"], s.difficulty, s.account, first=kind not in s.bought)


def can_buy_tower(kind, s: GuardState):
    """Can a `kind` tower be bought (placement aside)?"""
    if kind not in s.towers or kind in s.locked:
        return ActionResult(False, "unknown_tower")
    return _money(tower_price(kind, s), s, "buy_tower")


def upgrade_cost(t: TowerRec, path, s):
    tier = t.tiers[path] + 1
    base = s.towers[t.kind]["upgrades"][path][tier - 1]["cost"]
    return rules.upgrade_price(t.kind, path, tier, base, s.difficulty, s.account)


def can_upgrade(tid, path, s: GuardState):
    """Next tier on `path` (0-2) for tower `tid`: crosspath rule, tier <= 5, one tier 5 per
    path per tower type, and the money."""
    t = s.registry.get(tid)
    if t is None:
        return ActionResult(False, "no_tower")
    if not 0 <= path <= 2:
        raise ValueError(path)
    if t.tiers[path] >= 5:
        return ActionResult(False, "max_tier")
    if not rules.upgrade_allowed(t.tiers, path):
        return ActionResult(False, "crosspath")
    if t.tiers[path] == 4 and any(o.tid != tid and o.kind == t.kind and o.tiers[path] == 5 for o in s.registry.values()):
        return ActionResult(False, "tier5_taken")
    return _money(upgrade_cost(t, path, s), s, "upgrade")


def hero_price(name, s):
    return rules.shop_price(name, s.heroes[name]["cost"], s.difficulty, s.account)


def can_place_hero(s: GuardState):
    if s.hero is not None:
        return ActionResult(False, "hero_placed")
    if s.hero_choice is None or s.hero_choice not in s.heroes:
        return ActionResult(False, "unknown_hero")
    return _money(hero_price(s.hero_choice, s), s, "place_hero")


def _training_discount(account):
    return sum(e.get("training_discount", 0) for e in rules._mk_effects(account))


def hero_level_price(s, xp_into_level=None):
    """The hero's next level's price: $1 per XP still missing (`xp_into_level` default: the
    estimate on the HeroRec; 0 = the whole level)."""
    h = s.hero
    xp = h.xp_into_level if xp_into_level is None else xp_into_level
    return rules.hero_level_cost(s.heroes[h.name], h.level, xp, _training_discount(s.account))


def can_buy_hero_level(s: GuardState):
    if s.hero is None:
        return ActionResult(False, "no_hero")
    if s.hero.level >= 20:
        return ActionResult(False, "hero_max_level")
    return _money(hero_level_price(s), s, "buy_hero_level")


def sell_value(spent):
    return int(spent * SELL_RATE + 0.5)


def can_sell(tid, s: GuardState):
    """tid = a registry tid, or 'hero'. cost is the refund as a negative number."""
    if tid == "hero":
        return ActionResult(False, "no_hero") if s.hero is None else ActionResult(True, "sell", -sell_value(s.hero.spent))
    t = s.registry.get(tid)
    if t is None:
        return ActionResult(False, "no_tower")
    return ActionResult(True, "sell", -sell_value(t.spent))


def can_use_ability(slot, s: GuardState):
    a = next((a for a in s.abilities if a.slot == slot), None)
    if a is None:
        return ActionResult(False, "no_ability")
    if not a.ready:
        return ActionResult(False, "cooldown")
    if not s.round_active and not a.between_rounds:
        return ActionResult(False, "round_only")
    return ActionResult(True, "use_ability", 0)


def can_start_round(s: GuardState):
    return ActionResult(False, "round_active") if s.round_active else ActionResult(True, "start_round", 0)


def available_actions(s: GuardState):
    """The actions legal right now: [{'action', ..., 'cost'}], in a stable order. Placing a
    tower still needs a spot (placement.can_place)."""
    out = []

    def add(r, **kw):
        if r.ok:
            out.append({"action": r.reason, **kw, "cost": r.cost})

    add(can_start_round(s))
    for kind in sorted(s.towers):
        add(can_buy_tower(kind, s), kind=kind)
    if s.hero_choice is not None:
        add(can_place_hero(s), hero=s.hero_choice)
    add(can_buy_hero_level(s))
    for tid in sorted(s.registry):
        for path in range(3):
            add(can_upgrade(tid, path, s), tid=tid, path=path)
    for tid in sorted(s.registry):
        add(can_sell(tid, s), tid=tid)
    if s.hero is not None:
        add(can_sell("hero", s), tid="hero")
    for a in sorted(s.abilities, key=lambda a: a.slot):
        add(can_use_ability(a.slot, s), slot=a.slot)
    return out
