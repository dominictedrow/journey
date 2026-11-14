"""Pure game rules: difficulty pricing, the upgrade crosspath rule, tower and hero stats (RESEARCH.md)."""

from fractions import Fraction
from math import floor

# Exact fractions: e.g. 320 x 0.85 must be exactly 272 for the rounding below.
DIFFICULTY_MULT = {"easy": Fraction(85, 100), "medium": Fraction(1), "hard": Fraction(108, 100),
                   "impoppable": Fraction(120, 100)}


def cost(base, difficulty, item="upgrade", flat=0, mult=1):
    """In-game price of a $`base` (medium) item at `difficulty`.

    `flat` / `mult` are Monkey Knowledge discounts (see `shop_price` / `upgrade_price`): the
    scaled price gets `flat` added, then is multiplied by `mult`, and is rounded once at the end.

    Rounding to a multiple of $5 differs by item, both measured in game on Easy (2.5, 3.1):
    - "upgrade": up once it's $2 or more past one, so ties go up: $187 -> 190, $272 -> 275,
      $212.50 -> 215, $161.50 -> 160. That's 5 * floor((x + 3) / 5).
    - "tower" (shop base price): nearest, ties to the even multiple: $212.50 -> 210 (Wizard,
      Beast Handler), $722.50 -> 720 (Dartling), $297.50 -> 300 (Sniper), $467.50 -> 470.
      This is the wiki Module:BTD6_costs rule; it's wrong for upgrades.
    Hard/Impoppable haven't been checked in game.
    """
    x = (base * DIFFICULTY_MULT[difficulty] + Fraction(flat)) * Fraction(mult).limit_denominator(1000)
    if item == "tower":
        return 5 * round(x / 5)          # Fraction rounds half to even
    if item == "upgrade":
        return 5 * floor((x + 3) / 5)
    raise ValueError(item)


def _mk_effects(account):
    return [e for e in (account or {}).get("monkey_knowledge", {}).values() if e.get("active")]


def shop_price(name, base, difficulty, account=None, first=True):
    """Shop price of tower `name` (towers.json key, base $`base`), with the active Monkey
    Knowledge in `account` (data/account.json). `first`: no tower of this kind bought yet this
    game (Farm Subsidy / First Last Line of Defense apply only then)."""
    flat, mult = 0, Fraction(1)
    for e in _mk_effects(account):
        if name not in e.get("towers", ()) or (e.get("first_only") and not first):
            continue
        flat += e.get("flat", 0)
        mult *= Fraction(str(e.get("mult", 1)))
    return cost(base, difficulty, "tower", flat, mult)


def upgrade_price(name, path, tier, base, difficulty, account=None):
    """Price of tower `name`'s upgrade on `path` (0-2) to `tier` (1-5), base $`base`, with the
    active Monkey Knowledge flat discounts in `account`."""
    flat = sum(e.get("flat", 0) for e in _mk_effects(account) if [name, path, tier] in e.get("upgrades", ()))
    return cost(base, difficulty, "upgrade", flat)


def upgrade_allowed(tiers, path):
    """Can a tower at `tiers` (e.g. (2, 0, 3)) buy the next tier on `path` (0-2)?

    At most 2 paths upgraded, and only one of them past tier 2.
    Per-map "one tier-5 per path per tower type" is the harness's job, not this function's.
    """
    new = list(tiers)
    new[path] += 1
    if new[path] > 5:
        return False
    return sum(t > 0 for t in new) <= 2 and sum(t > 2 for t in new) <= 1


# --- tower stats (towers.json "stats" block; see scripts/fetch_wiki_data.py) ---------------
# Data covers the base tower and single-path tiers. Crosspaths aren't in it: the game builds
# them by applying each upgrade's changes, some additive (Long Range Darts +8) and some
# multiplicative (Spike-o-pult x1.15), in an order the data doesn't give.

def tower_range(tower, tiers):
    """(range in game units, exact) for a tower dict from towers.json at `tiers` (e.g. (2, 0, 3)).
    With one upgraded path the value is the game's; with two it's base + the sum of each
    path's change, flagged exact=False (PLAN 5.5 measures the circle). (None, False) if unknown."""
    st = tower.get("stats")
    if not st or st.get("range") is None:
        return None, False
    base, total, upgraded = st["range"], st["range"], 0
    for path, tier in enumerate(tiers):
        if tier:
            r = st["range_by_tier"][path][tier - 1]
            if r is None:
                return None, False
            total += r - base
            upgraded += 1
    return round(total, 2), upgraded <= 1


def tower_abilities(tower, tiers):
    """Activated abilities a tower at `tiers` has: [{'name', 'cooldown', 'between_rounds', ...}].
    Each upgraded path contributes the abilities listed for its current tier. None if unknown."""
    st = tower.get("stats")
    if not st or st.get("abilities") is None:
        return None
    return [a for a in st["abilities"] if a["tier"] == tiers[a["path"]]]


def tower_shape(tower, tiers):
    """{'placement', 'footprint', 'targeting'} at `tiers`: the base values with any override
    for an upgraded path's current tier applied (e.g. Sun Temple's 37x40 rect footprint)."""
    st = tower.get("stats")
    if not st:
        return None
    shape = {k: st[k] for k in ("placement", "footprint", "targeting")}
    for o in st["overrides"]:
        if tiers[o["path"]] == o["tier"]:
            shape.update({k: v for k, v in o.items() if k in shape})
    return shape


# --- heroes (heroes.json; see scripts/fetch_wiki_data.py) ----------------------------------

def hero_xp_for_round(r):
    """XP a hero gains at the end of round r (before map-difficulty multipliers)."""
    if r <= 20:
        return 20 * r + 20
    if r <= 50:
        return 40 * r - 380
    return 90 * r - 2880


def hero_xp_to_reach(hero, level):
    """Total XP from level 1 to `level` (1..20)."""
    return sum(hero["level_xp"][:level - 1])


def hero_level_cost(hero, level, xp_into_level=0, training_discount=0.0):
    """Cash to buy the next level for a hero at `level` with `xp_into_level` XP already earned
    toward it: $1 per remaining XP, not scaled by difficulty, minus Monkey Knowledge
    "Scholarships" (training_discount=0.1). Checked in game (step 2.5): Benjamin level 1 -> 2
    on Easy with Scholarships = $243 = 270 x 0.9. None at level 20."""
    if level >= 20:
        return None
    remaining = max(0, hero["level_xp"][level - 1] - xp_into_level)
    return int(remaining * (1 - training_discount) + 0.5)


def hero_abilities(hero, level):
    """Activated abilities unlocked at `level`, each with its current cooldown:
    [{'name', 'level', 'cooldown'}]. None if the wiki has no data for this hero."""
    st = hero.get("stats")
    if not st:
        return None
    out = []
    for a in st["abilities"]:
        if a["level"] <= level:
            cds = [cd for lv, cd in a["cooldown"].items() if int(lv) <= level]
            out.append({"name": a["name"], "level": a["level"], "cooldown": cds[-1] if cds else None})
    return out


def hero_range(hero, level):
    st = hero.get("stats")
    return st["range_by_level"][level - 1] if st else None

