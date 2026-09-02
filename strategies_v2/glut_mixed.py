"""GLUT_MIXED -- premium dumper: melon + milk + wool at volume, all sold at any price.

Phase 5.1 glutter opponent (watcher_svm_repair_plan.md). Where glut_melon and
glut_livestock each crater one side of the premium market, glut_mixed spreads
a large build across melon, cows and sheep so all three premium prices fall
at once -- the regime a real opponent converging on "everything expensive"
creates. Central-hub livestock ring (RING's tile logic) with melon filling
the rest of three quadrants; feed wheat bought daily; every melon, milk and
wool unit sold every turn (SELL_MIN_PRICE_FRAC = 0). This build spends its
whole opening bankroll on capacity and occasionally runs itself insolvent
late-season -- acceptable here: by then milk (and often melon/wool) is
already floored and stays floored, which is the regime Phase 5 needs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (
    QUADRANT_TILES, chunk_indices, crop_action, last_buy_day, livestock_action,
    next_land_buy, sell_orders,
)

ANIMAL_MIX = {"COW": 16, "SHEEP": 10}
ANIMAL_COST = {"COW": 400, "SHEEP": 500}
LIVESTOCK_HANDS = 6
MELON_HANDS = 3
MAX_MELON_TILES = 26
MELON_YIELD_DAY = 12
TARGET_QUADRANTS = 3  # NW + NE + SW
SELL_MIN_PRICE_FRAC = 0.0  # dump melon/milk/wool at any price; WHEAT held for feed
BASE_PRICE = {"MELON": 250, "MILK": 160, "WOOL": 200, "CARROT": 35}
LAST_BUY_DAY = {"COW": last_buy_day("COW"), "SHEEP": last_buy_day("SHEEP"), "MELON": last_buy_day("MELON")}
LAND_LAST_BUY_DAY = LAST_BUY_DAY["MELON"]

HUB = (4.5, 4.5)

_NW_SET = set(QUADRANT_TILES["NW"])
_HUB_SORTED = sorted(
    QUADRANT_TILES["NW"] + QUADRANT_TILES["NE"] + QUADRANT_TILES["SW"],
    key=lambda p: (0 if p in _NW_SET else 1, abs(p[0] - HUB[0]) + abs(p[1] - HUB[1]), p[1], p[0]),
)


def _livestock_types():
    seq = []
    for animal, count in ANIMAL_MIX.items():
        seq += [animal] * count
    return seq


LIVESTOCK_TYPES = _livestock_types()
LIVESTOCK_POSITIONS = _HUB_SORTED[: len(LIVESTOCK_TYPES)]
_claimed = set(LIVESTOCK_POSITIONS)
MELON_POSITIONS = [p for p in _HUB_SORTED if p not in _claimed][:MAX_MELON_TILES]

LIVESTOCK_GROUPS = chunk_indices(len(LIVESTOCK_POSITIONS), 1 + LIVESTOCK_HANDS)
MELON_GROUPS = chunk_indices(len(MELON_POSITIONS), MELON_HANDS)
N_HIRES = LIVESTOCK_HANDS + MELON_HANDS


def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]
    hour = obs["hour"]
    tiles = me["tiles"]
    shed = private.get("shed", {})
    seeds = private.get("seeds", {})
    money = me["money"]
    inventories = private.get("inventories", [{}])
    prices = obs.get("market", {}).get("prices", {})

    buy_orders = []
    sells = sell_orders(shed, prices, BASE_PRICE, SELL_MIN_PRICE_FRAC)

    if hour == 0:
        for _ in range(N_HIRES):
            buy_orders.append(["HIRE"])

    if hour == 1:
        wheat_needed = len(LIVESTOCK_TYPES)
        wheat_have = shed.get("WHEAT", 0)
        for inv in inventories:
            wheat_have += inv.get("WHEAT", 0)
        deficit = wheat_needed - wheat_have
        if deficit > 0:
            wheat_price = prices.get("WHEAT", 25)
            afford = int(money // wheat_price) if wheat_price > 0 else deficit
            n_buy = min(deficit, afford)
            if n_buy > 0:
                buy_orders.append(["BUY_PRODUCT", "WHEAT", n_buy])

    if hour == 2:
        needed_counts = {}
        for i, atype in enumerate(LIVESTOCK_TYPES):
            x, y = LIVESTOCK_POSITIONS[i]
            t = tiles[y][x]
            if not (isinstance(t, dict) and t.get("animal") == atype):
                needed_counts[atype] = needed_counts.get(atype, 0) + 1
        cash = money
        for atype, need in needed_counts.items():
            if day > LAST_BUY_DAY[atype]:
                continue
            have = shed.get(atype, 0)
            for inv in inventories:
                have += inv.get(atype, 0)
            to_buy = max(0, need - have)
            cost_each = ANIMAL_COST[atype]
            afford = int(cash // cost_each)
            n_buy = min(to_buy, afford)
            if n_buy > 0:
                buy_orders.append(["BUY_ANIMAL", atype, n_buy])
                cash -= n_buy * cost_each

    if hour == 3 and day <= LAND_LAST_BUY_DAY:
        land = next_land_buy(me, money, TARGET_QUADRANTS)
        if land:
            buy_orders.append(land)

    if hour == 4 and day <= LAST_BUY_DAY["MELON"]:
        empty_melon = sum(1 for x, y in MELON_POSITIONS if tiles[y][x] is None)
        have_seeds = seeds.get("MELON", 0)
        if empty_melon > have_seeds:
            buy_orders.append(["BUY_SEED", "MELON", empty_melon - have_seeds])

    market = buy_orders + sells

    fx, fy = me["farmer"]
    farmer_inv = inventories[0] if inventories else {}
    farmer_idxs = LIVESTOCK_GROUPS[0] if LIVESTOCK_GROUPS else []
    farmer_action = (
        livestock_action((fx, fy), farmer_inv, tiles, shed, farmer_idxs, LIVESTOCK_POSITIONS, LIVESTOCK_TYPES)
        if farmer_idxs else ["PASS"]
    )

    hands = me.get("hands", [])
    hands_actions = []
    for h_idx, hpos in enumerate(hands):
        unit_num = h_idx + 1
        hinv = inventories[unit_num] if unit_num < len(inventories) else {}
        if unit_num < len(LIVESTOCK_GROUPS):
            idxs = LIVESTOCK_GROUPS[unit_num]
            hands_actions.append(
                livestock_action(tuple(hpos), hinv, tiles, shed, idxs, LIVESTOCK_POSITIONS, LIVESTOCK_TYPES)
                if idxs else ["PASS"]
            )
        elif unit_num - len(LIVESTOCK_GROUPS) < len(MELON_GROUPS):
            idxs = MELON_GROUPS[unit_num - len(LIVESTOCK_GROUPS)]
            hands_actions.append(
                crop_action("MELON", MELON_YIELD_DAY, tuple(hpos), day, tiles, idxs, MELON_POSITIONS, seeds.get("MELON", 0))
                if idxs else ["PASS"]
            )
        else:
            hands_actions.append(["PASS"])

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}


def _demo():
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": 200}, debug=False)
    env.run([agent, "random"])
    final = env.steps[-1][0].observation["farms"][0]["money"]
    assert final >= 0
    print("glut_mixed self-check OK, final money:", final)


if __name__ == "__main__":
    _demo()
