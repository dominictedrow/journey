"""GLUT_LIVESTOCK -- cow + sheep herd at maximum volume, dump milk/wool at any price.

Phase 5.1 glutter opponent (watcher_svm_repair_plan.md). Sibling of
glut_melon for the animal-product side of the shared market: wool (T=105,
above_func "sq", above_target 3.20) and milk (T=122, above_func "linear",
above_target 1.60) are both premium and glut-fragile, with only partial shop
coverage. A large herd selling every unit every day floors both.

Design: RING's central-livestock tile logic (11 COW / 6 SHEEP / 6 hands is
its proven, non-bankrupting build), scaled modestly to NW + NE with a bigger
herd and one extra hand, no melon. Feed wheat is bought from the market each
day. SELL_MIN_PRICE_FRAC = 0 -- dump milk and wool at any price (WHEAT is
held back for feed). Keeping to two quadrants matters: a herd spread over
three quadrants outruns the crew's daily feeding reach and starves, which
collapses supply instead of flooding it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (
    QUADRANT_TILES, chunk_indices, last_buy_day, livestock_action,
    next_land_buy, sell_orders,
)

ANIMAL_MIX = {"COW": 16, "SHEEP": 10}
ANIMAL_COST = {"COW": 400, "SHEEP": 500}
LIVESTOCK_HANDS = 7
TARGET_QUADRANTS = 2  # NW + NE
SELL_MIN_PRICE_FRAC = 0.0  # dump milk/wool at any price; WHEAT held for feed
BASE_PRICE = {"MELON": 250, "MILK": 160, "WOOL": 200, "CARROT": 35}
LAST_BUY_DAY = {"COW": last_buy_day("COW"), "SHEEP": last_buy_day("SHEEP")}
LAND_LAST_BUY_DAY = min(LAST_BUY_DAY.values())

HUB = (4.5, 4.5)

_NW_SET = set(QUADRANT_TILES["NW"])
_HUB_SORTED = sorted(
    QUADRANT_TILES["NW"] + QUADRANT_TILES["NE"],
    key=lambda p: (0 if p in _NW_SET else 1, abs(p[0] - HUB[0]) + abs(p[1] - HUB[1]), p[1], p[0]),
)


def _livestock_types():
    seq = []
    for animal, count in ANIMAL_MIX.items():
        seq += [animal] * count
    return seq


LIVESTOCK_TYPES = _livestock_types()
LIVESTOCK_POSITIONS = _HUB_SORTED[: len(LIVESTOCK_TYPES)]
LIVESTOCK_GROUPS = chunk_indices(len(LIVESTOCK_POSITIONS), 1 + LIVESTOCK_HANDS)
N_HIRES = LIVESTOCK_HANDS


def agent(obs):
    player = obs["player"]
    me = obs["farms"][player]
    private = obs["private"]
    day = obs["day"]
    hour = obs["hour"]
    tiles = me["tiles"]
    shed = private.get("shed", {})
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
        else:
            hands_actions.append(["PASS"])

    return {"farmer": farmer_action, "hands": hands_actions, "market": market}


def _demo():
    from kaggle_environments import make

    env = make("kaggriculture", configuration={"episodeSteps": 200}, debug=False)
    env.run([agent, "random"])
    final = env.steps[-1][0].observation["farms"][0]["money"]
    assert final >= 0
    print("glut_livestock self-check OK, final money:", final)


if __name__ == "__main__":
    _demo()
