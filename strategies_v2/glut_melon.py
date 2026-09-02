"""GLUT_MELON -- melon monoculture, maximum volume, dump at any price.

Phase 5.1 glutter opponent (watcher_svm_repair_plan.md). Not a profit-optimal
build: its job is to put the shared-market price-collapse regime that
`MD files/CLAUDE.md` says the watcher exists for *into the training and
evaluation distribution*. Melon is the single most glut-fragile product
(T=300, above_func "sq", above_target 3.60, zero town demand) -- a dedicated
high-volume seller drives its realized price to the $1 floor and keeps it
there for the rest of the season.

Design: fill NW + NE with melon (well past the ~19-tile self-limiting count
the submitted agent stops at), staff 6 hands, replant every vacancy up to
melon's season-end cutoff, and SELL every melon every turn regardless of
price (SELL_MIN_PRICE_FRAC = 0 -- the whole point is to flood, not to time
the market). Scale is deliberately kept to the two-quadrant / ~6-hand
envelope that ring/solid prove does not bankrupt on the $3000 cash curve;
the flood comes from the zero sell-gate and the extra tiles, not from an
all-quadrant build that stalls before melon's day-10 first yield.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import (
    QUADRANT_TILES, chunk_indices, crop_action, last_buy_day, next_land_buy,
    sell_orders,
)

MELON_YIELD_DAY = 12  # CROPS["MELON"] max_yield_day
MAX_MELON_TILES = 32
MELON_HANDS = 6
TARGET_QUADRANTS = 2  # NW + NE
SELL_MIN_PRICE_FRAC = 0.0  # dump everything, every turn, at any price
BASE_PRICE = {"MELON": 250, "MILK": 160, "WOOL": 200, "CARROT": 35}
MELON_LAST_BUY_DAY = last_buy_day("MELON")
LAND_LAST_BUY_DAY = MELON_LAST_BUY_DAY

# NW first (owned at start), then NE -- row-major within each so the dense
# block stays contiguous and hand travel is short.
MELON_POSITIONS = (
    QUADRANT_TILES["NW"] + QUADRANT_TILES["NE"]
)[:MAX_MELON_TILES]
MELON_GROUPS = chunk_indices(len(MELON_POSITIONS), 1 + MELON_HANDS)
N_HIRES = MELON_HANDS


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
    prices = obs.get("market", {}).get("prices", {})

    buy_orders = []
    sells = sell_orders(shed, prices, BASE_PRICE, SELL_MIN_PRICE_FRAC, hold_exempt=())

    if hour == 0:
        for _ in range(N_HIRES):
            buy_orders.append(["HIRE"])

    if hour == 1 and day <= LAND_LAST_BUY_DAY:
        land = next_land_buy(me, money, TARGET_QUADRANTS)
        if land:
            buy_orders.append(land)

    if hour == 2 and day <= MELON_LAST_BUY_DAY:
        empty_melon = sum(1 for x, y in MELON_POSITIONS if tiles[y][x] is None)
        have_seeds = seeds.get("MELON", 0)
        if empty_melon > have_seeds:
            buy_orders.append(["BUY_SEED", "MELON", empty_melon - have_seeds])

    market = buy_orders + sells

    fx, fy = me["farmer"]
    farmer_idxs = MELON_GROUPS[0] if MELON_GROUPS else []
    farmer_action = (
        crop_action("MELON", MELON_YIELD_DAY, (fx, fy), day, tiles, farmer_idxs, MELON_POSITIONS, seeds.get("MELON", 0))
        if farmer_idxs else ["PASS"]
    )

    hands = me.get("hands", [])
    hands_actions = []
    for h_idx, hpos in enumerate(hands):
        unit_num = h_idx + 1
        if unit_num < len(MELON_GROUPS):
            idxs = MELON_GROUPS[unit_num]
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
    print("glut_melon self-check OK, final money:", final)


if __name__ == "__main__":
    _demo()
