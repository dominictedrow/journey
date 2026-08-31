"""Evaluation-only operational telemetry for Kaggriculture episodes.

This module deliberately stays outside the generated submission.  It consumes
the environment's recorded steps after a game and never changes agent actions.
"""

from collections import Counter, defaultdict
from statistics import mean


MOVE_ACTIONS = {"NORTH", "SOUTH", "EAST", "WEST"}
WORK_ACTIONS = {
    "BUILD_COOP", "BUILD_PASTURE", "CARE", "COLLECT_FERTILIZER", "DIG",
    "DROP", "FEED", "FERTILIZE", "HARVEST", "PICKUP", "PLACE", "PLANT", "WATER",
}
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_COST = {1: 1000, 2: 2000, 3: 4000}


def _marginal_hire_cost(prior_hires):
    a, b = 1, 1
    for _ in range(prior_hires):
        a, b = b, a + b
    return a


def _farm(step, player_idx):
    return step[player_idx].observation["farms"][player_idx]


def _private(step, player_idx):
    return step[player_idx].observation.get("private", {})


def _tiles(farm):
    return farm.get("tiles", [])


def _asset_counts(farm):
    animals = Counter()
    crops = Counter()
    weeds = 0
    ready = 0
    feed_risk = 0
    water_risk = 0
    for row in _tiles(farm):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "WEED":
                weeds += 1
            animal = tile.get("animal")
            crop = tile.get("crop")
            if animal:
                animals[animal] += 1
                feed_risk += int(not tile.get("fed_today", False))
            if crop:
                crops[crop] += 1
                water_risk += int(not tile.get("watered_today", False))
            ready += max(0, tile.get("yield_units", 0) or 0)
    return animals, crops, weeds, ready, feed_risk, water_risk


def _inventory(private):
    total = Counter(private.get("shed", {}))
    for inv in private.get("inventories", []):
        total.update(inv)
    return total


def _action_name(action):
    if isinstance(action, (list, tuple)) and action:
        return action[0]
    return "PASS"


def episode_metrics(env, player_idx):
    """Return JSON-serializable metrics for one player in a completed episode."""
    steps = env.steps
    first = _farm(steps[0], player_idx)
    last = _farm(steps[-1], player_idx)
    money_curve = [_farm(step, player_idx).get("money", 0) for step in steps]
    action_counts = defaultdict(Counter)
    requested_orders = Counter()
    requested_market = submitted_within_cap = over_cap_requests = 0
    hire_requested_by_day = Counter()
    hand_count_by_day = defaultdict(list)
    land_request_days = []
    requested_sale_quantity = Counter()
    requested_sale_prices = defaultdict(list)
    purchase_quantity_requested = Counter()
    spending_requested = Counter()
    planted = Counter()
    harvested = Counter()
    planting_days = defaultdict(list)
    harvest_days = defaultdict(list)
    crop_exit_days = defaultdict(list)
    animal_losses = crop_losses = 0
    unfed_tile_turns = unwatered_tile_turns = 0
    unused_tile_turns = 0
    opponent_exposure = Counter()

    prior_animals, prior_crops, _, _, _, _ = _asset_counts(first)
    for turn, step in enumerate(steps):
        state = step[player_idx]
        obs = state.observation
        farm = obs["farms"][player_idx]
        opponent = obs["farms"][1 - player_idx]
        day = obs.get("day", turn // 24)
        hands = farm.get("hands", [])
        hand_count_by_day[day].append(len(hands))
        unlocked = set(farm.get("unlocked_quadrants", ["NW"]))
        unlocked_cells = 25 * len(unlocked)
        occupied = sum(isinstance(tile, dict) for row in _tiles(farm) for tile in row)
        unused_tile_turns += max(0, unlocked_cells - occupied)

        action = state.action if isinstance(state.action, dict) else {}
        unit_actions = [action.get("farmer", ["PASS"])] + list(action.get("hands", []))
        unit_positions = [farm.get("farmer", [4, 4])] + list(hands)
        for slot, unit_action in enumerate(unit_actions):
            name = _action_name(unit_action)
            category = "move" if name in MOVE_ACTIONS else "work" if name in WORK_ACTIONS else "pass"
            action_counts[str(slot)][category] += 1
            if name == "PLANT" and len(unit_action) > 1:
                planted[unit_action[1]] += 1
                planting_days[unit_action[1]].append(day)
            if name == "HARVEST":
                harvested["actions"] += 1
                if slot < len(unit_positions):
                    x, y = unit_positions[slot]
                    board = _tiles(farm)
                    if 0 <= y < len(board) and 0 <= x < len(board[y]):
                        tile = board[y][x]
                        product = tile.get("crop") if isinstance(tile, dict) else None
                        if product:
                            harvested[product] += 1
                            harvest_days[product].append(day)

        market = action.get("market", []) if isinstance(action.get("market", []), list) else []
        requested_market += len(market)
        submitted_within_cap += min(10, len(market))
        over_cap_requests += max(0, len(market) - 10)
        prices = obs.get("market", {}).get("prices", {})
        hires_in_action = 0
        land_buys_in_action = 0
        unlocked_count = len(farm.get("unlocked_quadrants", ["NW"]))
        for order in market:
            if not isinstance(order, (list, tuple)) or not order:
                continue
            name = order[0]
            requested_orders[name] += 1
            item = order[1] if len(order) > 1 else None
            qty = order[2] if len(order) > 2 and isinstance(order[2], (int, float)) else 1
            if name == "HIRE":
                hire_requested_by_day[day] += 1
                purchase_quantity_requested["hire"] += 1
                spending_requested["hire"] += _marginal_hire_cost(hires_in_action)
                hires_in_action += 1
            elif name == "BUY_LAND":
                land_request_days.append(day)
                purchase_quantity_requested["land"] += 1
                spending_requested["land"] += LAND_COST.get(
                    unlocked_count + land_buys_in_action, 0)
                land_buys_in_action += 1
            elif name == "SELL" and item:
                requested_sale_quantity[item] += qty
                requested_sale_prices[item].extend([prices.get(item, 0)] * int(qty))
            elif name == "BUY_SEED" and item:
                purchase_quantity_requested["seed"] += qty
                spending_requested["seed"] += SEED_COST.get(item, 0) * qty
            elif name == "BUY_ANIMAL" and item:
                purchase_quantity_requested["animal"] += qty
                spending_requested["animal"] += ANIMAL_COST.get(item, 0) * qty
            elif name == "BUY_PRODUCT" and item == "WHEAT":
                purchase_quantity_requested["feed"] += qty
                spending_requested["feed"] += prices.get(item, 0) * qty

        animals, crops, _, _, turn_feed_risk, turn_water_risk = _asset_counts(farm)
        unfed_tile_turns += turn_feed_risk
        unwatered_tile_turns += turn_water_risk
        if turn and obs.get("hour") == 0:
            animal_losses += sum(max(0, prior_animals[k] - animals[k]) for k in prior_animals)
            for crop in prior_crops:
                exits = max(0, prior_crops[crop] - crops[crop])
                crop_exit_days[crop].extend([day] * exits)
            # A new weed at refresh is the observable decay/death signal;
            # ordinary one-time harvest removals are not counted as failures.
            prior_weeds = _asset_counts(_farm(steps[turn - 1], player_idx))[2]
            current_weeds = _asset_counts(farm)[2]
            crop_losses += max(0, current_weeds - prior_weeds)
        prior_animals, prior_crops = animals, crops
        opp_animals, opp_crops, _, _, _, _ = _asset_counts(opponent)
        opponent_exposure.update(opp_animals)
        opponent_exposure.update(opp_crops)

    final_animals, final_crops, final_weeds, final_ready, feed_risk, water_risk = _asset_counts(last)
    ending_inventory = _inventory(_private(steps[-1], player_idx))
    status = steps[-1][player_idx].status
    role_shares = {}
    for slot, counts in action_counts.items():
        total = sum(counts.values()) or 1
        role_shares[slot] = {name: counts[name] / total for name in ("work", "move", "pass")}
    daily_hands = {str(day): max(values) for day, values in sorted(hand_count_by_day.items())}
    return {
        "final_money": last.get("money", 0),
        "minimum_cash": min(money_curve),
        "status": status,
        "requested_market_orders": requested_market,
        # Recorded steps expose submitted actions but no per-order execution
        # receipts. Do not manufacture fills from order position: orders can
        # silently fail for affordability, inventory, or other engine rules.
        "executed_market_orders": None,
        "market_execution_observable": False,
        "submitted_market_orders_within_cap": submitted_within_cap,
        "market_order_cap_excess_requests": over_cap_requests,
        "requested_orders_by_type": dict(requested_orders),
        "executed_orders_by_type": None,
        "hire_requests_by_day": dict(hire_requested_by_day),
        "active_hands_by_day": daily_hands,
        "land_purchase_request_days": land_request_days,
        "hand_action_shares": role_shares,
        "requested_sale_quantity_by_product": dict(requested_sale_quantity),
        "requested_sale_price_by_product": {
            item: {"mean": mean(values), "min": min(values), "max": max(values), "n": len(values)}
            for item, values in requested_sale_prices.items() if values
        },
        "requested_purchase_quantity": dict(purchase_quantity_requested),
        "requested_purchase_spending": dict(spending_requested),
        "plant_actions_by_crop": dict(planted),
        "planting_days_by_crop": dict(planting_days),
        "harvest_actions": harvested["actions"],
        "harvest_actions_by_crop": {key: value for key, value in harvested.items() if key != "actions"},
        "harvest_days_by_crop": dict(harvest_days),
        "crop_exit_days": dict(crop_exit_days),
        "escaped_animals_estimate": animal_losses,
        "decayed_crops_estimate": crop_losses,
        "unfed_tile_turns": unfed_tile_turns,
        "unwatered_tile_turns": unwatered_tile_turns,
        "unlocked_unused_tile_turns": unused_tile_turns,
        "ending_assets": {"animals": dict(final_animals), "crops": dict(final_crops), "weeds": final_weeds, "ready_yield": final_ready},
        "ending_inventory": dict(ending_inventory),
        "unsold_ending_inventory": sum(ending_inventory.get(item, 0) for item in PRODUCTS),
        "opponent_production_exposure_tile_turns": dict(opponent_exposure),
        "bankruptcy": min(money_curve) <= 0,
    }


def summarize_episodes(episodes):
    """Compact aggregate used in A/B reports while preserving raw episodes."""
    if not episodes:
        return {"n": 0}
    numeric = ("final_money", "minimum_cash", "market_order_cap_excess_requests",
               "escaped_animals_estimate", "decayed_crops_estimate",
               "unlocked_unused_tile_turns", "unsold_ending_inventory")
    out = {"n": len(episodes)}
    for key in numeric:
        values = [episode[key] for episode in episodes]
        out[key] = {"mean": mean(values), "min": min(values), "max": max(values)}
    out["failures"] = sum(episode["status"] != "DONE" for episode in episodes)
    out["bankruptcies"] = sum(episode["bankruptcy"] for episode in episodes)
    return out
