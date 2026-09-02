"""Generate auditable counterfactual training pairs for the watcher SVM.

The baseline and intervention are separate deterministic replays.  A row is
accepted only when their checkpoint observations and all earlier focal-agent
actions hash identically.  This deliberately favours trustworthy examples
over forcing every requested deviation into the dataset.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from kaggle_environments import make
import kaggle_environments.envs.kaggriculture.kaggriculture as _kenv
from shared_features import FEATURE_ORDER, SCHEMA_HASH, SCHEMA_VERSION, extract_features
import watcher_config as wc


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TURNS_PER_DAY = wc.TURNS_PER_DAY


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL")
PREMIUM_PRODUCTS = ("MELON", "STRAWBERRY", "MILK", "WOOL")
STRUCTURES = ("COOP", "PASTURE")
SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
         "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50,
             "STRAWBERRY": 100, "MELON": 80}
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
LAND_COST = {1: 1000, 2: 2000, 3: 4000}
THROUGHPUT = {"WHEAT": 400, "CARROT": 450, "TOMATO": 200, "STRAWBERRY": 100,
              "MELON": 300, "EGG": 332, "MILK": 122, "WOOL": 105}
DEVIATIONS = ("crop_add_two", "crop_remove_two", "crop_redirect_two",
              "animal_add_one", "animal_defer_one", "hand_add_one",
              "hand_remove_one", "land_advance", "land_defer",
              "crop_divest_all", "animal_defer_all")
# Market-only full-divest kinds (repair.md Phase A). Unlike the bounded
# remove/defer kinds they strip *every* matching future purchase on *every*
# turn from the checkpoint to the end of the label window, driving incremental
# exposure to X to zero without touching already-planted tiles.
DIVEST_KINDS = ("crop_divest_all", "animal_defer_all")
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
# 60/15/15/10 train/val/test/confirm. `confirm` is opened once by the trainer,
# after a candidate is otherwise finalised, and never feeds any decision back.
SPLIT_BUCKETS = (("train", 12), ("val", 15), ("test", 18), ("confirm", 20))


def checkpoint_specs(episode_steps=720, turns_per_day=24, unlock_interval=3):
    """Unique hour-zero checkpoints, with all coincident reason flags."""
    last_day = (episode_steps - 1) // turns_per_day
    rows = []
    for day in range(last_day + 1):
        reasons = ["day_start"]
        if day > 0 and day % unlock_interval == 0:
            reasons.append("shop_unlock")
        rows.append({"step": day * turns_per_day, "day": day, "hour": 0,
                     "reasons": reasons})
    return rows


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


class RecordingAgent:
    def __init__(self, base, checkpoint_step, deviation=None):
        self.base, self.checkpoint_step, self.deviation = base, checkpoint_step, deviation
        self.pre_actions, self.checkpoint_obs, self.post_checkpoint_obs = [], None, None
        self.audit = {"attempted": 0, "applied": 0, "eligible": False,
                      "execution_verified": False, "events": []}

    def __call__(self, obs):
        action = copy.deepcopy(self.base(obs))
        step = obs.get("step", obs["day"] * 24 + obs["hour"])
        if step < self.checkpoint_step:
            self.pre_actions.append(action)
        elif step == self.checkpoint_step and self.checkpoint_obs is None:
            self.checkpoint_obs = copy.deepcopy(obs)
        elif step == self.checkpoint_step + 1 and self.post_checkpoint_obs is None:
            self.post_checkpoint_obs = copy.deepcopy(obs)
        if self.deviation and step >= self.checkpoint_step:
            action = self._intervene(action, obs)
        return action

    def _intervene(self, action, obs):
        kind, item, target = self.deviation["kind"], self.deviation.get("item"), self.deviation.get("target")
        if kind in DIVEST_KINDS:
            return self._intervene_divest(action, obs, kind, item)
        remaining = self.deviation.setdefault("remaining", 2 if "two" in kind else 1)
        if remaining <= 0: return action
        self.audit["attempted"] += 1
        market = action.setdefault("market", [])
        before = copy.deepcopy(action)
        reason = _intervention_eligibility(kind, item, action, obs, remaining, target)
        if reason is not None:
            counts = self.audit.setdefault("ineligible_reason_counts", {})
            counts[reason] = counts.get(reason, 0) + 1
            return action
        self.audit["eligible"] = True
        if kind in ("crop_add_two", "animal_add_one", "hand_add_one", "land_advance"):
            order = (["BUY_SEED", item, remaining] if kind == "crop_add_two" else
                     ["BUY_ANIMAL", item, 1] if kind == "animal_add_one" else
                     ["HIRE"] if kind == "hand_add_one" else ["BUY_LAND"])
            if len(market) < 10: market.append(order); self.deviation["remaining"] = 0
        elif kind in ("crop_remove_two", "animal_defer_one", "hand_remove_one", "land_defer"):
            op = {"crop_remove_two": "BUY_SEED", "animal_defer_one": "BUY_ANIMAL",
                  "hand_remove_one": "HIRE", "land_defer": "BUY_LAND"}[kind]
            for i, order in enumerate(market):
                if order and order[0] == op and (op != "BUY_SEED" or order[1] == item):
                    qty = min(remaining, order[2] if len(order) > 2 else 1)
                    if len(order) > 2 and order[2] > qty: order[2] -= qty
                    else: market.pop(i)
                    self.deviation["remaining"] -= qty; break
        elif kind == "crop_redirect_two":
            for i, order in enumerate(market):
                if (order and order[0] == "BUY_SEED" and len(order) > 2
                        and order[1] != target and (item is None or order[1] == item)):
                    qty = min(remaining, order[2])
                    remainder = order[2] - qty
                    if remainder == 0:
                        order[1] = target
                    elif len(market) < 10:
                        # Only part of the order is redirected: keep the
                        # un-redirected units as their own order instead of
                        # silently dropping them.
                        order[2] = remainder
                        market.insert(i + 1, ["BUY_SEED", target, qty])
                    else:
                        continue  # no room to split without losing units
                    self.deviation["remaining"] -= qty
                    break
        if action != before:
            self.audit["applied"] += 1
            self.audit["events"].append({"step": obs.get("step"), "before": before, "after": copy.deepcopy(action)})
        return action

    def _intervene_divest(self, action, obs, kind, item):
        """Strip every future BUY_SEED X / BUY_ANIMAL <species> order this turn,
        and opportunistically DIG an X plant a focal unit already stands on.
        Active only from the checkpoint to the end of the label window."""
        step = obs.get("step", obs.get("day", 0) * 24 + obs.get("hour", 0))
        if step > _divest_window_end(self.checkpoint_step, kind, item):
            return action
        self.audit["attempted"] += 1
        before = copy.deepcopy(action)
        reason = _intervention_eligibility(kind, item, action, obs, 0)
        if reason is not None:
            counts = self.audit.setdefault("ineligible_reason_counts", {})
            counts[reason] = counts.get(reason, 0) + 1
            return action
        self.audit["eligible"] = True

        op = "BUY_SEED" if kind == "crop_divest_all" else "BUY_ANIMAL"
        market = action.get("market", [])
        kept = [o for o in market if not _divest_matches(o, op, item)]
        if len(kept) != len(market):
            action["market"] = kept

        if kind == "crop_divest_all":
            farm = _farm(obs)
            tiles = farm.get("tiles", [])
            hands_action = action.get("hands", [])
            positions = [farm.get("farmer")] + list(farm.get("hands", []))
            for u_idx, pos in enumerate(positions):
                if not pos:
                    continue
                x, y = pos[0], pos[1]
                if not (0 <= y < len(tiles) and 0 <= x < len(tiles[y])):
                    continue
                tile = tiles[y][x]
                if not (isinstance(tile, dict) and tile.get("crop") == item):
                    continue
                cur = action["farmer"] if u_idx == 0 else (
                    hands_action[u_idx - 1] if u_idx - 1 < len(hands_action) else ["PASS"])
                # Only steal a genuinely idle (PASS) unit's turn. A unit that is
                # moving (NORTH/...) is en route to husbandry it still owes other
                # tiles; hijacking it -- or a unit mid-WATER/HARVEST/FEED --
                # strands those tiles and trips the missed-watering /
                # escaped-animal hard-safety reject, which was rejecting the
                # majority of :avoid pairs on an early checkpoint. The market
                # strip already drives incremental X exposure to zero on its own.
                if not cur or cur[0] != "PASS":
                    continue
                if u_idx == 0:
                    action["farmer"] = ["DIG"]
                else:
                    while len(hands_action) <= u_idx - 1:
                        hands_action.append(["PASS"])
                    hands_action[u_idx - 1] = ["DIG"]
                    action["hands"] = hands_action

        if action != before:
            self.audit["applied"] += 1
            self.audit["events"].append({"step": obs.get("step"), "before": before, "after": copy.deepcopy(action)})
        return action


def _divest_matches(order, op, item):
    """A future BUY_SEED X / BUY_ANIMAL <species> order the divest strips."""
    return bool(order) and order[0] == op and len(order) > 1 and order[1] == item


def _divest_window_end(checkpoint_step, kind, item):
    product = item if kind == "crop_divest_all" else ANIMAL_PRODUCT.get(item, item)
    horizon = wc.PRODUCT_HORIZON_DAYS.get(product, wc.LABEL_HORIZON_DAYS_PRODUCT)
    return checkpoint_step + horizon * TURNS_PER_DAY


def _unit_on_crop_tile(obs, item):
    farm = _farm(obs)
    tiles = farm.get("tiles", [])
    for pos in [farm.get("farmer")] + list(farm.get("hands", [])):
        if not pos:
            continue
        x, y = pos[0], pos[1]
        if 0 <= y < len(tiles) and 0 <= x < len(tiles[y]):
            tile = tiles[y][x]
            if isinstance(tile, dict) and tile.get("crop") == item:
                return True
    return False


def _farm(obs):
    player = int(obs.get("player", 0))
    farms = obs.get("farms", [])
    return farms[player] if player < len(farms) else {}


def _private_count(obs, bucket, item):
    return int(obs.get("private", {}).get(bucket, {}).get(item, 0))


def _matching_structures(farm, animal):
    wanted = "COOP" if animal == "GOOSE" else "PASTURE"
    return sum(1 for row in farm.get("tiles", []) for tile in row
               if isinstance(tile, dict) and tile.get("kind") == wanted and not tile.get("animal"))


def _future_crop_slots(farm):
    return sum(tile is None for row in farm.get("tiles", []) for tile in row)


def _hire_cost_after(existing_hires):
    a, b = 1, 1
    for _ in range(existing_hires):
        a, b = b, a + b
    return a


def _committed_spend(market, obs):
    prices = obs.get("market", {}).get("prices", {})
    unlocked = len(_farm(obs).get("unlocked_quadrants", ["NW"]))
    spend = 0.0
    hires = 0
    land_buys = 0
    for order in market:
        if not order: continue
        op = order[0]
        qty = int(order[2]) if len(order) > 2 else 1
        if op == "BUY_SEED": spend += SEED_COST.get(order[1], float("inf")) * qty
        elif op == "BUY_ANIMAL": spend += ANIMAL_COST.get(order[1], float("inf")) * qty
        elif op == "BUY_PRODUCT": spend += prices.get(order[1], float("inf")) * qty
        elif op == "HIRE":
            spend += _hire_cost_after(hires)
            hires += 1
        elif op == "BUY_LAND":
            spend += LAND_COST.get(unlocked + land_buys, float("inf"))
            land_buys += 1
    return spend


def _intervention_eligibility(kind, item, action, obs, remaining, target=None):
    """Return None only when the altered order is legal and actionable now."""
    market = action.get("market", [])
    farm = _farm(obs)
    money = float(farm.get("money", 0)) - _committed_spend(market, obs)
    if len(market) >= 10 and kind in ("crop_add_two", "animal_add_one", "hand_add_one", "land_advance"):
        return "market_order_cap_full"
    if kind == "crop_add_two":
        if _future_crop_slots(farm) < remaining: return "no_future_crop_slot"
        if money < SEED_COST[item] * remaining: return "unaffordable_seed"
    elif kind == "animal_add_one":
        if _matching_structures(farm, item) < 1: return "no_available_structure"
        if money < ANIMAL_COST[item]: return "unaffordable_animal"
    elif kind == "hand_add_one":
        hires = sum(bool(o) and o[0] == "HIRE" for o in market)
        if money < _hire_cost_after(hires): return "unaffordable_hire"
    elif kind == "land_advance":
        unlocked = len(farm.get("unlocked_quadrants", ["NW"]))
        if unlocked not in LAND_COST: return "no_locked_land"
        if money < LAND_COST[unlocked]: return "unaffordable_land"
    elif kind in ("crop_remove_two", "animal_defer_one", "hand_remove_one", "land_defer"):
        op = {"crop_remove_two": "BUY_SEED", "animal_defer_one": "BUY_ANIMAL",
              "hand_remove_one": "HIRE", "land_defer": "BUY_LAND"}[kind]
        if not any(o and o[0] == op and (op != "BUY_SEED" or o[1] == item) for o in market):
            return "no_matching_future_purchase"
    elif kind == "crop_redirect_two":
        if not any(o and o[0] == "BUY_SEED" and len(o) > 2 and o[1] != target
                   and (item is None or o[1] == item) for o in market):
            return "no_redirectable_purchase"
    elif kind in ("crop_divest_all", "animal_defer_all"):
        op = "BUY_SEED" if kind == "crop_divest_all" else "BUY_ANIMAL"
        buys_x = any(_divest_matches(o, op, item) for o in market)
        # A crop divest can also bite by digging a plant a focal unit stands on;
        # an animal defer has no standing-on-tile lever (DIG can't remove animals).
        on_x_tile = kind == "crop_divest_all" and _unit_on_crop_tile(obs, item)
        if not buys_x and not on_x_tile:
            return "focal_holds_no_x_and_buys_no_x"
    return None


def _obs_at(env, step, idx):
    if step < 0 or step >= len(env.steps):
        return None
    return env.steps[step][idx].observation


def _crop_units(obs, item):
    """Seeds on hand plus tiles already planted with the crop -- 'seeds acquired
    to date'. A seed that is bought and immediately planted leaves the seed
    count unchanged, so the raw seed count alone misses the purchase."""
    farm = _farm(obs)
    planted = sum(1 for row in farm.get("tiles", []) for tile in row
                  if isinstance(tile, dict) and tile.get("crop") == item)
    return _private_count(obs, "seeds", item) + planted


def _animal_units(obs, item):
    farm = _farm(obs)
    placed = sum(1 for row in farm.get("tiles", []) for tile in row
                 if isinstance(tile, dict) and tile.get("animal") == item)
    return _private_count(obs, "shed", item) + placed


def _hand_count(obs):
    return len(_farm(obs).get("hands", []))


def _quadrant_count(obs):
    return len(_farm(obs).get("unlocked_quadrants", []))


_VERIFY_METRICS = {
    "crop_add_two": ("crop_units", True),
    "crop_remove_two": ("crop_units", False),
    "crop_divest_all": ("crop_units", False),
    "crop_redirect_two": ("crop_units", True),
    "animal_add_one": ("animal_units", True),
    "animal_defer_one": ("animal_units", False),
    "animal_defer_all": ("animal_units", False),
    "hand_add_one": ("hands", True),
    "hand_remove_one": ("hands", False),
    "land_advance": ("quadrants", True),
    "land_defer": ("quadrants", False),
}
_VERIFY_EXTRACTORS = {
    "crop_units": _crop_units, "animal_units": _animal_units,
    "hands": lambda obs, _item: _hand_count(obs),
    "quadrants": lambda obs, _item: _quadrant_count(obs),
}


# The full-divest kinds are verified differently (repair.md Phase B): not "the
# X unit count dropped by >= 1 at some step" but "it stayed at least
# DIVEST_VERIFY_MIN_GAP below baseline through the end of the window and never
# rose above it" -- a sustained strip, not a transient dip.
_SUSTAINED_VERIFY_KINDS = ("crop_divest_all", "animal_defer_all")


def _verify_sustained_gap(extract, metric, item, benv, ienv, idx, start, end):
    """The strip is verified only if the focal's X unit count opens a gap of at
    least DIVEST_VERIFY_MIN_GAP below baseline and holds it at every step from
    that onset to the end of the window -- a sustained divest, not a blip."""
    K = wc.DIVEST_VERIFY_MIN_GAP
    onset = None
    min_gap_after_onset = None
    for step in range(start, end):
        gap = (extract(_obs_at(benv, step, idx), item)
               - extract(_obs_at(ienv, step, idx), item))
        if onset is None:
            if gap >= K:
                onset = step
                min_gap_after_onset = gap
        else:
            min_gap_after_onset = min(min_gap_after_onset, gap)
    last = max(start, end - 1)
    ok = onset is not None and min_gap_after_onset >= K
    return ok, {"metric": metric, "window": [start, end], "min_gap_required": K,
                "onset_step": onset, "min_gap_after_onset": min_gap_after_onset,
                "baseline_end": extract(_obs_at(benv, last, idx), item),
                "intervention_end": extract(_obs_at(ienv, last, idx), item)}


def _execution_verified(kind, item, benv, ienv, idx, applied_step, window_steps):
    """Verify the intended state delta held at *any* env step in a window after
    the order, not at a single fixed step. The window absorbs planting lag,
    onboarding travel and land-unlock lag that a one-step check trips over."""
    if not isinstance(applied_step, int):
        return False, {"reason": "intervention_never_applied"}
    spec = _VERIFY_METRICS.get(kind)
    if spec is None:
        return False, {"reason": "unknown_deviation_kind", "kind": kind}
    metric, want_increase = spec
    extract = _VERIFY_EXTRACTORS[metric]
    start = applied_step + 1
    end = min(start + window_steps, len(benv.steps), len(ienv.steps))
    if kind in _SUSTAINED_VERIFY_KINDS:
        return _verify_sustained_gap(extract, metric, item, benv, ienv, idx, start, end)
    for step in range(start, end):
        bv = extract(_obs_at(benv, step, idx), item)
        iv = extract(_obs_at(ienv, step, idx), item)
        if (iv > bv) if want_increase else (iv < bv):
            return True, {"metric": metric, "step": step, "window": [start, end],
                          "baseline": bv, "intervention": iv}
    last = max(start, end - 1)
    return False, {"metric": metric, "window": [start, end],
                   "baseline_end": extract(_obs_at(benv, last, idx), item),
                   "intervention_end": extract(_obs_at(ienv, last, idx), item)}


def _metrics(env, idx):
    feed = water = cap = missed = 0
    escaped = 0
    previous_tiles = None
    for pair in env.steps:
        state = pair[idx]
        obs = state.observation
        farm = obs["farms"][idx]
        tiles = farm["tiles"]
        for y, row in enumerate(tiles):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and tile.get("kind") in STRUCTURES and tile.get("animal"):
                    feed += int(tile.get("consecutive_unfed", 0) > 0)
                elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
                    water += int(tile.get("consecutive_unwatered", 0) > 0)
                if previous_tiles is not None:
                    old = previous_tiles[y][x]
                    if isinstance(old, dict) and old.get("kind") in STRUCTURES and old.get("animal"):
                        escaped += int(isinstance(tile, dict) and tile.get("kind") == old.get("kind") and not tile.get("animal"))
                    if isinstance(old, dict) and old.get("kind") == "PLANT" and old.get("consecutive_unwatered", 0) > 0:
                        missed += int(isinstance(tile, dict) and tile.get("kind") == "WEED")
        previous_tiles = tiles
        action = state.action or {}
        cap += int(len(action.get("market", [])) > 10)
    final_state = env.steps[-1][idx]
    final_private = final_state.observation.get("private", {})
    money_curve = [pair[idx].observation["farms"][idx].get("money", 0)
                   for pair in env.steps]
    ending = sum(final_private.get("shed", {}).get(x, 0) for x in PRODUCTS)
    overflow = int(sum(final_private.get("shed", {}).values()) >= 100)
    return {"feed_risk_tile_turns": feed, "water_risk_tile_turns": water,
            "escaped_animals": escaped, "missed_watering_deaths": missed,
            "shed_overflow_indicators": overflow, "ending_saleable_inventory": ending,
            "market_order_cap_violations": cap,
            "episode_status_failures": int(final_state.status != "DONE"),
            "cash_trough_touched": int(bool(money_curve) and min(money_curve) <= 0)}


# Hard rejects are genuine causal-validity failures: the intervention broke the
# game in a way that makes the pair a bad example of *either* class -- an animal
# starved, plants died of thirst, the episode aborted. Soft flags are recorded
# on the row but never reject it -- a profitable-but-messy pair (a few extra
# unsold units, a deeper early cash dip, a turn that over-fills the order queue
# the env silently truncates anyway) is a legitimate example, and the follower
# clamps by cash / deadline capacity at runtime.
HARD_SAFETY_METRICS = ("escaped_animals", "missed_watering_deaths",
                       "episode_status_failures")
SOFT_SAFETY_METRICS = ("feed_risk_tile_turns", "water_risk_tile_turns",
                       "shed_overflow_indicators", "ending_saleable_inventory",
                       "cash_trough_touched", "market_order_cap_violations")


def safety_split(baseline, intervention):
    """Return (hard_fail, soft_flags). hard_fail rejects the pair; soft_flags is
    a dict of the lower-is-better metrics that worsened without rejecting it."""
    hard_fail = any(intervention.get(key, 0) > baseline.get(key, 0)
                    for key in HARD_SAFETY_METRICS)
    soft_flags = {key: {"baseline": baseline.get(key, 0), "intervention": intervention.get(key, 0)}
                  for key in SOFT_SAFETY_METRICS
                  if intervention.get(key, 0) > baseline.get(key, 0)}
    return hard_fail, soft_flags


def label_delta(delta):
    """Season-end money classifier, kept only for the recorded ``season_end_label``
    audit field. The trained label is the bounded-horizon blend below."""
    if delta < -250: return -1
    if delta > 250: return 1
    return 0


def _held_product_value(obs, item):
    """Sellable X plus X still accruing on a plant or animal (yield_units), at
    the current price. Counting work-in-progress lets a short window still see
    the value of a slow crop that was planted but has not yielded yet."""
    if obs is None:
        return 0.0
    private = obs.get("private", {})
    units = private.get("shed", {}).get(item, 0)
    units += sum(inv.get(item, 0) for inv in private.get("inventories", []))
    farm = _farm(obs)
    animal = next((a for a, p in ANIMAL_PRODUCT.items() if p == item), None)
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if item in CROPS and tile.get("crop") == item:
                units += tile.get("yield_units", 0)
            elif animal is not None and tile.get("animal") == animal:
                units += tile.get("yield_units", 0)
    price = obs.get("market", {}).get("prices", {}).get(item, BASE_PRICE.get(item, 0.0))
    return units * price


def _window_revenue(sales_log, item, start_step, end_step):
    return sum(entry["sold"].get(item, {}).get("revenue", 0.0)
               for entry in sales_log if start_step <= entry["step"] < end_step)


def _money_at(env, step, idx):
    obs = _obs_at(env, step, idx)
    return float(_farm(obs).get("money", 0.0)) if obs is not None else 0.0


def label_product(product, checkpoint_step, horizon_days, benv, ienv, idx,
                  baseline_sales, intervention_sales):
    """Signed only when the near-term total-money delta and the X-scoped
    value-flow delta agree in sign and the money delta clears the band. A signed
    row is then causally about exposure to X, not about where freed capital
    happened to land."""
    end = min(checkpoint_step + horizon_days * TURNS_PER_DAY, len(benv.steps) - 1)
    d_money_b = _money_at(benv, end, idx) - _money_at(benv, checkpoint_step, idx)
    d_money_i = _money_at(ienv, end, idx) - _money_at(ienv, checkpoint_step, idx)
    money_delta = d_money_i - d_money_b

    b_rev = _window_revenue(baseline_sales, product, checkpoint_step, end)
    i_rev = _window_revenue(intervention_sales, product, checkpoint_step, end)
    b_flow = b_rev + _held_product_value(_obs_at(benv, end, idx), product)
    i_flow = i_rev + _held_product_value(_obs_at(ienv, end, idx), product)
    x_flow_delta = i_flow - b_flow

    band = max(wc.LABEL_NEUTRAL_BAND_FRACTION * abs(b_rev),
               wc.LABEL_NEUTRAL_BAND_FLOOR_PRODUCT)
    components = {"basis": "product_blend", "horizon_days": horizon_days,
                 "window": [checkpoint_step, end], "money_delta": money_delta,
                 "x_flow_delta": x_flow_delta, "band": band,
                 "baseline_window_revenue": b_rev,
                 "baseline_window_money_delta": d_money_b}
    if abs(money_delta) <= band or x_flow_delta == 0:
        return 0, components
    if (money_delta > 0) == (x_flow_delta > 0):
        return (1 if money_delta > 0 else -1), components
    return 0, components


def label_capacity(checkpoint_step, horizon_days, benv, ienv, idx):
    """land_expansion / labor_expansion: near-term total-money delta. The
    intervention already carries the up-front cost as a negative, so a capacity
    buy that merely earns its keep lands neutral."""
    end = min(checkpoint_step + horizon_days * TURNS_PER_DAY, len(benv.steps) - 1)
    d_money_b = _money_at(benv, end, idx) - _money_at(benv, checkpoint_step, idx)
    d_money_i = _money_at(ienv, end, idx) - _money_at(ienv, checkpoint_step, idx)
    money_delta = d_money_i - d_money_b
    band = max(wc.LABEL_NEUTRAL_BAND_FRACTION * abs(d_money_b),
               wc.LABEL_NEUTRAL_BAND_FLOOR_CAPACITY)
    components = {"basis": "capacity_money", "horizon_days": horizon_days,
                 "window": [checkpoint_step, end], "money_delta": money_delta,
                 "band": band, "baseline_window_money_delta": d_money_b}
    return (1 if money_delta > band else -1 if money_delta < -band else 0), components


def compute_label(output_id, checkpoint_step, benv, ienv, idx,
                  baseline_sales, intervention_sales):
    if output_id in ("land_expansion", "labor_expansion"):
        return label_capacity(checkpoint_step, wc.LABEL_HORIZON_DAYS_CAPACITY,
                              benv, ienv, idx)
    product = output_id.split(":")[1]
    horizon = wc.PRODUCT_HORIZON_DAYS.get(product, wc.LABEL_HORIZON_DAYS_PRODUCT)
    return label_product(product, checkpoint_step, horizon,
                         benv, ienv, idx, baseline_sales, intervention_sales)


def _install_sales_probe(player_indices):
    """Wrap the env's per-step market phase to record realised sell volume and
    revenue by product for each player in ``player_indices``. During the market
    phase only SELL reduces a player's shed (BUY_PRODUCT adds WHEAT/FERTILIZER
    only), so the per-step shed decrease is exactly units sold; revenue uses the
    mean of the pre/post-refresh price. Log entries are
    ``{"step", "player", "sold": {item: {"units", "revenue"}}}``.
    Returns (log, original) -- the caller must restore ``original``."""
    idxs = [player_indices] if isinstance(player_indices, int) else list(player_indices)
    log = []
    original = _kenv._process_market

    def wrapped(state, env):
        obs0 = state[0].observation
        step = int(_kenv.get(obs0, "step", 0))
        sheds_before = {p: dict(state[p].observation.private.get("shed", {})) for p in idxs}
        prices_before = dict(obs0.market["prices"])
        original(state, env)
        prices_after = obs0.market["prices"]
        for p in idxs:
            shed_after = state[p].observation.private.get("shed", {})
            sold = {}
            for item in PRODUCTS:
                units = sheds_before[p].get(item, 0) - shed_after.get(item, 0)
                if units > 0:
                    px = 0.5 * (prices_before.get(item, BASE_PRICE[item]) +
                                prices_after.get(item, BASE_PRICE[item]))
                    sold[item] = {"units": units, "revenue": units * px}
            if sold:
                log.append({"step": step, "player": p, "sold": sold})

    _kenv._process_market = wrapped
    return log, original


def _player_sales(sales_log, player):
    return [entry for entry in sales_log if entry.get("player", player) == player]


def _opponent_volume_days(sales_log, player, product, units_per_day):
    """Days on which ``player`` realised at least ``units_per_day`` units of
    ``product``. The first such day is the checkpoint floor for product:*:avoid
    and premium product:*:increase, so the label window contains the opponent's
    price crash rather than ending before the glut matures (repair.md Phase C)."""
    per_day = {}
    for entry in sales_log:
        if entry.get("player") != player:
            continue
        units = entry["sold"].get(product, {}).get("units", 0)
        if units:
            day = entry["step"] // TURNS_PER_DAY
            per_day[day] = per_day.get(day, 0) + units
    return sorted(day for day, units in per_day.items() if units >= units_per_day)


_REMOVAL_OP = {"crop_remove_two": "BUY_SEED", "animal_defer_one": "BUY_ANIMAL",
               "hand_remove_one": "HIRE", "land_defer": "BUY_LAND",
               "crop_divest_all": "BUY_SEED", "animal_defer_all": "BUY_ANIMAL"}


def _emission_key(op, item):
    return f"{op}:{item}" if op in ("BUY_SEED", "BUY_ANIMAL") else op


def _focal_order_days(env, idx):
    """{"BUY_SEED:MELON": [days], "HIRE": [days], ...} -- every day the focal
    emits each kind of market order. Precomputed once per (family, opponent) for
    the probe cache (repair.md Phase F); the focal is opponent-blind so this is
    stable enough to reuse across seeds."""
    days: dict[str, set] = {}
    for step, pair in enumerate(env.steps):
        action = pair[idx].action
        orders = action.get("market", []) if isinstance(action, dict) else []
        for order in orders:
            if not order:
                continue
            op = order[0]
            if op not in ("BUY_SEED", "BUY_ANIMAL", "HIRE", "BUY_LAND"):
                continue
            item = order[1] if op in ("BUY_SEED", "BUY_ANIMAL") and len(order) > 1 else None
            days.setdefault(_emission_key(op, item), set()).add(step // TURNS_PER_DAY)
    return {key: sorted(value) for key, value in days.items()}


def _order_emission_days(env, idx, kind, item):
    """Days on which the focal actually emits the order this removal/defer
    deviation cancels. Drives the intervention checkpoint off day 0."""
    return _focal_order_days(env, idx).get(_emission_key(_REMOVAL_OP[kind], item), [])


def _probe_summary(env, focal_idx, sales_log):
    """The two per-(family, opponent) facts a checkpoint needs: which days the
    focal emits each order, and which days the opponent floods each product."""
    opp = 1 - focal_idx
    return {"focal_order_days": _focal_order_days(env, focal_idx),
            "opponent_volume_days": {
                product: _opponent_volume_days(sales_log, opp, product,
                                               wc.OPPONENT_VOLUME_UNITS_PER_DAY)
                for product in PRODUCTS}}


def _run(base, opponent, idx, seed, steps, checkpoint, deviation=None):
    wrapped = RecordingAgent(base, checkpoint, copy.deepcopy(deviation))
    # Kaggle's callable preparation may copy callable instances.  A function
    # closure keeps the recorder identity visible to this process.
    def focal(observation):
        return wrapped(observation)
    agents = [focal, opponent] if idx == 0 else [opponent, focal]
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    # Both players: focal sales feed the label, opponent sales feed the
    # opponent-volume checkpoint floor (repair.md Phase C).
    sales_log, original_market = _install_sales_probe((idx, 1 - idx))
    try:
        env.run(agents)
    finally:
        _kenv._process_market = original_market
    if checkpoint < steps and wrapped.checkpoint_obs is None:
        raise RuntimeError(f"recorder never observed checkpoint step {checkpoint}; "
                           f"the env may have copied the focal callable")
    money = env.steps[-1][idx].observation["farms"][idx]["money"]
    return wrapped, env, money, _metrics(env, idx), sales_log


def random_deviation(rng):
    kind = rng.choice(DEVIATIONS)
    item = (rng.choice(CROPS) if "crop" in kind else
            rng.choice(ANIMALS) if "animal" in kind else None)
    target = rng.choice([x for x in CROPS if x != item]) if kind == "crop_redirect_two" else None
    return {"kind": kind, "item": item, "target": target, "remaining": 2 if "two" in kind else 1}


def deviation_for_output(output_id, rng):
    """Choose a deterministic intervention that supervises ``output_id``."""
    if output_id in ("labor_expansion", "land_expansion"):
        kind = rng.choice(("hand_add_one", "hand_remove_one") if output_id == "labor_expansion"
                          else ("land_advance", "land_defer"))
        return {"kind": kind, "item": None, "target": None,
                "remaining": 1}
    parts = output_id.split(":")
    if (len(parts) != 3 or parts[0] != "product" or parts[1] not in PRODUCTS or
            parts[2] not in ("increase", "avoid")):
        raise ValueError(f"unsupported output_id: {output_id}")
    product, direction = parts[1], parts[2]
    if product in CROPS:
        if direction == "avoid":
            # Full market divest over the whole label window (repair.md Phase B):
            # crop_remove_two only clipped 2 units on one turn -- too small to
            # move a bounded-horizon money label.
            return {"kind": "crop_divest_all", "item": product, "target": None, "remaining": 2}
        # Half the increase rows come from redirecting whatever crop the focal is
        # already buying, so product:X:increase is supervised by both the add and
        # the redirect path (repair-plan 2.6). item=None -> redirect the first
        # non-target BUY_SEED, so this does not need the focal to buy one
        # specific source crop.
        if rng.random() < 0.5:
            return {"kind": "crop_redirect_two", "item": None, "target": product, "remaining": 2}
        return {"kind": "crop_add_two", "item": product, "target": None, "remaining": 2}
    animal = next(key for key, value in ANIMAL_PRODUCT.items() if value == product)
    kind = "animal_add_one" if direction == "increase" else "animal_defer_all"
    return {"kind": kind, "item": animal, "target": None, "remaining": 1}


REMOVAL_KINDS = ("crop_remove_two", "animal_defer_one", "hand_remove_one", "land_defer",
                 "crop_divest_all", "animal_defer_all")


def _volume_floor_applies(kind, product):
    """The opponent-volume checkpoint floor applies to every product:*:avoid
    kind and to premium product:*:increase (add / redirect / animal_add). For
    all of these the label can only be signed if the window sees the crash."""
    if not product:
        return False
    if kind in ("crop_divest_all", "animal_defer_all", "crop_remove_two", "animal_defer_one"):
        return True
    if kind in ("crop_add_two", "crop_redirect_two", "animal_add_one"):
        return product in PREMIUM_PRODUCTS
    return False


def checkpoint_for_deviation(rng, deviation, episode_steps, preferred_days=None, min_day=0):
    """Schedule interventions early enough to affect an actionable future slot.
    ``min_day`` floors the checkpoint at the opponent's volume day so the label
    window still contains the price crash (repair.md Phase C)."""
    specs = checkpoint_specs(episode_steps)
    last = len(specs) - 1
    min_day = max(0, min(int(min_day), last))
    kind = deviation["kind"]
    if kind in REMOVAL_KINDS:
        # A removal/defer only registers if it is active before the focal emits
        # the matching purchase. preferred_days is that emission set (from a
        # probe run). Pick from its earlier half: early enough to still catch a
        # later multi-unit emission, varied enough for split/family diversity.
        # Day 0 is only right when the focal buys on day 0.
        all_days = sorted(d for d in (preferred_days or []) if 0 <= d <= last)
        # Prefer emission days at/after the volume floor (window catches the
        # crash). If the focal emits nothing that late -- e.g. an animal herd
        # bought once in the opening -- fall back to the real emission days so
        # the strip still bites, rather than a floored checkpoint it cannot act
        # on (repair.md Phase C: the floor degrades gracefully).
        pool = [d for d in all_days if d >= min_day] or all_days
        if not pool:
            return specs[min_day]
        return specs[rng.choice(pool[:max(1, (len(pool) + 1) // 2)])]
    last_day = min(last, 12 if kind.startswith(("crop", "animal")) else 18)
    window = specs[:last_day + 1]
    eligible = [spec for spec in window if spec["day"] >= min_day] or window
    preferred = [spec for spec in eligible if preferred_days and spec["day"] in preferred_days]
    return rng.choice(preferred or eligible)


def split_for_seed_group(seed_group):
    """Stable 70/15/15 split; all rows from a seed group resolve identically."""
    bucket = int(hashlib.sha256(str(seed_group).encode()).hexdigest()[:8], 16) % 20
    for name, upper_bound in SPLIT_BUCKETS:
        if bucket < upper_bound:
            return name
    raise AssertionError("unreachable split bucket")


def modeled_output_id(deviation):
    """Map a single intervention to the one classifier it supervises."""
    kind, item = deviation["kind"], deviation.get("item")
    if kind == "crop_redirect_two":
        return f"product:{deviation['target']}:increase"
    if kind == "crop_add_two": return f"product:{item}:increase"
    if kind in ("crop_remove_two", "crop_divest_all"): return f"product:{item}:avoid"
    if kind == "animal_add_one": return f"product:{ANIMAL_PRODUCT[item]}:increase"
    if kind in ("animal_defer_one", "animal_defer_all"): return f"product:{ANIMAL_PRODUCT[item]}:avoid"
    if kind.startswith("hand_"): return "labor_expansion"
    if kind.startswith("land_"): return "land_expansion"
    raise ValueError(f"deviation has no modeled output: {deviation!r}")


def build_tasks(pairs, seed_base, opponents, episode_steps):
    """Construct the stable task order used by sequential and parallel runs."""
    return [{"pair_index": n, "seed": seed_base + n,
             "opponent_name": opponents[n % len(opponents)],
             "episode_steps": episode_steps} for n in range(pairs)]


def generate_pair(base, opponent, opponent_name, seed, pair_index, episode_steps=720,
                  target_output=None, strategy_family="strategy_follower", checkpoint_days=None,
                  probe_cache=None):
    rng = random.Random((seed << 16) ^ pair_index)
    deviation = (deviation_for_output(target_output, rng) if target_output
                 else random_deviation(rng))
    idx = pair_index % 2
    kind = deviation["kind"]
    output_id = modeled_output_id(deviation)
    product = output_id.split(":")[1] if output_id.startswith("product:") else None
    floor_applies = _volume_floor_applies(kind, product)

    probe_days = None
    volume_days = None
    if not checkpoint_days and (kind in REMOVAL_KINDS or floor_applies):
        # Which days the focal emits the order a removal/defer cancels (so the
        # checkpoint does not pin to day 0 -- repair-plan C1), and which days the
        # opponent floods the product (so the label window contains the price
        # crash -- repair.md Phase C). Both come from one baseline replay, or
        # from a precomputed per-(family, opponent) cache (repair.md Phase F).
        cached = (probe_cache or {}).get(f"{strategy_family}|{opponent_name}")
        if cached is not None:
            summary = cached
        else:
            probe = _run(base, opponent, idx, seed, episode_steps, episode_steps + 1)
            summary = _probe_summary(probe[1], idx, probe[4])
        if kind in REMOVAL_KINDS:
            probe_days = summary["focal_order_days"].get(
                _emission_key(_REMOVAL_OP[kind], deviation.get("item")), [])
        if floor_applies:
            volume_days = summary["opponent_volume_days"].get(product, [])

    min_day = volume_days[0] if volume_days else 0
    spec = checkpoint_for_deviation(rng, deviation, episode_steps,
                                    checkpoint_days or probe_days, min_day=min_day)
    checkpoint_step = spec["step"]

    b, benv, bm, bs, b_sales = _run(base, opponent, idx, seed, episode_steps, checkpoint_step)
    i, ienv, im, ins, i_sales = _run(base, opponent, idx, seed, episode_steps, checkpoint_step, deviation)
    state_match = b.checkpoint_obs is not None and _hash(b.checkpoint_obs) == _hash(i.checkpoint_obs)
    # Replay actions cover both players, including named built-in opponents.
    # Keep the wrapper-side focal audit too, as a guard against replay layout
    # changes in kaggle-environments.
    baseline_all_actions = [[state.action for state in pair] for pair in benv.steps[:checkpoint_step]]
    intervention_all_actions = [[state.action for state in pair] for pair in ienv.steps[:checkpoint_step]]
    action_match = (_hash(b.pre_actions) == _hash(i.pre_actions) and
                    _hash(baseline_all_actions) == _hash(intervention_all_actions))
    hard_fail, soft_flags = safety_split(bs, ins)
    applied = i.audit["applied"] > 0
    applied_step = i.audit["events"][0]["step"] if i.audit["events"] else None
    # crop_redirect_two verifies the *target* crop rose (item is None -- it
    # redirects whatever the focal was buying).
    verify_item = deviation.get("target") if kind == "crop_redirect_two" else deviation.get("item")
    verify_window = wc.EXEC_VERIFY_WINDOW_STEPS
    if kind in _SUSTAINED_VERIFY_KINDS and isinstance(applied_step, int):
        # A full divest is verified across the whole label window, not just the
        # 4-day exec-verify window (repair.md Phase B).
        verify_window = max(wc.EXEC_VERIFY_WINDOW_STEPS,
                            _divest_window_end(checkpoint_step, kind, verify_item) - applied_step)
    executed, evidence = _execution_verified(kind, verify_item, benv, ienv,
                                             idx, applied_step, verify_window)
    evidence["applied_step"] = applied_step
    i.audit["execution_verified"] = executed
    i.audit["execution_evidence"] = evidence

    b_sales_focal = _player_sales(b_sales, idx)
    i_sales_focal = _player_sales(i_sales, idx)
    label, label_components = compute_label(output_id, checkpoint_step, benv, ienv,
                                            idx, b_sales_focal, i_sales_focal)
    season_end_delta = im - bm
    eligible = i.audit["eligible"]
    accepted = bool(state_match and action_match and applied and eligible
                    and executed and not hard_fail)
    return {"schema_version": SCHEMA_VERSION, "schema_hash": SCHEMA_HASH,
            "feature_order": list(FEATURE_ORDER),
            "features": extract_features(b.checkpoint_obs, episode_steps) if b.checkpoint_obs else None,
            "seed_group": str(seed), "seed": seed, "player_order": idx, "opponent": opponent_name,
            "strategy_family": strategy_family, "split": split_for_seed_group(seed),
            "sample_weight": 1.0,
            "checkpoint": spec, "deviation": deviation, "probe_order_days": probe_days,
            "probe_volume_days": volume_days, "checkpoint_min_day": min_day,
            "output_id": output_id, "intervention_audit": i.audit,
            "baseline_money": bm, "intervention_money": im,
            "season_end_delta": season_end_delta, "season_end_label": label_delta(season_end_delta),
            "label": label, "label_components": label_components,
            "baseline_safety": bs, "intervention_safety": ins,
            "hard_safety_failure": hard_fail, "soft_safety_flags": soft_flags,
            "accepted": accepted,
            "rejection_reasons": [name for name, ok in (("checkpoint_state_mismatch", state_match),
                                                          ("pre_checkpoint_action_mismatch", action_match),
                                                          ("intervention_ineligible", eligible),
                                                          ("unapplied_intervention", applied),
                                                          ("execution_not_verified", executed),
                                                          ("hard_safety_regression", not hard_fail)) if not ok]}


def resolve_opponent(name):
    if name == "starter": return "starter"
    module_name = name.rsplit(".agent", 1)[0]
    # v2 agents intentionally support direct script execution and therefore
    # import their sibling as ``engine`` rather than package-relative.
    if module_name.startswith("strategies_v2.") and "engine" not in sys.modules:
        sys.modules["engine"] = importlib.import_module("strategies_v2.engine")
    module = importlib.import_module(module_name)
    return module.agent


def _generate_task(task):
    focal = resolve_opponent(task.get("strategy_family", "strategy_follower"))
    return generate_pair(focal, resolve_opponent(task["opponent_name"]),
                         task["opponent_name"], task["seed"], task["pair_index"],
                         task["episode_steps"], task.get("target_output"),
                         task.get("strategy_family", "strategy_follower"),
                         task.get("checkpoint_days"), task.get("probe_cache"))


def load_probe_cache(path):
    """Read a Phase F probe-cache sidecar: {"entries": {"family|opponent": ...}}."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("entries", payload)


def summarize_rows(rows):
    labels, outputs, rejections = Counter(), Counter(), Counter()
    split_labels, family_labels = Counter(), Counter()
    soft_flags, verify_fail_metrics = Counter(), Counter()
    for row in rows:
        for key in row.get("soft_safety_flags", {}):
            soft_flags[key] += 1
        if row["accepted"]:
            labels[str(row["label"])] += 1
            outputs[f"{row['output_id']}|{row['label']}"] += 1
            split_labels[f"{row['output_id']}|{row.get('split')}|{row['label']}"] += 1
            family_labels[f"{row['output_id']}|{row.get('strategy_family')}|{row['label']}"] += 1
        else:
            rejections.update(row["rejection_reasons"])
            if "execution_not_verified" in row["rejection_reasons"]:
                metric = row.get("intervention_audit", {}).get("execution_evidence", {}).get("metric")
                verify_fail_metrics[f"{row['output_id']}|{metric}"] += 1
    return {"schema_version": SCHEMA_VERSION, "schema_hash": SCHEMA_HASH,
            "pairs": len(rows), "accepted": sum(r["accepted"] for r in rows),
            "rejected": sum(not r["accepted"] for r in rows),
            "class_counts": dict(sorted(labels.items())),
            "output_class_counts": dict(sorted(outputs.items())),
            "output_split_class_counts": dict(sorted(split_labels.items())),
            "output_family_class_counts": dict(sorted(family_labels.items())),
            "rejection_counts": dict(sorted(rejections.items())),
            "soft_flag_counts": dict(sorted(soft_flags.items())),
            "execution_verify_failure_metrics": dict(sorted(verify_fail_metrics.items()))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/counterfactual_smoke.jsonl"))
    p.add_argument("--pairs", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--episode-steps", type=int, default=720)
    p.add_argument("--workers", type=int, default=1,
                   help="worker processes; output remains in deterministic task order")
    p.add_argument("--opponents", nargs="+", default=["starter", "strategies_v2.ring.agent", "strategies_v2.leader_clone.agent"])
    p.add_argument("--target-output", default=None,
                   help="generate only interventions supervising this output_id")
    p.add_argument("--strategy-family", default="strategy_follower")
    p.add_argument("--checkpoint-days", nargs="+", type=int,
                   help="replay-guided preferred checkpoint days; infeasible days fall back safely")
    p.add_argument("--probe-cache", type=Path,
                   help="Phase F sidecar of precomputed per-(family, opponent) emission / "
                        "opponent-volume days; skips the per-pair probe replay when a match exists")
    p.add_argument("--append", action="store_true",
                   help="append deterministically ordered new rows to an existing output")
    args = p.parse_args(argv)
    if args.pairs > 2000:
        p.error("--pairs may not exceed the phase-10 per-output attempt cap of 2000")
    probe_cache = load_probe_cache(args.probe_cache) if args.probe_cache else None
    tasks = build_tasks(args.pairs, args.seed_base, args.opponents, args.episode_steps)
    for task in tasks:
        task["target_output"] = args.target_output
        task["strategy_family"] = args.strategy_family
        task["checkpoint_days"] = args.checkpoint_days
        task["probe_cache"] = probe_cache

    # Validate against existing corpus state before running any simulation, so a
    # run that is doomed to fail (reused seed group, attempt cap already spent)
    # fails immediately instead of burning the full batch's runtime first.
    existing_rows = []
    if args.append and args.output.exists():
        existing_rows = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()
                         if line.strip()]
        existing_seeds = {str(row.get("seed")) for row in existing_rows}
        new_seed_groups = {str(task["seed"]) for task in tasks}
        reused = sorted(existing_seeds.intersection(new_seed_groups))
        if reused:
            p.error(f"seed groups already exist in output (first: {reused[0]})")
        if args.target_output:
            prior_attempts = sum(row.get("output_id") == args.target_output for row in existing_rows)
            if prior_attempts + args.pairs > 2000:
                p.error(f"{args.target_output} would exceed the 2000-attempt cap")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.append:
        args.output.write_text("", encoding="utf-8")

    # Each pair is written and flushed as soon as it finishes, rather than held in
    # memory until the whole batch completes. A run that is killed partway
    # (timeout, crash) keeps every pair already paid for on disk, so a retry with
    # --append resumes instead of redoing the same work and hitting the same wall.
    workers = min(max(1, args.workers), max(1, args.pairs), os.cpu_count() or 1)
    new_rows = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        # executor.map preserves input order even when workers finish out of order.
        row_iter = pool.map(_generate_task, tasks) if pool else (_generate_task(task) for task in tasks)
        with args.output.open("a", encoding="utf-8") as fh:
            for row in row_iter:
                new_rows.append(row)
                fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                fh.flush()
    finally:
        if pool:
            pool.shutdown(wait=True)

    all_rows = existing_rows + new_rows
    summary = summarize_rows(all_rows)
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    accepted = summary["accepted"]
    print(f"wrote {len(new_rows)} new pairs; corpus has {len(all_rows)} pairs "
          f"({accepted} accepted, {len(all_rows)-accepted} rejected) at {args.output}")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
