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
from shared_features import FEATURE_ORDER, SCHEMA_HASH, SCHEMA_VERSION, extract_features


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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
              "hand_remove_one", "land_advance", "land_defer")
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
        remaining = self.deviation.setdefault("remaining", 2 if "two" in kind else 1)
        if remaining <= 0: return action
        self.audit["attempted"] += 1
        market = action.setdefault("market", [])
        before = copy.deepcopy(action)
        reason = _intervention_eligibility(kind, item, action, obs, remaining)
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
            for order in market:
                if order and order[0] == "BUY_SEED" and order[1] == item:
                    qty = min(remaining, order[2]); order[1] = target
                    if qty < order[2]: order[2] = qty
                    self.deviation["remaining"] -= qty; break
        if action != before:
            self.audit["applied"] += 1
            self.audit["events"].append({"step": obs.get("step"), "before": before, "after": copy.deepcopy(action)})
        return action


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


def _intervention_eligibility(kind, item, action, obs, remaining):
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
    return None


def _execution_verified(kind, item, baseline_obs, intervention_obs):
    """Verify the intended state delta in the observation after the order."""
    if baseline_obs is None or intervention_obs is None:
        return False, {"reason": "missing_post_checkpoint_observation"}
    bf, inf = _farm(baseline_obs), _farm(intervention_obs)
    if kind in ("crop_add_two", "crop_remove_two"):
        before = _private_count(baseline_obs, "seeds", item)
        after = _private_count(intervention_obs, "seeds", item)
        ok = after > before if kind == "crop_add_two" else after < before
        return ok, {"metric": f"seeds:{item}", "baseline": before, "intervention": after}
    if kind in ("animal_add_one", "animal_defer_one"):
        before = _private_count(baseline_obs, "shed", item)
        after = _private_count(intervention_obs, "shed", item)
        ok = after > before if kind == "animal_add_one" else after < before
        return ok, {"metric": f"shed:{item}", "baseline": before, "intervention": after}
    if kind in ("hand_add_one", "hand_remove_one"):
        before, after = len(bf.get("hands", [])), len(inf.get("hands", []))
        ok = after > before if kind == "hand_add_one" else after < before
        return ok, {"metric": "hands", "baseline": before, "intervention": after}
    if kind in ("land_advance", "land_defer"):
        before = len(bf.get("unlocked_quadrants", []))
        after = len(inf.get("unlocked_quadrants", []))
        ok = after > before if kind == "land_advance" else after < before
        return ok, {"metric": "unlocked_quadrants", "baseline": before, "intervention": after}
    return False, {"reason": "unknown_deviation_kind", "kind": kind}


def _metrics(env, idx):
    feed = water = cap = overflow = missed = noops = 0
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
            "invalid_noop_purchase_indicators": noops, "market_order_cap_violations": cap,
            "episode_status_failures": int(final_state.status != "DONE"),
            "bankruptcy_indicators": int(bool(money_curve) and min(money_curve) <= 0)}


def safety_regression(baseline, intervention):
    """True when any collected lower-is-better operational metric worsens."""
    metrics = (
        "feed_risk_tile_turns", "water_risk_tile_turns", "escaped_animals",
        "missed_watering_deaths", "shed_overflow_indicators",
        "ending_saleable_inventory", "invalid_noop_purchase_indicators",
        "market_order_cap_violations", "episode_status_failures",
        "bankruptcy_indicators",
    )
    return any(intervention.get(key, 0) > baseline.get(key, 0) for key in metrics)


def label_delta(delta):
    """Profitability signal only. A safety regression is a separate,
    causal-validity rejection reason (see ``generate_pair``'s ``accepted``),
    not a reason to flip this label -- a money-positive but unsafe pair is
    not a valid example of either class and must not be trained on."""
    if delta < -250: return -1
    if delta > 250: return 1
    return 0


def _run(base, opponent, idx, seed, steps, checkpoint, deviation=None):
    wrapped = RecordingAgent(base, checkpoint, copy.deepcopy(deviation))
    # Kaggle's callable preparation may copy callable instances.  A function
    # closure keeps the recorder identity visible to this process.
    def focal(observation):
        return wrapped(observation)
    agents = [focal, opponent] if idx == 0 else [opponent, focal]
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed}, debug=False)
    env.run(agents)
    money = env.steps[-1][idx].observation["farms"][idx]["money"]
    return wrapped, env, money, _metrics(env, idx)


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
        kind = "crop_add_two" if direction == "increase" else "crop_remove_two"
        return {"kind": kind, "item": product, "target": None, "remaining": 2}
    animal = next(key for key, value in ANIMAL_PRODUCT.items() if value == product)
    kind = "animal_add_one" if direction == "increase" else "animal_defer_one"
    return {"kind": kind, "item": animal, "target": None, "remaining": 1}


def checkpoint_for_deviation(rng, deviation, episode_steps, preferred_days=None):
    """Schedule interventions early enough to affect an actionable future slot."""
    specs = checkpoint_specs(episode_steps)
    kind = deviation["kind"]
    if kind in ("crop_remove_two", "animal_defer_one", "hand_remove_one", "land_defer"):
        return specs[0]
    last_day = min(len(specs) - 1, 12 if kind.startswith(("crop", "animal")) else 18)
    eligible = specs[:last_day + 1]
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
    if kind == "crop_remove_two": return f"product:{item}:avoid"
    if kind == "animal_add_one": return f"product:{ANIMAL_PRODUCT[item]}:increase"
    if kind == "animal_defer_one": return f"product:{ANIMAL_PRODUCT[item]}:avoid"
    if kind.startswith("hand_"): return "labor_expansion"
    if kind.startswith("land_"): return "land_expansion"
    raise ValueError(f"deviation has no modeled output: {deviation!r}")


def build_tasks(pairs, seed_base, opponents, episode_steps):
    """Construct the stable task order used by sequential and parallel runs."""
    return [{"pair_index": n, "seed": seed_base + n,
             "opponent_name": opponents[n % len(opponents)],
             "episode_steps": episode_steps} for n in range(pairs)]


def generate_pair(base, opponent, opponent_name, seed, pair_index, episode_steps=720,
                  target_output=None, strategy_family="strategy_follower", checkpoint_days=None):
    rng = random.Random((seed << 16) ^ pair_index)
    deviation = (deviation_for_output(target_output, rng) if target_output
                 else random_deviation(rng))
    spec = checkpoint_for_deviation(rng, deviation, episode_steps, checkpoint_days)
    idx = pair_index % 2
    b, benv, bm, bs = _run(base, opponent, idx, seed, episode_steps, spec["step"])
    i, ienv, im, ins = _run(base, opponent, idx, seed, episode_steps, spec["step"], deviation)
    state_match = b.checkpoint_obs is not None and _hash(b.checkpoint_obs) == _hash(i.checkpoint_obs)
    # Replay actions cover both players, including named built-in opponents.
    # Keep the wrapper-side focal audit too, as a guard against replay layout
    # changes in kaggle-environments.
    baseline_all_actions = [[state.action for state in pair] for pair in benv.steps[:spec["step"]]]
    intervention_all_actions = [[state.action for state in pair] for pair in ienv.steps[:spec["step"]]]
    action_match = (_hash(b.pre_actions) == _hash(i.pre_actions) and
                    _hash(baseline_all_actions) == _hash(intervention_all_actions))
    safety = safety_regression(bs, ins)
    applied = i.audit["applied"] > 0
    applied_step = i.audit["events"][0]["step"] if i.audit["events"] else None
    baseline_post = intervention_post = None
    if isinstance(applied_step, int) and applied_step + 1 < len(benv.steps):
        baseline_post = benv.steps[applied_step + 1][idx].observation
        intervention_post = ienv.steps[applied_step + 1][idx].observation
    executed, evidence = _execution_verified(deviation["kind"], deviation.get("item"),
                                              baseline_post, intervention_post)
    evidence["applied_step"] = applied_step
    i.audit["execution_verified"] = executed
    i.audit["execution_evidence"] = evidence
    delta = im - bm
    return {"schema_version": SCHEMA_VERSION, "schema_hash": SCHEMA_HASH,
            "feature_order": list(FEATURE_ORDER), "features": extract_features(b.checkpoint_obs, episode_steps) if b.checkpoint_obs else None,
            "seed_group": str(seed), "seed": seed, "player_order": idx, "opponent": opponent_name,
            "strategy_family": strategy_family, "split": split_for_seed_group(seed),
            "sample_weight": 1.0,
            "checkpoint": spec, "deviation": deviation,
            "output_id": modeled_output_id(deviation), "intervention_audit": i.audit,
            "baseline_money": bm, "intervention_money": im, "delta": delta,
            "baseline_safety": bs, "intervention_safety": ins, "safety_failure": safety,
            "label": label_delta(delta),
            "accepted": bool(state_match and action_match and applied and i.audit["eligible"] and
                             executed and not safety),
            "rejection_reasons": [name for name, ok in (("checkpoint_state_mismatch", state_match),
                                                          ("pre_checkpoint_action_mismatch", action_match),
                                                          ("intervention_ineligible", i.audit["eligible"]),
                                                          ("unapplied_intervention", applied),
                                                          ("execution_not_verified", executed),
                                                          ("safety_regression", not safety)) if not ok]}


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
                         task.get("strategy_family", "strategy_follower"), task.get("checkpoint_days"))


def summarize_rows(rows):
    labels, outputs, rejections = Counter(), Counter(), Counter()
    split_labels, family_labels = Counter(), Counter()
    for row in rows:
        if row["accepted"]:
            labels[str(row["label"])] += 1
            outputs[f"{row['output_id']}|{row['label']}"] += 1
            split_labels[f"{row['output_id']}|{row.get('split')}|{row['label']}"] += 1
            family_labels[f"{row['output_id']}|{row.get('strategy_family')}|{row['label']}"] += 1
        else:
            rejections.update(row["rejection_reasons"])
    return {"schema_version": SCHEMA_VERSION, "schema_hash": SCHEMA_HASH,
            "pairs": len(rows), "accepted": sum(r["accepted"] for r in rows),
            "rejected": sum(not r["accepted"] for r in rows),
            "class_counts": dict(sorted(labels.items())),
            "output_class_counts": dict(sorted(outputs.items())),
            "output_split_class_counts": dict(sorted(split_labels.items())),
            "output_family_class_counts": dict(sorted(family_labels.items())),
            "rejection_counts": dict(sorted(rejections.items()))}


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
    p.add_argument("--append", action="store_true",
                   help="append deterministically ordered new rows to an existing output")
    args = p.parse_args(argv)
    if args.pairs > 2000:
        p.error("--pairs may not exceed the phase-10 per-output attempt cap of 2000")
    tasks = build_tasks(args.pairs, args.seed_base, args.opponents, args.episode_steps)
    for task in tasks:
        task["target_output"] = args.target_output
        task["strategy_family"] = args.strategy_family
        task["checkpoint_days"] = args.checkpoint_days

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
