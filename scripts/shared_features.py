"""Shared feature schema and extraction for watcher training datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter


# v5 (repair.md Phase E): count features are divided by their natural caps so
# the raw matrix is already O(1)-conditioned before StandardScaler, which
# matters at n ~ 150. The feature *list* is unchanged from v4 -- only the
# values are rescaled -- so the version string is bumped explicitly to move
# SCHEMA_HASH. This bump also stamps the Phase B/C deviation-contract change.
# v4 (repair-plan Phase 3.3 feature-matrix audit): every counterfactual
# checkpoint is taken at hour 0, so `turn` (== 24*day), `hour` (== 0) and `day`
# were mutually collinear and collinear with `remaining_fraction`. Only
# `remaining_fraction` plus the non-linear `phase_*` one-hot survive; `turn`,
# `hour` and `day` are dropped. v3 already carried the Phase 2 row-contract
# change (bounded-horizon blend label, hard/soft safety split).
SCHEMA_VERSION = "counterfactual-v5"

# Natural caps for the count features (repair.md Phase E). A reachable farm is
# at most 4 quadrants x 16 tiles; hands top out well under 12; the shed holds
# 100; a sane seed backlog is one board's worth.
TILE_CAP = 64
HANDS_CAP = 12
SHED_CAP = 100
SEED_CAP = 64
CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = CROPS + ("EGG", "MILK", "WOOL")
STRUCTURES = ("COOP", "PASTURE")
SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
         "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
BASE_PRICE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
              "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200}
THROUGHPUT = {"WHEAT": 400, "CARROT": 450, "TOMATO": 200, "STRAWBERRY": 100,
              "MELON": 300, "EGG": 332, "MILK": 122, "WOOL": 105}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _feature_names():
    names = ["remaining_fraction", "phase_opening", "phase_midgame", "phase_endgame"]
    for side in ("own", "opponent"):
        names += [f"{side}_money", f"{side}_hands", f"{side}_quadrants",
                  f"{side}_weeds", f"{side}_empty", f"{side}_ready_yield",
                  f"{side}_feed_risk", f"{side}_water_risk"]
        names += [f"{side}_structure_{x}" for x in STRUCTURES]
        names += [f"{side}_animal_{x}" for x in ANIMALS]
        for crop in CROPS:
            names += [f"{side}_crop_{crop}", f"{side}_crop_{crop}_age_0_2",
                      f"{side}_crop_{crop}_age_3_7", f"{side}_crop_{crop}_age_8_plus"]
    names += [f"own_shed_{x}" for x in PRODUCTS + ANIMALS + ("FERTILIZER",)]
    names += [f"own_carried_{x}" for x in PRODUCTS + ANIMALS + ("FERTILIZER",)]
    names += [f"own_seed_{x}" for x in CROPS]
    names += [f"shop_{x}" for x in SHOPS]
    for product in PRODUCTS:
        names += [f"price_ratio_{product}", f"market_displacement_{product}"]
    names += ["capacity_backlog", "estimated_travel", "asset_service_ratio"]
    return tuple(names)


FEATURE_ORDER = _feature_names()
SCHEMA_HASH = _hash({"version": SCHEMA_VERSION, "features": FEATURE_ORDER})


def _normalizers():
    """Per-feature divisor; features not listed pass through unscaled
    (already-normalized ``price_ratio_*`` / ``market_displacement_*`` / ``phase_*``
    / ``remaining_fraction`` / ``shop_*``, and money, which StandardScaler
    handles alone)."""
    norm = {}
    for side in ("own", "opponent"):
        for k in ("weeds", "empty", "ready_yield", "feed_risk", "water_risk"):
            norm[f"{side}_{k}"] = TILE_CAP
        norm[f"{side}_hands"] = HANDS_CAP
        norm[f"{side}_quadrants"] = 4
        for x in STRUCTURES:
            norm[f"{side}_structure_{x}"] = TILE_CAP
        for x in ANIMALS:
            norm[f"{side}_animal_{x}"] = TILE_CAP
        for crop in CROPS:
            norm[f"{side}_crop_{crop}"] = TILE_CAP
            for bucket in ("0_2", "3_7", "8_plus"):
                norm[f"{side}_crop_{crop}_age_{bucket}"] = TILE_CAP
    for item in PRODUCTS + ANIMALS + ("FERTILIZER",):
        norm[f"own_shed_{item}"] = SHED_CAP
        norm[f"own_carried_{item}"] = SHED_CAP
    for crop in CROPS:
        norm[f"own_seed_{crop}"] = SEED_CAP
    # capacity_backlog is a sum of raw tile-turn / yield counts; estimated_travel
    # and asset_service_ratio are already bounded ratios and pass through.
    norm["capacity_backlog"] = TILE_CAP
    return norm


FEATURE_NORMALIZERS = _normalizers()


def extract_features(obs, episode_steps=720):
    """Return a fixed-order raw aggregate vector; opponent private data is absent."""
    player = obs["player"]
    day, step = obs["day"], obs.get("step", obs["day"] * 24 + obs["hour"])
    values = {"remaining_fraction": max(0.0, (episode_steps - 1 - step) / max(1, episode_steps - 1)),
              "phase_opening": int(day < 10), "phase_midgame": int(10 <= day < 20),
              "phase_endgame": int(day >= 20)}
    for label, idx in (("own", player), ("opponent", 1 - player)):
        farm = obs["farms"][idx]
        counts, ages = Counter(), Counter()
        ready = feed = water = weeds = empty = 0
        for row in farm["tiles"]:
            for tile in row:
                if tile is None:
                    empty += 1
                elif isinstance(tile, dict):
                    kind = tile.get("kind")
                    counts[kind] += 1
                    if kind == "WEED": weeds += 1
                    if kind == "PLANT":
                        crop = tile.get("crop")
                        counts["crop_" + str(crop)] += 1
                        age = max(0, day - tile.get("planted_day", day))
                        ages[(crop, "0_2" if age <= 2 else "3_7" if age <= 7 else "8_plus")] += 1
                        ready += tile.get("yield_units", 0)
                        water += int(tile.get("consecutive_unwatered", 0) > 0)
                    if kind in STRUCTURES:
                        animal = tile.get("animal")
                        if animal: counts["animal_" + animal] += 1
                        ready += tile.get("yield_units", 0)
                        feed += int(bool(animal) and tile.get("consecutive_unfed", 0) > 0)
        values.update({f"{label}_money": farm["money"], f"{label}_hands": len(farm.get("hands", [])),
                       f"{label}_quadrants": len(farm.get("unlocked_quadrants", [])),
                       f"{label}_weeds": weeds, f"{label}_empty": empty,
                       f"{label}_ready_yield": ready, f"{label}_feed_risk": feed,
                       f"{label}_water_risk": water})
        for x in STRUCTURES: values[f"{label}_structure_{x}"] = counts[x]
        for x in ANIMALS: values[f"{label}_animal_{x}"] = counts["animal_" + x]
        for crop in CROPS:
            values[f"{label}_crop_{crop}"] = counts["crop_" + crop]
            for bucket in ("0_2", "3_7", "8_plus"):
                values[f"{label}_crop_{crop}_age_{bucket}"] = ages[(crop, bucket)]
    private = obs["private"]
    carried = Counter()
    for inv in private.get("inventories", []): carried.update(inv)
    for item in PRODUCTS + ANIMALS + ("FERTILIZER",):
        values[f"own_shed_{item}"] = private.get("shed", {}).get(item, 0)
        values[f"own_carried_{item}"] = carried[item]
    for crop in CROPS: values[f"own_seed_{crop}"] = private.get("seeds", {}).get(crop, 0)
    shops = Counter(obs.get("town", {}).get("unlocked_shops", []))
    for shop in SHOPS: values[f"shop_{shop}"] = shops[shop]
    for product in PRODUCTS:
        values[f"price_ratio_{product}"] = obs["market"]["prices"].get(product, BASE_PRICE[product]) / BASE_PRICE[product]
        values[f"market_displacement_{product}"] = (obs["market"]["inventory"].get(product, 10000) - 10000) / THROUGHPUT[product]
    backlog = values["own_feed_risk"] + values["own_water_risk"] + values["own_ready_yield"]
    units = 1 + values["own_hands"]
    assets = sum(values[f"own_animal_{x}"] for x in ANIMALS) + sum(values[f"own_crop_{x}"] for x in CROPS)
    values["capacity_backlog"] = backlog
    values["estimated_travel"] = assets / units
    values["asset_service_ratio"] = assets / (24 * units)
    return [values[name] / FEATURE_NORMALIZERS.get(name, 1) for name in FEATURE_ORDER]
