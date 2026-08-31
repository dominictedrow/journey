"""Replay-derived evaluation challengers and behavioral-family clustering.

This module is evaluation-only.  It treats replay actions as templates for
opponents, never as causal watcher labels.  Representatives use nearest
training-state action lookup; held-out replay rows measure action agreement
and feature-distance trajectory similarity before a challenger is promoted.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import shared_features


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "replay_states.jsonl"
DEFAULT_REPORT = ROOT / "artifacts" / "replay_challenger_report.json"
TOP_NAMES = ROOT / "kaggle_replays" / "leaderboard" / "top10_current.json"
FINGERPRINT_FEATURES = (
    "own_hands", "own_quadrants", "own_animal_GOOSE", "own_animal_COW",
    "own_animal_SHEEP", "own_crop_WHEAT", "own_crop_CARROT", "own_crop_TOMATO",
    "own_crop_STRAWBERRY", "own_crop_MELON",
)
MATCH_FEATURES = FINGERPRINT_FEATURES + ("own_money", "own_empty", "own_ready_yield")
_CACHE = None


def _load_rows(path=DEFAULT_INPUT):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _top_names():
    if not TOP_NAMES.exists():
        return set()
    data = json.loads(TOP_NAMES.read_text())
    return {entry["team_name"] for entry in data.get("entries", [])}


def _vector(row, names):
    values = dict(zip(row["feature_order"], row["features"]))
    return [float(values.get(name, 0)) for name in names]


def fingerprints(rows):
    """Aggregate phase/layout/hire/sale/action fingerprints by leaderboard name."""
    grouped = defaultdict(list)
    for row in rows:
        if row["hour"] == 0 and row["day"] in (0, 5, 8, 10, 15, 20, 29):
            grouped[row["strategy_family"]].append(row)
    output = {}
    for name, samples in grouped.items():
        vector = []
        for day in (0, 5, 8, 10, 15, 20, 29):
            day_rows = [row for row in samples if row["day"] == day]
            for idx in range(len(FINGERPRINT_FEATURES)):
                vector.append(mean(_vector(row, FINGERPRINT_FEATURES)[idx] for row in day_rows) if day_rows else 0.0)
        sales = defaultdict(list)
        actions = Counter()
        for row in samples:
            action = row.get("observed_action") or {}
            for unit in [action.get("farmer", ["PASS"])] + list(action.get("hands", [])):
                actions[unit[0] if unit else "PASS"] += 1
            for order in action.get("market", []):
                if len(order) >= 3 and order[0] == "SELL":
                    sales[order[1]].append(order[2])
        total = sum(actions.values()) or 1
        vector.extend([actions[move] / total for move in ("NORTH", "SOUTH", "EAST", "WEST", "PASS")])
        vector.extend(mean(sales[item]) if sales[item] else 0.0 for item in ("WHEAT", "MELON", "STRAWBERRY", "WOOL"))
        output[name] = vector
    return output


def cluster_fingerprints(values, threshold=0.42):
    """Deterministic normalized-distance connected components."""
    names = sorted(values)
    if not names:
        return []
    columns = list(zip(*(values[name] for name in names)))
    scales = [max(column) - min(column) or 1.0 for column in columns]
    parent = list(range(len(names)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for i in range(len(names)):
        for j in range(i):
            distance = math.sqrt(mean(((a - b) / scale) ** 2 for a, b, scale in
                                      zip(values[names[i]], values[names[j]], scales)))
            if distance <= threshold:
                union(i, j)
    groups = defaultdict(list)
    for i, name in enumerate(names):
        groups[find(i)].append(name)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: (-len(group), group[0]))


def _sanitize(action, hand_count):
    action = action if isinstance(action, dict) else {}
    hands = list(action.get("hands", []))[:hand_count]
    hands.extend([["PASS"]] * (hand_count - len(hands)))
    return {"farmer": list(action.get("farmer", ["PASS"])), "hands": hands,
            "market": list(action.get("market", []))[:10]}


def _index():
    global _CACHE
    if _CACHE is None:
        rows = _load_rows()
        top = _top_names()
        rows = [row for row in rows if row["strategy_family"] in top]
        groups = cluster_fingerprints(fingerprints(rows))
        representatives = [max(group, key=lambda name: sum(r["strategy_family"] == name for r in rows)) for group in groups]
        train = defaultdict(list)
        for row in rows:
            if row["split"] == "train" and row["strategy_family"] in representatives:
                train[(row["strategy_family"], row["day"], row["hour"])].append(row)
        _CACHE = representatives, train
    return _CACHE


def _agent(obs, representative_index):
    representatives, train = _index()
    if representative_index >= len(representatives):
        return {"farmer": ["PASS"], "hands": [["PASS"]] * len(obs["farms"][obs["player"]].get("hands", [])), "market": []}
    name = representatives[representative_index]
    candidates = train.get((name, obs["day"], obs["hour"]), [])
    if not candidates:
        return _sanitize({}, len(obs["farms"][obs["player"]].get("hands", [])))
    current = shared_features.extract_features(obs)
    order = shared_features.FEATURE_ORDER
    indices = [order.index(feature) for feature in MATCH_FEATURES]
    chosen = min(candidates, key=lambda row: sum((current[i] - row["features"][i]) ** 2 for i in indices))
    return _sanitize(chosen.get("observed_action"), len(obs["farms"][obs["player"]].get("hands", [])))


def family_0_agent(obs):
    return _agent(obs, 0)


def family_1_agent(obs):
    return _agent(obs, 1)


def family_2_agent(obs):
    return _agent(obs, 2)


def build_report(rows):
    top = _top_names()
    rows = [row for row in rows if not top or row["strategy_family"] in top]
    values = fingerprints(rows)
    groups = cluster_fingerprints(values)
    counts = Counter(row["strategy_family"] for row in rows)
    validations = {}
    for group in groups:
        representative = max(group, key=counts.get)
        train_rows = [row for row in rows if row["strategy_family"] == representative and row["split"] == "train"]
        held_out = [row for row in rows if row["strategy_family"] == representative and row["split"] in ("val", "test")]
        buckets = defaultdict(list)
        for row in train_rows:
            buckets[(row["day"], row["hour"])].append(row)
        exact = farmer = market = 0
        distances = []
        evaluated = 0
        feature_indices = [rows[0]["feature_order"].index(feature) for feature in MATCH_FEATURES] if rows else []
        for row in held_out:
            candidates = buckets.get((row["day"], row["hour"]), [])
            if not candidates:
                continue
            chosen = min(candidates, key=lambda candidate: sum(
                (row["features"][i] - candidate["features"][i]) ** 2 for i in feature_indices))
            actual = row.get("observed_action") or {}
            predicted = chosen.get("observed_action") or {}
            exact += predicted == actual
            farmer += predicted.get("farmer") == actual.get("farmer")
            market += predicted.get("market") == actual.get("market")
            distances.append(math.sqrt(mean(
                (row["features"][i] - chosen["features"][i]) ** 2 for i in feature_indices)))
            evaluated += 1
        validations[representative] = {
            "held_out_rows": evaluated,
            "exact_action_rate": exact / evaluated if evaluated else 0.0,
            "farmer_action_rate": farmer / evaluated if evaluated else 0.0,
            "market_action_rate": market / evaluated if evaluated else 0.0,
            "state_feature_rmse_mean": mean(distances) if distances else None,
            "trajectory_metric": "nearest same-day/hour state RMSE; gameplay trajectory pending runtime",
        }
    return {
        "source_kind": "observational_replay_challenger",
        "causal_labels": False,
        "names": sorted(values),
        "clusters": [{"members": group, "representative": max(group, key=counts.get),
                      "player_episodes": len({(r["episode_id"], r["player_order"]) for r in rows if r["strategy_family"] in group})}
                     for group in groups],
        "held_out_validation": validations,
        "validation_status": "offline_action_and_state_similarity_complete_gameplay_trajectory_pending",
        "promotion_status": "not_added_to_promotion_opponents",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    report = build_report(_load_rows(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {len(report['clusters'])} behavioral families")


if __name__ == "__main__":
    main()
