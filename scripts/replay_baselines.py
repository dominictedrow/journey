"""Build phase-3 diagnostic ranges from observational leaderboard states."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "replay_states.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "replay_operational_baselines.json"


def _quantiles(values):
    values = sorted(values)
    if not values:
        return None
    def pick(frac):
        return values[round((len(values) - 1) * frac)]
    return {"n": len(values), "min": values[0], "p25": pick(.25),
            "median": median(values), "p75": pick(.75), "max": values[-1],
            "mean": mean(values)}


def build(rows):
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[(row["episode_id"], row["player_order"])].append(row)
    hand_counts = defaultdict(list)
    crop_mix = defaultdict(lambda: defaultdict(list))
    land_days = []
    action_counts = Counter()
    sale_batches = defaultdict(list)
    transition_days = defaultdict(list)

    for episode_rows in by_episode.values():
        episode_rows.sort(key=lambda row: row["step"])
        previous_quadrants = 1
        previous_crops = Counter()
        seen_transition = set()
        for row in episode_rows:
            features = dict(zip(row["feature_order"], row["features"]))
            day = int(row["day"])
            action = row.get("observed_action") or {}
            # Replay observations intentionally omit private hand positions;
            # the recorded hand-action vector is the reliable public count.
            hand_counts[day].append(len(action.get("hands", [])))
            quadrants = int(features.get("own_quadrants", 1))
            if quadrants > previous_quadrants:
                land_days.extend([day] * (quadrants - previous_quadrants))
            previous_quadrants = quadrants
            current_crops = Counter()
            for crop in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"):
                count = int(features.get(f"own_crop_{crop}", 0))
                crop_mix[day][crop].append(count)
                current_crops[crop] = count
                if crop in ("STRAWBERRY", "WHEAT") and count > previous_crops[crop] and crop not in seen_transition:
                    transition_days[crop].append(day)
                    seen_transition.add(crop)
            if previous_crops["MELON"] > 0 and current_crops["MELON"] < previous_crops["MELON"] and "MELON_EXIT" not in seen_transition:
                transition_days["MELON_EXIT"].append(day)
                seen_transition.add("MELON_EXIT")
            previous_crops = current_crops
            units = [action.get("farmer", ["PASS"])] + list(action.get("hands", []))
            for unit_action in units:
                name = unit_action[0] if isinstance(unit_action, list) and unit_action else "PASS"
                category = "move" if name in {"NORTH", "SOUTH", "EAST", "WEST"} else "pass" if name == "PASS" else "work"
                action_counts[category] += 1
            for order in action.get("market", []):
                if isinstance(order, list) and len(order) >= 3 and order[0] == "SELL":
                    sale_batches[order[1]].append(order[2])

    total_actions = sum(action_counts.values()) or 1
    return {
        "source_kind": "observational_replay_diagnostic_only",
        "episodes": len(by_episode),
        "daily_hand_count_ranges": {str(day): _quantiles(values) for day, values in sorted(hand_counts.items())},
        "land_purchase_day_ranges": _quantiles(land_days),
        "daily_crop_mix_ranges": {
            str(day): {crop: _quantiles(values) for crop, values in crops.items()}
            for day, crops in sorted(crop_mix.items())
        },
        "unit_action_shares": {key: value / total_actions for key, value in action_counts.items()},
        "sale_batch_ranges": {product: _quantiles(values) for product, values in sale_batches.items()},
        "transition_day_ranges": {name: _quantiles(values) for name, values in transition_days.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    with args.input.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    report = build(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} from {report['episodes']} player-episodes")


if __name__ == "__main__":
    main()
