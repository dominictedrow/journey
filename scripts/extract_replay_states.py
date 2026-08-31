"""Extract realistic, explicitly non-causal watcher states from Kaggle replays.

These rows are useful for feature-coverage diagnostics, strategy-family
clustering, and choosing intervention checkpoints.  They are deliberately not
accepted by train_watcher.py: a factual replay contains no matched alternate
decision and therefore cannot supply a causal +/-1 label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from shared_features import FEATURE_ORDER, SCHEMA_HASH, SCHEMA_VERSION, extract_features


def stable_split(episode_id):
    bucket = int(hashlib.sha256(str(episode_id).encode()).hexdigest()[:8], 16) % 20
    return "train" if bucket < 14 else "val" if bucket < 17 else "test"


def observed_decisions(action):
    action = action if isinstance(action, dict) else {}
    market = action.get("market", []) if isinstance(action.get("market", []), list) else []
    result = Counter()
    for order in market:
        if not isinstance(order, list) or not order:
            continue
        op = order[0]
        if op == "HIRE":
            result["labor_expansion"] += 1
        elif op == "BUY_LAND":
            result["land_expansion"] += 1
        elif op == "BUY_SEED" and len(order) >= 3:
            result[f"product:{order[1]}:increase"] += int(order[2])
        elif op == "BUY_ANIMAL" and len(order) >= 3:
            product = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}.get(order[1])
            if product:
                result[f"product:{product}:increase"] += int(order[2])
    return dict(sorted(result.items()))


def replay_rows(path, checkpoint_hour=0):
    replay = json.loads(Path(path).read_text(encoding="utf-8"))
    episode_id = replay.get("info", {}).get("EpisodeId", replay.get("id", Path(path).stem))
    agents = replay.get("info", {}).get("Agents", [])
    rewards = replay.get("rewards", [])
    steps = replay.get("steps", [])
    episode_steps = int(replay.get("configuration", {}).get("episodeSteps", len(steps) or 720))
    rows = []
    for step_index, pair in enumerate(steps):
        if not isinstance(pair, list):
            continue
        for player, state in enumerate(pair):
            if not isinstance(state, dict):
                continue
            obs = state.get("observation")
            if not isinstance(obs, dict) or int(obs.get("hour", step_index % 24)) != checkpoint_hour:
                continue
            features = extract_features(obs, episode_steps)
            if len(features) != len(FEATURE_ORDER) or not all(math.isfinite(x) for x in features):
                continue
            agent = agents[player] if player < len(agents) and isinstance(agents[player], dict) else {}
            rows.append({
                "corpus_kind": "observational_replay_state",
                "trainable_for_causal_svm": False,
                "schema_version": SCHEMA_VERSION,
                "schema_hash": SCHEMA_HASH,
                "feature_order": list(FEATURE_ORDER),
                "features": features,
                "episode_id": episode_id,
                "seed_group": f"replay:{episode_id}",
                "split": stable_split(episode_id),
                "player_order": player,
                "strategy_family": agent.get("Name", f"unknown-player-{player}"),
                "step": int(obs.get("step", step_index)),
                "day": int(obs.get("day", step_index // 24)),
                "hour": int(obs.get("hour", step_index % 24)),
                "observed_action": state.get("action"),
                "observed_decisions": observed_decisions(state.get("action")),
                "final_reward": rewards[player] if player < len(rewards) else None,
                "label": None,
                "accepted": False,
                "rejection_reasons": ["observational_only_no_counterfactual"],
                "source_replay": str(path),
            })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=Path("kaggle_replays/replays"))
    parser.add_argument("--output", type=Path, default=Path("data/replay_states.jsonl"))
    parser.add_argument("--checkpoint-hour", type=int, default=0, choices=range(24))
    parser.add_argument("--limit", type=int, default=None,
                        help="maximum unique normal replay files, for deterministic smoke runs")
    args = parser.parse_args(argv)

    paths = sorted(args.replay_dir.glob("episode-*-replay.json"))
    if args.limit is not None:
        paths = paths[:max(0, args.limit)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    families, decisions, splits = Counter(), Counter(), Counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for path in paths:
            for row in replay_rows(path, args.checkpoint_hour):
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                row_count += 1
                families[row["strategy_family"]] += 1
                splits[row["split"]] += 1
                decisions.update(row["observed_decisions"])
    summary = {
        "replays": len(paths), "rows": row_count, "checkpoint_hour": args.checkpoint_hour,
        "schema_version": SCHEMA_VERSION, "schema_hash": SCHEMA_HASH,
        "strategy_family_counts": dict(families.most_common()),
        "split_counts": dict(sorted(splits.items())),
        "observed_decision_counts": dict(sorted(decisions.items())),
        "causal_labels": 0,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {row_count} observational rows from {len(paths)} replays to {args.output}")


if __name__ == "__main__":
    main()
