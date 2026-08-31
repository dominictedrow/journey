"""Run promotion-quality phase-4/5 screens from a round configuration."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import compare_paired, run_matches, run_matches_instrumented
from test_gen import generate
from strategies_v2 import leader_clone, ring


OPPONENTS = {"starter": "starter", "ring": ring.agent, "leader_clone": leader_clone.agent}


def _load(path):
    spec = importlib.util.spec_from_file_location(f"_phase_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent


def run(config_path, control_name, n=20, seed_base=30000):
    variants = json.loads(Path(config_path).read_text())
    if control_name not in variants:
        raise ValueError(f"control {control_name!r} is absent from {config_path}")
    agents = {name: _load(generate(name, overrides, n_episodes=1, episode_steps=1))
              for name, overrides in variants.items()}
    control = agents[control_name]
    report = {"round": Path(config_path).stem, "control": control_name, "n": n,
              "seed_base": seed_base, "smoke": {}, "paired": {}}
    for offset, (name, agent) in enumerate(agents.items()):
        report["smoke"][name] = {
            "short": run_matches(agent, "starter", 4, episode_steps=120,
                                 seeds=list(range(seed_base + offset * 10, seed_base + offset * 10 + 4))),
            "full": run_matches_instrumented(
                agent, "starter", 2, episode_steps=720,
                seeds=list(range(seed_base + 1000 + offset * 10, seed_base + 1002 + offset * 10)),
            ),
        }
        if name == control_name:
            continue
        report["paired"][name] = {
            opponent_name: compare_paired(
                agent, control, opponent, n, episode_steps=720,
                seed_base=seed_base + 2000 + opponent_offset * n,
            )
            for opponent_offset, (opponent_name, opponent) in enumerate(OPPONENTS.items())
        }
    out = ROOT / "ab_rounds" / f"{Path(config_path).stem}_results.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--control", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=30000)
    args = parser.parse_args(argv)
    out, _ = run(args.config, args.control, args.episodes, args.seed_base)
    print(out)


if __name__ == "__main__":
    main()
