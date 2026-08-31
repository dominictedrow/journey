"""AB-test variant generator for strategy_follower.py.

Takes a round config -- {variant_name: {CONST_NAME: value, ...}} -- and for
each variant: copies strategy_follower.py with those constants substituted,
writes it to ab_tests/<variant_name>.py, then runs that file as its own
subprocess. Each variant script, run alone, starts real games against the
configured AI opponents and writes its own results.json -- exactly the
"run the script, a game starts, results come back" flow asked for.

Iteration loop (no subagents needed): call gen_and_run() with a round's
variants, read the printed leaderboard / ab_tests/results/*.json, write the
next round's variants based on what won, call gen_and_run() again.

Usage:
    python test_gen.py rounds/round1.json   # round file: {name: {const: val}}
    python test_gen.py                       # self-check (tiny smoke round)
"""

import json
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "strategy_follower.py"
OUT_DIR = ROOT / "ab_tests"
RESULTS_DIR = OUT_DIR / "results"

OPPONENTS = ["random", "starter"]
N_EPISODES = 10
EPISODE_STEPS = 720

_CUT_MARKER = "def _demo():"

_RUN_BLOCK = '''

def _run_ab():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import json as _json
    from harness import run_matches

    results = {{}}
    for opp in {opponents!r}:
        results[opp] = run_matches(agent, opp, n_episodes={n_episodes!r}, episode_steps={episode_steps!r})
        print({name!r}, "vs", opp, results[opp])
    out_path = _Path(__file__).resolve().parent / "results" / ({name!r} + ".json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        _json.dump(results, f, indent=2)


if __name__ == "__main__":
    _run_ab()
'''


def generate(name, overrides, n_episodes=N_EPISODES, episode_steps=EPISODE_STEPS, opponents=OPPONENTS):
    text = SOURCE.read_text()
    body = text[: text.index(_CUT_MARKER)]
    for const_name, value in overrides.items():
        tree = ast.parse(body)
        assignment = next((node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                           and any(isinstance(target, ast.Name) and target.id == const_name
                                   for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))), None)
        if assignment is None:
            raise ValueError(f"constant {const_name!r} not found in {SOURCE}")
        lines = body.splitlines()
        lines[assignment.lineno - 1:assignment.end_lineno] = [f"{const_name} = {value!r}"]
        body = "\n".join(lines) + "\n"
    body += _RUN_BLOCK.format(opponents=opponents, n_episodes=n_episodes, episode_steps=episode_steps, name=name)
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.py"
    path.write_text(body)
    return path


def run_variant(path):
    # Sequential subprocesses -- each variant's own run_matches() already
    # parallelizes its episodes across all cores, so running variants one at
    # a time (not concurrently) avoids oversubscribing the machine.
    proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{path} failed:\n{proc.stderr}")
    result_path = RESULTS_DIR / f"{path.stem}.json"
    with open(result_path) as f:
        return json.load(f)


def _score(results):
    return sum(r["mean"] for r in results.values()) / len(results)


def gen_and_run(variants, **kwargs):
    """variants: {name: {const_name: value}}. Returns {name: results_dict},
    ranked best-first by combined mean-money score, and prints a leaderboard.
    """
    scored = {}
    for name, overrides in variants.items():
        path = generate(name, overrides, **kwargs)
        scored[name] = run_variant(path)

    ranked = sorted(scored.items(), key=lambda kv: _score(kv[1]), reverse=True)
    print("\n=== leaderboard ===")
    for name, results in ranked:
        print(f"{name}: combined={_score(results):.0f}  " + "  ".join(f"{opp}={r['mean']:.0f}" for opp, r in results.items()))
    return dict(ranked)


def _demo():
    variants = {"_smoke_baseline": {}}
    out = gen_and_run(variants, n_episodes=2, episode_steps=60)
    assert "_smoke_baseline" in out
    assert out["_smoke_baseline"]["random"]["n"] == 2
    print("test_gen self-check OK")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            round_config = json.load(f)
        kwargs = {"n_episodes": int(sys.argv[2])} if len(sys.argv) > 2 else {}
        gen_and_run(round_config, **kwargs)
    else:
        _demo()
