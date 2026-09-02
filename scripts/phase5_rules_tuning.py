"""Phase 5.3 -- tune the deterministic watcher_signals constants against the
market-glutting opponents, paired vs the current (untuned) constants.

watcher_svm_repair_plan.md Phase 5.3: "Take the deterministic watcher_signals
rules path and tune its constants ... against the glutter opponents ... Then
run paired gameplay: legacy rules vs. tuned rules vs. SVM-backed."

This script does the legacy-vs-tuned half (the SVM does not exist yet). It is
a one-factor-at-a-time sweep from the shipped baseline
(OPPONENT_SIGNAL_WEIGHT=4.0, TOWN_EXPOSURE_MULTIPLIER=2.0,
OPPONENT_EXPANSION_THRESHOLD=0.0): screen every variant paired against the
baseline at n=10 per opponent, then confirm the finalists at n=30.

Each variant is generated as its own importable script under scripts/ab_tests/
and run as a subprocess -- compare_paired's ProcessPoolExecutor uses the
'spawn' start method, so the agent callables it pickles must be resolvable by
module name in a fresh interpreter (the tuned agent as __mp_main__.agent, the
baseline as strategy_follower.agent). A parent-process importlib.load of the
variant module does NOT survive that pickle round-trip.

    python phase5_rules_tuning.py                 # full screen + confirm
    python phase5_rules_tuning.py --screen-only   # just the n=10 screen
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "strategy_follower.py"
OUT_DIR = ROOT / "ab_tests"
RESULTS_DIR = OUT_DIR / "results"
ARTIFACTS = ROOT.parent / "artifacts"

_CUT_MARKER = "def _demo():"

BASELINE = {
    "OPPONENT_SIGNAL_WEIGHT": 4.0,
    "TOWN_EXPOSURE_MULTIPLIER": 2.0,
    "OPPONENT_EXPANSION_THRESHOLD": 0.0,
}

# One-factor-at-a-time grid. The baseline value of each knob is omitted (it is
# the control). ~4 values per knob, per Dom's "coarse grid" call.
GRID = {
    "OPPONENT_SIGNAL_WEIGHT": [2.0, 6.0, 8.0],
    "TOWN_EXPOSURE_MULTIPLIER": [1.5, 2.5, 3.0],
    "OPPONENT_EXPANSION_THRESHOLD": [-0.2, 0.15, 0.3],
}

GLUTTERS = ("glut_melon", "glut_livestock", "glut_mixed")
# starter is the non-glut regression guard: a tuning that only helps vs
# glutters but regresses vs a weak opponent is not a promotion candidate.
GUARD = "starter"
GUARD_MAX_REGRESSION = 600.0  # ~2.5% of a typical ~24k baseline final money

_PAIRED_BLOCK = '''

if __name__ == "__main__":
    import sys as _sys, json as _json
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    _sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
    import strategy_follower as _baseline
    from harness import compare_paired
    from strategies_v2 import glut_melon, glut_livestock, glut_mixed

    _OPPS = {
        "glut_melon": glut_melon.agent,
        "glut_livestock": glut_livestock.agent,
        "glut_mixed": glut_mixed.agent,
        "starter": "starter",
    }
    _n = int(_sys.argv[1])
    _seed_base = int(_sys.argv[2])
    _out = {}
    for _i, (_oname, _opp) in enumerate(_OPPS.items()):
        _out[_oname] = compare_paired(agent, _baseline.agent, _opp, _n,
                                      episode_steps=720, seed_base=_seed_base + _i * 5000)
    _rd = _P(__file__).resolve().parent / "results"
    _rd.mkdir(exist_ok=True)
    (_rd / (_P(__file__).stem + "_paired.json")).write_text(_json.dumps(_out, indent=2))
'''


def _generate(name: str, overrides: dict) -> Path:
    text = SOURCE.read_text()
    body = text[: text.index(_CUT_MARKER)]
    for const_name, value in overrides.items():
        tree = ast.parse(body)
        assignment = next(
            (node for node in tree.body
             if isinstance(node, (ast.Assign, ast.AnnAssign))
             and any(isinstance(t, ast.Name) and t.id == const_name
                     for t in (node.targets if isinstance(node, ast.Assign) else [node.target]))),
            None,
        )
        if assignment is None:
            raise ValueError(f"constant {const_name!r} not found in {SOURCE}")
        lines = body.splitlines()
        lines[assignment.lineno - 1:assignment.end_lineno] = [f"{const_name} = {value!r}"]
        body = "\n".join(lines) + "\n"
    body += _PAIRED_BLOCK
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"{name}.py"
    path.write_text(body)
    return path


def _run(name: str, overrides: dict, n: int, seed_base: int) -> dict:
    path = _generate(name, overrides)
    proc = subprocess.run([sys.executable, str(path), str(n), str(seed_base)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return json.loads((RESULTS_DIR / f"{name}_paired.json").read_text())


def _glut_score(paired: dict) -> float:
    return sum(paired[o]["diff_mean"] for o in GLUTTERS) / len(GLUTTERS)


def _guard_ok(paired: dict) -> bool:
    return paired[GUARD]["diff_mean"] >= -GUARD_MAX_REGRESSION


def _variants() -> dict:
    out = {}
    for knob, values in GRID.items():
        for v in values:
            tag = f"{knob.split('_')[1][:3].lower()}_{str(v).replace('.', 'p').replace('-', 'm')}"
            out[f"p5tune_{tag}"] = {**BASELINE, knob: v}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-n", type=int, default=10)
    parser.add_argument("--confirm-n", type=int, default=30)
    parser.add_argument("--confirm-top", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=770000)
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args(argv)

    variants = _variants()
    report = {"baseline": BASELINE, "grid": GRID, "screen_n": args.screen_n,
              "confirm_n": args.confirm_n, "screen": {}, "confirm": {}}

    print(f"=== SCREEN (n={args.screen_n}, {len(variants)} variants) ===", flush=True)
    screened = {}
    for i, (name, ov) in enumerate(variants.items()):
        paired = _run(name, ov, args.screen_n, args.seed_base + i * 100000)
        screened[name] = paired
        gs, ok = _glut_score(paired), _guard_ok(paired)
        report["screen"][name] = {
            "overrides": {k: v for k, v in ov.items() if v != BASELINE[k]},
            "glut_score": gs, "guard_ok": ok,
            "diffs": {o: {"mean": paired[o]["diff_mean"], "ci95": paired[o]["diff_ci95"]}
                      for o in list(GLUTTERS) + [GUARD]},
        }
        print(f"  {name}: glut_score={gs:+.0f} guard={'ok' if ok else 'REGRESS'} "
              + " ".join(f"{o}={paired[o]['diff_mean']:+.0f}" for o in list(GLUTTERS) + [GUARD]),
              flush=True)

    ranked = sorted((n for n in screened if _guard_ok(screened[n])),
                    key=lambda n: _glut_score(screened[n]), reverse=True)
    report["screen_ranking"] = [{"name": n, "glut_score": _glut_score(screened[n])} for n in ranked]
    print("\nscreen ranking (guard-passing):")
    for n in ranked:
        print(f"  {n}: {_glut_score(screened[n]):+.0f}")

    if not args.screen_only:
        finalists = ranked[: args.confirm_top]
        print(f"\n=== CONFIRM (n={args.confirm_n}): {finalists} ===", flush=True)
        for i, name in enumerate(finalists):
            paired = _run(f"{name}_c", variants[name], args.confirm_n,
                          args.seed_base + 900000 + i * 100000)
            gs, ok = _glut_score(paired), _guard_ok(paired)
            report["confirm"][name] = {
                "glut_score": gs, "guard_ok": ok,
                "diffs": {o: {"mean": paired[o]["diff_mean"], "ci95": paired[o]["diff_ci95"],
                              "wins": paired[o]["wins"], "losses": paired[o]["losses"]}
                          for o in list(GLUTTERS) + [GUARD]},
            }
            print(f"  {name}: glut_score={gs:+.0f} guard={'ok' if ok else 'REGRESS'} "
                  + " ".join(f"{o}={paired[o]['diff_mean']:+.0f}{paired[o]['diff_ci95']}"
                             for o in list(GLUTTERS) + [GUARD]), flush=True)

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "phase5_rules_tuning_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {ARTIFACTS / 'phase5_rules_tuning_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
