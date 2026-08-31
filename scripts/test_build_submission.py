"""Tests for deterministic source-to-submission builds and action parity."""

import ast
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import build_submission


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildSubmissionTests(unittest.TestCase):
    def test_watcher_build_mode_rejects_silent_backend_mismatches(self):
        rules = "WATCHER_BACKEND = 'rules'\n"
        svm = "WATCHER_BACKEND = 'linear_svm'\n"
        build_submission._validate_watcher_build_mode(rules, None, None)
        with self.assertRaisesRegex(ValueError, "requires --watcher-model"):
            build_submission._validate_watcher_build_mode(svm, None, None)
        with self.assertRaisesRegex(ValueError, "require WATCHER_BACKEND"):
            build_submission._validate_watcher_build_mode(rules, "model.py", "report.json")
        build_submission._validate_watcher_build_mode(svm, "model.py", "report.json")

    def test_embedded_watcher_exports_schema_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            features = tmp / "features.py"
            model = tmp / "model.py"
            report = tmp / "report.json"
            features.write_text(
                "SCHEMA_VERSION='v1'\nSCHEMA_HASH='h1'\nFEATURE_ORDER=('x',)\n"
                "def extract_features(obs): return [0.0]\n")
            model.write_text(
                "SCHEMA_VERSION='v1'\nSCHEMA_HASH='h1'\nFEATURE_ORDER=('x',)\n"
                "MODELS={'land_expansion': {}}\ndef score(output_id, values): return 1.0\n")
            report.write_text(json.dumps({
                "metadata": {"schema_version": "v1", "schema_hash": "h1"},
                "outputs": {"land_expansion": {"accepted": True, "status": "accepted",
                                                "leave_one_family_out_gate_passed": True}},
            }))
            namespace = {}
            exec(build_submission._watcher_replacement(model, report, features), namespace)
            self.assertEqual(namespace["_watcher_features"].SCHEMA_VERSION, "v1")
            self.assertEqual(namespace["_watcher_model"].SCHEMA_VERSION, "v1")

    def test_action_validator_rejects_bad_hand_count_and_market_grammar(self):
        obs = {"player": 0, "farms": [{"hands": [[1, 1]]}, {"hands": []}]}
        with self.assertRaises(ValueError):
            build_submission._validate_action(
                {"farmer": ["PASS"], "hands": [], "market": []}, obs)
        with self.assertRaises(ValueError):
            build_submission._validate_action(
                {"farmer": ["PASS"], "hands": [["PASS"]],
                 "market": [["SELL", "WHEAT", 0]]}, obs)

    def test_deterministic_rules_build_and_last_callable(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "one.py", Path(tmp) / "two.py"
            build_submission.build(output=first)
            build_submission.build(output=second)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            tree = ast.parse(first.read_text())
            callables = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            self.assertEqual(callables[-1].name, "agent")

    def test_thousands_of_replay_observations_match(self):
        source = load(ROOT / "strategy_follower.py", "_parity_source")
        with tempfile.TemporaryDirectory() as tmp:
            built_path = Path(tmp) / "main.py"
            build_submission.build(output=built_path)
            built = load(built_path, "_parity_built")
            checked = 0
            replay_dir = ROOT.parent / "kaggle_replays" / "replays"
            for replay_path in sorted(replay_dir.glob("episode-*-replay.json")):
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
                for step in replay["steps"]:
                    for state in step:
                        if state.get("status") not in ("ACTIVE", "DONE"):
                            continue
                        obs = copy.deepcopy(state["observation"])
                        expected = source.agent(copy.deepcopy(obs))
                        actual = built.agent(copy.deepcopy(obs))
                        self.assertEqual(expected, actual, msg=json.dumps({
                            "checked": checked, "step": obs.get("step"), "player": obs.get("player"),
                            "expected": expected, "actual": actual}, sort_keys=True))
                        self.assertEqual(set(actual), {"farmer", "hands", "market"})
                        self.assertTrue(all(isinstance(actual[key], list) for key in actual))
                        self.assertLessEqual(len(actual["market"]), 10)
                        checked += 1
                        if checked >= 5000:
                            break
                    if checked >= 5000:
                        break
                if checked >= 5000:
                    break
            self.assertGreaterEqual(checked, 5000)


if __name__ == "__main__":
    unittest.main()
