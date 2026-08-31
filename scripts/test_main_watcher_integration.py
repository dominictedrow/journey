"""The watcher-model path must actually reach the packaged submission.

Diagnosis A1 of the SVM repair plan: `main.py` is a build artifact, and a
rules-mode build emits `_watcher_features = None` / `_watcher_model = None`.
When a model IS supplied, `build_submission.py` embeds `shared_features.py` and
`watcher_model.py` inline as namespaces. These tests exercise that real embed
path end to end -- a stub model that returns fixed scores must move
`land_expansion_pressure` / `product_attractiveness`, and a build with no model
must fall back to the rules result without raising.
"""

import ast
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import build_submission
import shared_features as sf


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _linear_svm_source(tmp):
    """A copy of strategy_follower.py with WATCHER_BACKEND flipped to linear_svm."""
    text = (ROOT / "strategy_follower.py").read_text(encoding="utf-8")
    text = re.sub(r'WATCHER_BACKEND = "rules"', 'WATCHER_BACKEND = "linear_svm"', text, count=1)
    path = tmp / "strategy_follower_svm.py"
    path.write_text(text, encoding="utf-8")
    return path


STUB_SCORES = {"land_expansion": 0.75, "labor_expansion": -0.6,
               "product:MELON:increase": 0.9, "product:MELON:avoid": -0.4}


def _stub_watcher(tmp):
    model = tmp / "watcher_model.py"
    report = tmp / "watcher_report.json"
    model.write_text(
        '"""stub"""\nimport math\n'
        f"SCHEMA_VERSION = {sf.SCHEMA_VERSION!r}\n"
        f"SCHEMA_HASH = {sf.SCHEMA_HASH!r}\n"
        f"FEATURE_ORDER = {tuple(sf.FEATURE_ORDER)!r}\n"
        f"MODELS = {{k: {{}} for k in {tuple(STUB_SCORES)!r}}}\n"
        f"_SCORES = {STUB_SCORES!r}\n"
        "def score(output_id, features):\n"
        "    return _SCORES.get(output_id)\n",
        encoding="utf-8")
    report.write_text(json.dumps({
        "metadata": {"schema_version": sf.SCHEMA_VERSION, "schema_hash": sf.SCHEMA_HASH},
        "outputs": {k: {"accepted": True, "status": "accepted",
                        "leave_one_family_out_gate_passed": True} for k in STUB_SCORES},
    }), encoding="utf-8")
    return model, report


def _sample_obs():
    replay_dir = ROOT.parent / "kaggle_replays" / "replays"
    for replay_path in sorted(replay_dir.glob("episode-*-replay.json")):
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        for step in replay["steps"]:
            for state in step:
                if state.get("status") in ("ACTIVE", "DONE"):
                    return state["observation"]
    raise AssertionError("no replay observation available")


class MainWatcherIntegrationTests(unittest.TestCase):
    def test_embedded_stub_model_moves_watcher_signals(self):
        obs = _sample_obs()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model, report = _stub_watcher(tmp)
            built_path = tmp / "main.py"
            build_submission.build(source=_linear_svm_source(tmp), output=built_path,
                                   watcher_model=model, watcher_report=report)
            built = _load(built_path, "_main_with_stub_watcher")

        self.assertIsNotNone(built._watcher_model)
        signals = built.watcher_signals(obs, "linear_svm")
        self.assertEqual(signals["backend"], "linear_svm")
        self.assertAlmostEqual(signals["land_expansion_pressure"], 0.75)
        self.assertAlmostEqual(signals["labor_expansion_pressure"], -0.6)
        # MELON attractiveness = clip(increase - avoid) = clip(0.9 - (-0.4)) = 1.0
        self.assertAlmostEqual(signals["product_attractiveness"]["MELON"], 1.0)
        rule_melon = signals["watcher_trace"]["rule_scores"]["product_attractiveness"]["MELON"]
        self.assertNotAlmostEqual(signals["product_attractiveness"]["MELON"], rule_melon)

    def test_rules_build_has_no_model_and_falls_back_cleanly(self):
        obs = _sample_obs()
        with tempfile.TemporaryDirectory() as tmp:
            built_path = Path(tmp) / "main.py"
            build_submission.build(output=built_path)   # no watcher supplied
            built = _load(built_path, "_main_rules_only")

        self.assertIsNone(built._watcher_model)
        self.assertIsNone(built._watcher_features)
        svm = built.watcher_signals(obs, "linear_svm")   # must not raise
        rules = built.watcher_signals(obs, "rules")
        self.assertEqual(svm["backend"], "rules")
        self.assertEqual(svm["land_expansion_pressure"], rules["land_expansion_pressure"])
        self.assertEqual(svm["product_attractiveness"], rules["product_attractiveness"])

    def test_embedded_submission_imports_no_scientific_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model, report = _stub_watcher(tmp)
            built_path = tmp / "main.py"
            build_submission.build(source=_linear_svm_source(tmp), output=built_path,
                                   watcher_model=model, watcher_report=report)
            tree = ast.parse(built_path.read_text(encoding="utf-8"))
        forbidden = {"numpy", "sklearn", "scipy", "pandas", "torch"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], forbidden)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)


if __name__ == "__main__":
    unittest.main()
