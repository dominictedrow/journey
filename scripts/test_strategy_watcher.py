import math
import unittest

import shared_features as sf
import strategy_follower as follower


def observation():
    farm = {"money": 1000, "hands": [], "unlocked_quadrants": [0],
            "tiles": [[None, {"kind": "PLANT", "crop": "WHEAT",
                               "planted_day": 1, "yield_units": 2,
                               "consecutive_unwatered": 0}]]}
    opponent = {"money": 900, "hands": [{}], "unlocked_quadrants": [0, 1],
                "tiles": [[{"kind": "WEED"}, None]]}
    return {"player": 0, "step": 72, "day": 3, "hour": 0,
            "farms": [farm, opponent],
            "private": {"shed": {}, "seeds": {}, "inventories": []},
            "market": {"prices": {}, "inventory": {}},
            "town": {"unlocked_shops": ["BAKERY"]}}


class FakeModel:
    SCHEMA_VERSION = sf.SCHEMA_VERSION
    SCHEMA_HASH = sf.SCHEMA_HASH
    FEATURE_ORDER = sf.FEATURE_ORDER
    MODELS = {"placeholder": {}}
    scores = {}

    @classmethod
    def score(cls, output_id, features):
        return cls.scores.get(output_id)


class RuntimeWatcherTests(unittest.TestCase):
    def setUp(self):
        self.old_features = follower._watcher_features
        self.old_model = follower._watcher_model
        follower._watcher_features = sf
        follower._watcher_model = FakeModel
        FakeModel.SCHEMA_HASH = sf.SCHEMA_HASH
        FakeModel.SCHEMA_VERSION = sf.SCHEMA_VERSION
        FakeModel.FEATURE_ORDER = sf.FEATURE_ORDER
        FakeModel.MODELS = {"placeholder": {}}
        FakeModel.scores = {}

    def tearDown(self):
        follower._watcher_features = self.old_features
        follower._watcher_model = self.old_model

    def test_svm_combines_product_scores_and_keeps_expansion_separate(self):
        FakeModel.scores = {"product:WHEAT:increase": 0.8,
                            "product:WHEAT:avoid": 0.1,
                            "land_expansion": -0.5,
                            "labor_expansion": 0.0}
        signals = follower.watcher_signals(observation(), "linear_svm")
        self.assertEqual(signals["backend"], "linear_svm")
        self.assertTrue(math.isclose(signals["product_attractiveness"]["WHEAT"], 0.7))
        self.assertTrue(math.isclose(signals["land_expansion_pressure"], -0.5))
        self.assertTrue(math.isclose(signals["labor_expansion_pressure"], 0.0))

    def test_missing_output_falls_back_per_signal(self):
        rules = follower.watcher_signals(observation(), "rules")
        FakeModel.scores = {"product:WHEAT:increase": 0.8}
        signals = follower.watcher_signals(observation(), "linear_svm")
        self.assertEqual(signals["backend"], "linear_svm")
        self.assertEqual(signals["product_attractiveness"]["MELON"],
                         rules["product_attractiveness"]["MELON"])

    def test_schema_mismatch_falls_back_to_rules(self):
        FakeModel.SCHEMA_HASH = "incompatible"
        self.assertEqual(follower.watcher_signals(observation(), "linear_svm")["backend"], "rules")

    def test_empty_export_falls_back_to_rules(self):
        FakeModel.MODELS = {}
        self.assertEqual(follower.watcher_signals(observation(), "linear_svm")["backend"], "rules")

    def test_non_finite_output_falls_back_per_signal(self):
        rules = follower.watcher_signals(observation(), "rules")
        FakeModel.scores = {"product:WHEAT:increase": float("nan")}
        signals = follower.watcher_signals(observation(), "linear_svm")
        self.assertEqual(signals["backend"], "rules")
        self.assertEqual(signals["product_attractiveness"]["WHEAT"],
                         rules["product_attractiveness"]["WHEAT"])

    def test_labor_model_changes_target_by_at_most_one(self):
        obs = observation()
        rules = follower.watcher_signals(obs, "rules")
        rule_targets = follower._adaptive_targets(
            obs, obs["farms"][0], obs["farms"][0]["tiles"], {}, rules)
        FakeModel.scores = {"labor_expansion": 1.0}
        model = follower.watcher_signals(obs, "linear_svm")
        model_targets = follower._adaptive_targets(
            obs, obs["farms"][0], obs["farms"][0]["tiles"], {}, model)
        self.assertLessEqual(abs(model_targets["daily_hands"] - rule_targets["daily_hands"]), 1)

    def test_schema_version_mismatch_falls_back_to_rules(self):
        FakeModel.SCHEMA_VERSION = "wrong"
        self.assertEqual(follower.watcher_signals(observation(), "linear_svm")["backend"], "rules")


if __name__ == "__main__":
    unittest.main()
