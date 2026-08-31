import unittest

import counterfactual_dataset as cd


class CounterfactualDatasetTests(unittest.TestCase):
    def test_checkpoints_are_deduplicated_with_both_reasons(self):
        rows = cd.checkpoint_specs(100)
        self.assertEqual(len({x["step"] for x in rows}), len(rows))
        self.assertEqual(rows[3]["reasons"], ["day_start", "shop_unlock"])

    def test_label_boundaries_are_neutral(self):
        self.assertEqual(cd.label_delta(250), 0)
        self.assertEqual(cd.label_delta(-250), 0)
        self.assertEqual(cd.label_delta(251), 1)
        self.assertEqual(cd.label_delta(-251), -1)

    def test_label_is_profitability_only_safety_is_a_separate_signal(self):
        # A money-positive delta stays positive regardless of safety -- a
        # safety regression must reject the pair (see generate_pair), not
        # flip its label.
        self.assertEqual(cd.label_delta(999), 1)

    def test_schema_is_stable_and_unique(self):
        self.assertEqual(len(cd.FEATURE_ORDER), len(set(cd.FEATURE_ORDER)))
        self.assertEqual(len(cd.SCHEMA_HASH), 64)

    def test_remove_hire_is_bounded_and_audited(self):
        base = lambda obs: {"farmer": ["PASS"], "hands": [], "market": [["HIRE"], ["HIRE"]]}
        wrapper = cd.RecordingAgent(base, 0, {"kind": "hand_remove_one", "item": None, "target": None, "remaining": 1})
        action = wrapper({"step": 0, "day": 0, "hour": 0})
        self.assertEqual(action["market"], [["HIRE"]])
        self.assertEqual(wrapper.audit["applied"], 1)
        action = wrapper({"step": 1, "day": 0, "hour": 1})
        self.assertEqual(action["market"], [["HIRE"], ["HIRE"]])

    def test_modeled_output_mapping(self):
        self.assertEqual(cd.modeled_output_id({"kind": "crop_add_two", "item": "MELON"}),
                         "product:MELON:increase")
        self.assertEqual(cd.modeled_output_id({"kind": "crop_redirect_two", "item": "MELON", "target": "WHEAT"}),
                         "product:WHEAT:increase")
        self.assertEqual(cd.modeled_output_id({"kind": "animal_defer_one", "item": "COW"}),
                         "product:MILK:avoid")
        self.assertEqual(cd.modeled_output_id({"kind": "land_advance", "item": None}),
                         "land_expansion")
        self.assertEqual(cd.modeled_output_id({"kind": "hand_add_one", "item": None}),
                         "labor_expansion")

    def test_retain_exit_interventions_are_gone(self):
        # premium_retain/premium_exit were removed: nothing in watcher_signals
        # ever consumed product:*:{retain,exit}, so generating that data was
        # wasted effort (repair plan Phase 3.1).
        self.assertNotIn("premium_retain", cd.DEVIATIONS)
        self.assertNotIn("premium_exit", cd.DEVIATIONS)
        with self.assertRaises(ValueError):
            cd.modeled_output_id({"kind": "premium_retain", "item": "MELON"})

    def test_task_construction_is_deterministic_and_alternating(self):
        opponents = ["starter", "strategies_v2.ring.agent"]
        first = cd.build_tasks(5, 20, opponents, 100)
        self.assertEqual(first, cd.build_tasks(5, 20, opponents, 100))
        self.assertEqual([x["seed"] for x in first], [20, 21, 22, 23, 24])
        self.assertEqual([x["pair_index"] % 2 for x in first], [0, 1, 0, 1, 0])
        self.assertEqual([x["opponent_name"] for x in first],
                         ["starter", "strategies_v2.ring.agent", "starter", "strategies_v2.ring.agent", "starter"])

    def test_targeted_deviation_maps_back_to_requested_output(self):
        import random
        for output_id in ("land_expansion", "labor_expansion", "product:MELON:increase",
                          "product:WOOL:avoid"):
            deviation = cd.deviation_for_output(output_id, random.Random(0))
            self.assertEqual(cd.modeled_output_id(deviation), output_id)

    def test_retain_exit_output_ids_are_rejected(self):
        import random
        with self.assertRaises(ValueError):
            cd.deviation_for_output("product:MELON:retain", random.Random(0))
        with self.assertRaises(ValueError):
            cd.deviation_for_output("product:WOOL:exit", random.Random(0))

    def test_seed_split_is_deterministic_and_complete(self):
        splits = [cd.split_for_seed_group(seed) for seed in range(200)]
        self.assertEqual(splits, [cd.split_for_seed_group(seed) for seed in range(200)])
        self.assertEqual(set(splits), {"train", "val", "test", "confirm"})

    def test_safety_regression_checks_all_operational_metrics(self):
        baseline = {key: 0 for key in (
            "feed_risk_tile_turns", "water_risk_tile_turns", "escaped_animals",
            "missed_watering_deaths", "shed_overflow_indicators",
            "ending_saleable_inventory", "invalid_noop_purchase_indicators",
            "market_order_cap_violations", "episode_status_failures",
            "bankruptcy_indicators",
        )}
        for key in baseline:
            intervention = dict(baseline)
            intervention[key] = 1
            self.assertTrue(cd.safety_regression(baseline, intervention), key)
        self.assertFalse(cd.safety_regression(baseline, dict(baseline)))

    def test_replay_guided_checkpoint_days_are_respected(self):
        import random
        deviation = {"kind": "crop_add_two", "item": "MELON"}
        checkpoint = cd.checkpoint_for_deviation(random.Random(1), deviation, 720, [6, 9, 11])
        self.assertIn(checkpoint["day"], (6, 9, 11))


if __name__ == "__main__":
    unittest.main()
