import random
import unittest
from types import SimpleNamespace
from unittest import mock

import counterfactual_dataset as cd


def _money_env(m0, m1, shed=None, prices=None):
    shed = shed or {}
    prices = prices or {}

    def obs(m):
        return {"player": 0, "farms": [{"money": m, "tiles": []}],
                "private": {"shed": dict(shed), "inventories": []},
                "market": {"prices": dict(prices)}}
    s0, s1 = SimpleNamespace(observation=obs(m0)), SimpleNamespace(observation=obs(m1))
    return SimpleNamespace(steps=[[s0, s0], [s1, s1]])


def _planted_env(planted_per_step):
    steps = []
    for n in planted_per_step:
        obs = {"player": 0,
               "farms": [{"tiles": [[{"kind": "PLANT", "crop": "MELON"} for _ in range(n)]]}],
               "private": {"seeds": {}}}
        s = SimpleNamespace(observation=obs)
        steps.append([s, s])
    return SimpleNamespace(steps=steps)


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

    def test_safety_split_separates_hard_rejects_from_soft_flags(self):
        baseline = {key: 0 for key in cd.HARD_SAFETY_METRICS + cd.SOFT_SAFETY_METRICS}
        for key in cd.HARD_SAFETY_METRICS:
            hard_fail, _ = cd.safety_split(baseline, dict(baseline, **{key: 1}))
            self.assertTrue(hard_fail, key)
        # A profitable-but-messy pair (extra unsold units, a deeper cash dip) is
        # a legitimate example -- soft flags are recorded, never rejecting.
        messy = dict(baseline, ending_saleable_inventory=9, cash_trough_touched=1,
                     feed_risk_tile_turns=3)
        hard_fail, soft = cd.safety_split(baseline, messy)
        self.assertFalse(hard_fail)
        self.assertEqual(set(soft), {"ending_saleable_inventory", "cash_trough_touched",
                                     "feed_risk_tile_turns"})
        self.assertEqual(cd.safety_split(baseline, dict(baseline)), (False, {}))

    def test_replay_guided_checkpoint_days_are_respected(self):
        deviation = {"kind": "crop_add_two", "item": "MELON"}
        checkpoint = cd.checkpoint_for_deviation(random.Random(1), deviation, 720, [6, 9, 11])
        self.assertIn(checkpoint["day"], (6, 9, 11))

    def test_removal_checkpoint_moves_off_day_zero_toward_emission_days(self):
        deviation = {"kind": "crop_remove_two", "item": "MELON"}
        days = {cd.checkpoint_for_deviation(random.Random(s), deviation, 720, [5, 9, 11, 13])["day"]
                for s in range(40)}
        self.assertTrue(days <= {5, 9})          # earlier half only
        self.assertNotIn(0, days)
        # No discovered emission days -> fall back to day 0 (the intervention
        # will then be rejected as no_matching_future_purchase, correctly).
        self.assertEqual(
            cd.checkpoint_for_deviation(random.Random(0), deviation, 720, [])["day"], 0)

    def test_order_emission_days_reports_days_focal_emits_the_cancelled_order(self):
        steps = []
        orders_by_step = {30: [["BUY_SEED", "MELON", 2]],
                          130: [["BUY_SEED", "WHEAT", 1], ["BUY_SEED", "MELON", 1]]}
        for t in range(200):
            state = SimpleNamespace(action={"market": orders_by_step.get(t, [])})
            steps.append([state, state])
        env = SimpleNamespace(steps=steps)
        self.assertEqual(cd._order_emission_days(env, 0, "crop_remove_two", "MELON"), [1, 5])

    def test_crop_redirect_partial_conserves_units(self):
        base = lambda obs: {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 5]]}
        wrapper = cd.RecordingAgent(base, 0, {"kind": "crop_redirect_two", "item": "WHEAT",
                                              "target": "MELON", "remaining": 2})
        action = wrapper({"step": 0, "day": 0, "hour": 0})
        seed_orders = [o for o in action["market"] if o[0] == "BUY_SEED"]
        self.assertEqual(sum(o[2] for o in seed_orders), 5)
        self.assertEqual({o[1]: o[2] for o in seed_orders}, {"WHEAT": 3, "MELON": 2})

    def test_crop_redirect_full_changes_crop_in_place(self):
        base = lambda obs: {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 2]]}
        wrapper = cd.RecordingAgent(base, 0, {"kind": "crop_redirect_two", "item": "WHEAT",
                                              "target": "MELON", "remaining": 2})
        action = wrapper({"step": 0, "day": 0, "hour": 0})
        self.assertEqual(action["market"], [["BUY_SEED", "MELON", 2]])

    def test_execution_verification_accepts_an_immediately_planted_seed(self):
        benv = _planted_env([0, 0, 0, 0, 0])
        ienv = _planted_env([0, 0, 1, 1, 1])  # seed bought and planted at step 2
        ok, evidence = cd._execution_verified("crop_add_two", "MELON", benv, ienv, 0,
                                              applied_step=0, window_steps=4)
        self.assertTrue(ok)
        self.assertEqual(evidence["step"], 2)

    def test_execution_verification_fails_when_state_never_moves(self):
        benv = _planted_env([1, 1, 1, 1])
        ienv = _planted_env([1, 1, 1, 1])
        ok, evidence = cd._execution_verified("crop_add_two", "MELON", benv, ienv, 0,
                                              applied_step=0, window_steps=3)
        self.assertFalse(ok)
        self.assertEqual(evidence["intervention_end"], 1)

    def test_product_label_neutral_when_money_and_x_flow_disagree(self):
        benv = _money_env(0, 100)
        ienv = _money_env(0, 300)  # money_delta = +200
        b_sales = [{"step": 0, "sold": {"MELON": {"units": 2, "revenue": 500.0}}}]
        i_sales = []  # intervention's freed capital went somewhere other than melon
        label, comp = cd.label_product("MELON", 0, 7, benv, ienv, 0, b_sales, i_sales)
        self.assertEqual(label, 0)
        self.assertLess(comp["x_flow_delta"], 0)
        self.assertGreater(comp["money_delta"], 0)

    def test_product_label_signed_when_money_and_x_flow_agree(self):
        benv = _money_env(0, 100)
        ienv = _money_env(0, 400)  # money_delta = +300
        b_sales = [{"step": 0, "sold": {"MELON": {"units": 1, "revenue": 200.0}}}]
        i_sales = [{"step": 0, "sold": {"MELON": {"units": 3, "revenue": 560.0}}}]
        label, _ = cd.label_product("MELON", 0, 7, benv, ienv, 0, b_sales, i_sales)
        self.assertEqual(label, 1)

    def test_capacity_label_is_money_delta_only(self):
        benv, ienv = _money_env(0, 100), _money_env(0, 1200)
        label, comp = cd.label_capacity(0, 11, benv, ienv, 0)
        self.assertEqual(label, 1)
        self.assertEqual(comp["basis"], "capacity_money")

    def test_run_raises_loudly_when_recorder_misses_the_checkpoint(self):
        class Blind(cd.RecordingAgent):
            def __call__(self, obs):  # never records checkpoint_obs
                return self.base(obs)
        base = lambda obs: {"farmer": ["PASS"], "hands": [], "market": []}
        with mock.patch.object(cd, "RecordingAgent", Blind):
            with self.assertRaises(RuntimeError):
                cd._run(base, "starter", 0, 1, 48, 24)

    def test_generate_pair_end_to_end_yields_a_labelled_row(self):
        focal = cd.resolve_opponent("strategy_follower")
        row = cd.generate_pair(focal, cd.resolve_opponent("starter"), "starter",
                               seed=7, pair_index=0, episode_steps=240,
                               target_output="land_expansion")
        self.assertEqual(row["output_id"], "land_expansion")
        self.assertIn(row["label"], (-1, 0, 1))
        self.assertIn("money_delta", row["label_components"])
        self.assertIsInstance(row["accepted"], bool)
        self.assertEqual(row["schema_version"], "counterfactual-v3")


if __name__ == "__main__":
    unittest.main()
