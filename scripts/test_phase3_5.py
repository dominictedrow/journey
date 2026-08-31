import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import evaluation_metrics
import replay_baselines
import strategy_follower as strategy


def _step(day, hour, money, action, hands=0, tile=None, status="ACTIVE"):
    farm = {
        "money": money, "hands": [[4, 4]] * hands, "farmer": [4, 4],
        "tiles": [[tile]], "unlocked_quadrants": ["NW"],
    }
    obs = {"day": day, "hour": hour, "farms": [farm, farm],
           "private": {"shed": {"MELON": 2}, "inventories": []},
           "market": {"prices": {"MELON": 200}}}
    state = SimpleNamespace(observation=obs, action=action, status=status)
    return [state, state]


class PhaseMetricsTests(unittest.TestCase):
    def test_episode_metrics_counts_actions_orders_and_inventory(self):
        plant = {"kind": "PLANT", "crop": "MELON", "watered_today": False, "yield_units": 0}
        env = SimpleNamespace(steps=[
            _step(0, 0, 3000, {"farmer": ["PASS"], "hands": [], "market": []}),
            _step(0, 1, 2999, {"farmer": ["WATER"], "hands": [["EAST"]],
                               "market": [["SELL", "MELON", 2]] + [["HIRE"]] * 10}, hands=1, tile=plant),
            _step(0, 2, 3200, {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}, hands=1, tile=plant, status="DONE"),
        ])
        result = evaluation_metrics.episode_metrics(env, 0)
        self.assertEqual(result["market_order_cap_excess_requests"], 1)
        self.assertEqual(result["requested_sale_quantity_by_product"]["MELON"], 2)
        self.assertIsNone(result["executed_market_orders"])
        self.assertFalse(result["market_execution_observable"])
        self.assertEqual(result["unsold_ending_inventory"], 2)
        self.assertGreater(result["hand_action_shares"]["0"]["work"], 0)

    def test_purchase_metrics_separate_quantity_from_spending(self):
        action = {"farmer": ["PASS"], "hands": [], "market": [
            ["BUY_SEED", "MELON", 2], ["BUY_ANIMAL", "SHEEP", 1],
            ["BUY_PRODUCT", "WHEAT", 3], ["BUY_LAND"], ["HIRE"], ["HIRE"],
        ]}
        step = _step(0, 0, 5000, action)
        step[0].observation["market"]["prices"]["WHEAT"] = 30
        env = SimpleNamespace(steps=[step])
        result = evaluation_metrics.episode_metrics(env, 0)
        self.assertEqual(result["requested_purchase_quantity"], {
            "seed": 2, "animal": 1, "feed": 3, "land": 1, "hire": 2})
        self.assertEqual(result["requested_purchase_spending"], {
            "seed": 160, "animal": 500, "feed": 90, "land": 1000, "hire": 2})

    def test_conditional_adaptive_target_can_request_configured_third_land(self):
        saved = strategy.LAND_EXPANSION_MODE, strategy.LAND_TARGET_QUADRANTS
        try:
            strategy.LAND_EXPANSION_MODE = "conditional"
            strategy.LAND_TARGET_QUADRANTS = 3
            tiles = [[None] * 10 for _ in range(10)]
            me = {"money": 10000, "hands": [], "unlocked_quadrants": ["NW", "NE"]}
            targets = strategy._adaptive_targets(
                {"day": 5, "hour": 0}, me, tiles, dict(strategy._BASE_PRICE))
            self.assertEqual(targets["quadrants"], 3)
        finally:
            strategy.LAND_EXPANSION_MODE, strategy.LAND_TARGET_QUADRANTS = saved

    def test_conditional_land_uses_unique_layout_and_can_emit_purchase(self):
        fixture = Path(__file__).resolve().parents[1] / "kaggle_replays" / "replays" / "episode-93479646-replay.json"
        obs = copy.deepcopy(json.loads(fixture.read_text())["steps"][0][0]["observation"])
        names = ("LAND_EXPANSION_MODE", "LAND_TARGET_QUADRANTS", "LAND_MIN_DAY",
                 "LAND_UTILIZATION_THRESHOLD", "LAND_MIN_REPAY_DAYS", "LAND_OPERATING_RESERVE",
                 "LIVESTOCK_POSITIONS", "MELON_POSITIONS", "WHEAT_POSITIONS", "CARROT_POSITIONS",
                 "_planner_has_capacity")
        saved = {name: getattr(strategy, name) for name in names}
        try:
            strategy.LAND_EXPANSION_MODE = "conditional"
            strategy.LAND_TARGET_QUADRANTS = 3
            strategy.LAND_MIN_DAY = 0
            strategy.LAND_UTILIZATION_THRESHOLD = 0.8
            strategy.LAND_MIN_REPAY_DAYS = 8
            strategy.LAND_OPERATING_RESERVE = 0
            strategy.LIVESTOCK_POSITIONS = [(x, y) for y in range(5) for x in range(5)]
            strategy.MELON_POSITIONS = [(x, y) for y in range(5) for x in range(5, 10)]
            strategy.WHEAT_POSITIONS = [(0, 5)]
            strategy.CARROT_POSITIONS = []
            strategy._planner_has_capacity = lambda *args, **kwargs: True
            self.assertEqual(len(strategy._planned_tile_positions()), 51)

            obs["day"], obs["hour"] = 10, 3
            me = obs["farms"][obs["player"]]
            me["money"] = 10000
            me["unlocked_quadrants"] = ["NW", "NE"]
            me["tiles"] = [[{"kind": "PASTURE"} if y < 4 else None
                            for x in range(10)] for y in range(10)]
            self.assertIn(["BUY_LAND"], strategy.agent(obs)["market"])

            strategy.WHEAT_POSITIONS = strategy.MELON_POSITIONS[:1]
            self.assertEqual(len(strategy._planned_tile_positions()), 50)
            self.assertNotIn(["BUY_LAND"], strategy.agent(obs)["market"])
        finally:
            for name, value in saved.items():
                setattr(strategy, name, value)

    def test_hiring_modes_are_bounded(self):
        old_mode = strategy.HIRING_RAMP_MODE
        try:
            for mode in ("fixed", "calendar", "workload", "cash_workload"):
                strategy.HIRING_RAMP_MODE = mode
                target = strategy._daily_hand_target(6, 200, 3000, 500, 20)
                self.assertGreaterEqual(target, 0)
                self.assertLessEqual(target, min(strategy.N_HIRES, strategy.HIRING_CEILING))
        finally:
            strategy.HIRING_RAMP_MODE = old_mode

    def test_replay_baseline_smoke(self):
        names = ["turn", "own_hands", "own_quadrants", "own_crop_WHEAT", "own_crop_CARROT",
                 "own_crop_TOMATO", "own_crop_STRAWBERRY", "own_crop_MELON"]
        rows = [{"episode_id": 1, "player_order": 0, "step": 0, "day": 0,
                 "feature_order": names, "features": [0, 4, 1, 0, 0, 0, 0, 2],
                 "observed_action": {"farmer": ["PASS"], "hands": [["PASS"]] * 4, "market": []}}]
        report = replay_baselines.build(rows)
        self.assertEqual(report["episodes"], 1)
        self.assertEqual(report["daily_hand_count_ranges"]["0"]["median"], 4)

    def test_fixed_crop_transition_deadlines(self):
        saved = {name: getattr(strategy, name) for name in (
            "MELON_LAST_PLANT_DAY", "TRANSITION_EMPTY_POLICY", "STRAWBERRY_START_DAY",
            "STRAWBERRY_TILES", "STRAWBERRY_LAST_PLANT_DAY", "LATE_WHEAT_START_DAY",
            "LATE_WHEAT_TILES", "WHEAT_LAST_PLANT_DAY",
        )}
        try:
            strategy.MELON_LAST_PLANT_DAY = 10
            strategy.TRANSITION_EMPTY_POLICY = "strawberry"
            strategy.STRAWBERRY_START_DAY = 11
            strategy.STRAWBERRY_TILES = 4
            strategy.STRAWBERRY_LAST_PLANT_DAY = 19
            strategy.LATE_WHEAT_START_DAY = 23
            strategy.LATE_WHEAT_TILES = 8
            strategy.WHEAT_LAST_PLANT_DAY = 25
            self.assertEqual(strategy._fixed_transition_crop(10, 0), "MELON")
            self.assertEqual(strategy._fixed_transition_crop(11, 3), "STRAWBERRY")
            self.assertIsNone(strategy._fixed_transition_crop(11, 4))
            self.assertEqual(strategy._fixed_transition_crop(23, 4), "WHEAT")
            self.assertIsNone(strategy._fixed_transition_crop(26, 0))
        finally:
            for name, value in saved.items():
                setattr(strategy, name, value)

    def test_phase7_and_phase8_controls_are_gated(self):
        self.assertEqual(strategy.SALE_POLICY_MODE, "legacy")
        self.assertEqual(strategy.LAND_EXPANSION_MODE, "legacy")
        self.assertEqual(strategy.LAND_TARGET_QUADRANTS, 2)
        self.assertLessEqual(strategy.SALE_ROUTINE_ORDER_BUDGET, 10)
        self.assertEqual(strategy.SALE_MIN_PRICE_FRAC_BY_PRODUCT["WHEAT"], 0.0)
        self.assertGreater(strategy.SALE_MIN_PRICE_FRAC_BY_PRODUCT["WOOL"], 0.0)

    def test_product_sale_policy_batches_and_liquidates(self):
        fixture = Path(__file__).resolve().parents[1] / "kaggle_replays" / "replays" / "episode-93479646-replay.json"
        obs = json.loads(fixture.read_text())["steps"][0][0]["observation"]
        saved = strategy.SALE_POLICY_MODE
        try:
            strategy.SALE_POLICY_MODE = "product_specific"
            obs = copy.deepcopy(obs)
            obs["day"], obs["hour"] = 15, 10
            obs["private"]["shed"] = {"WHEAT": 40, "WOOL": 10}
            obs["market"]["prices"].update({"WHEAT": 25, "WOOL": 20})
            sales = strategy.agent(obs)["market"]
            self.assertIn(["SELL", "WHEAT", 3], sales)
            self.assertFalse(any(order[0] == "SELL" and order[1] == "WOOL" for order in sales))
            obs["day"] = 29
            liquidation = strategy.agent(obs)["market"]
            self.assertIn(["SELL", "WHEAT", 40], liquidation)
            self.assertIn(["SELL", "WOOL", 10], liquidation)
        finally:
            strategy.SALE_POLICY_MODE = saved


if __name__ == "__main__":
    unittest.main()
