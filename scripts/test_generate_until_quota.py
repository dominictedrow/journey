import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import generate_until_quota as guq
import watcher_config as wc


def _rows(output_id, family, count, accepted, label=None):
    return [{"output_id": output_id, "strategy_family": family, "accepted": i < accepted,
             "label": label if i < accepted else None}
            for i in range(count)]


class GenerateUntilQuotaTests(unittest.TestCase):
    def test_family_with_zero_acceptance_past_threshold_is_dead(self):
        rows = _rows("product:MELON:avoid", "strategies_v2.ring.agent", 20, 0)
        dead = guq.dead_families(rows, "product:MELON:avoid", dead_min_attempts=15)
        self.assertIn("strategies_v2.ring.agent", dead)

    def test_family_under_the_attempt_threshold_is_not_yet_dead(self):
        rows = _rows("product:MELON:avoid", "strategies_v2.ring.agent", 10, 0)
        dead = guq.dead_families(rows, "product:MELON:avoid", dead_min_attempts=15)
        self.assertNotIn("strategies_v2.ring.agent", dead)

    def test_family_with_any_accepted_row_is_never_dead(self):
        rows = _rows("product:MELON:avoid", "strategy_follower", 400, 1)
        dead = guq.dead_families(rows, "product:MELON:avoid", dead_min_attempts=15)
        self.assertNotIn("strategy_follower", dead)

    def test_choose_family_skips_dead_family_already_proven_for_this_output(self):
        rows = (_rows("product:MELON:avoid", "strategy_follower", 150, 102)
                + _rows("product:MELON:avoid", "strategies_v2.ring.agent", 400, 0)
                + _rows("product:MELON:avoid", "strategies_v2.leader_clone.agent", 100, 0))
        family = guq.choose_family(rows, "product:MELON:avoid", dead_min_attempts=15)
        self.assertEqual(family, "strategy_follower")

    def test_choose_family_returns_none_when_every_family_is_dead(self):
        rows = (_rows("product:WOOL:exit", "strategy_follower", 700, 0)
                + _rows("product:WOOL:exit", "strategies_v2.ring.agent", 700, 0)
                + _rows("product:WOOL:exit", "strategies_v2.leader_clone.agent", 600, 0))
        self.assertIsNone(guq.choose_family(rows, "product:WOOL:exit", dead_min_attempts=15))

    def test_all_neutral_output_has_unreachable_signed_quota(self):
        rows = _rows("product:CARROT:avoid", "strategy_follower", 500, 200, label=0)
        self.assertTrue(guq.signed_quota_unreachable(rows, "product:CARROT:avoid", 400))

    def test_one_absent_sign_makes_the_signed_quota_unreachable(self):
        # 500 attempts, plenty of +1, zero -1 -> rarer sign is 0, cannot reach 40.
        rows = (_rows("product:WOOL:increase", "strategy_follower", 500, 220, label=0)
                + _rows("product:WOOL:increase", "strategies_v2.ring.agent", 60, 40, label=1))
        self.assertTrue(guq.signed_quota_unreachable(rows, "product:WOOL:increase", 400))

    def test_healthy_signed_rate_keeps_the_output_alive(self):
        rows = (_rows("land_expansion", "strategy_follower", 300, 90, label=1)
                + _rows("land_expansion", "strategies_v2.ring.agent", 200, 45, label=-1))
        self.assertFalse(guq.signed_quota_unreachable(rows, "land_expansion", 400))

    def test_signed_quota_check_waits_for_the_attempt_floor(self):
        rows = _rows("product:CARROT:avoid", "strategy_follower", 200, 80, label=0)
        self.assertFalse(guq.signed_quota_unreachable(rows, "product:CARROT:avoid", 400))

    def test_choose_family_prefers_least_attempted_among_live_families(self):
        rows = (_rows("land_expansion", "strategy_follower", 100, 45)
                + _rows("land_expansion", "strategies_v2.ring.agent", 150, 46))
        family = guq.choose_family(rows, "land_expansion", dead_min_attempts=15)
        self.assertEqual(family, "strategies_v2.leader_clone.agent")

    def test_signed_quota_unreachable_defaults_to_shared_min_signed(self):
        import inspect
        self.assertEqual(
            inspect.signature(guq.signed_quota_unreachable).parameters["min_signed"].default,
            wc.MIN_SIGNED)

    def test_probe_cache_is_built_and_round_trips(self):
        summary = {"focal_order_days": {"BUY_SEED:MELON": [1, 4]},
                   "opponent_volume_days": {"MELON": [10]}}
        fake_run = lambda *a, **k: (None, SimpleNamespace(steps=[]), 0, {}, [])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("counterfactual_dataset._run", fake_run), \
                mock.patch("counterfactual_dataset._probe_summary", lambda *a, **k: summary), \
                mock.patch("counterfactual_dataset.resolve_opponent", lambda name: name):
            path = Path(tmp) / "probe.json"
            guq.build_probe_cache(path, ["strategy_follower"], ["starter", "glut"], 240)
            payload = json.loads(path.read_text())
            self.assertEqual(set(payload["entries"]),
                             {"strategy_follower|starter", "strategy_follower|glut"})
            self.assertEqual(payload["entries"]["strategy_follower|starter"], summary)
            import counterfactual_dataset as cd
            self.assertEqual(cd.load_probe_cache(path)["strategy_follower|glut"], summary)


if __name__ == "__main__":
    unittest.main()
