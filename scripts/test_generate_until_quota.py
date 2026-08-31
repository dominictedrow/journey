import unittest

import generate_until_quota as guq


def _rows(output_id, family, count, accepted):
    return [{"output_id": output_id, "strategy_family": family, "accepted": i < accepted}
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

    def test_choose_family_prefers_least_attempted_among_live_families(self):
        rows = (_rows("land_expansion", "strategy_follower", 100, 45)
                + _rows("land_expansion", "strategies_v2.ring.agent", 150, 46))
        family = guq.choose_family(rows, "land_expansion", dead_min_attempts=15)
        self.assertEqual(family, "strategies_v2.leader_clone.agent")


if __name__ == "__main__":
    unittest.main()
