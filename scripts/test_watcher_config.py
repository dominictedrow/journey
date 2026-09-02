"""watcher_config.py is the single source of truth for quotas/thresholds.

The generator (generate_until_quota), the readiness audit (watcher_readiness)
and the trainer (train_watcher) must never disagree on what "ready for
training" means -- the pre-repair toolchain had three different values for
min_accepted, min_signed and min_per_class.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import generate_until_quota as guq
import train_watcher as tw
import watcher_config as wc
import watcher_readiness as wr
from test_train_watcher import make_row


class SharedThresholdTests(unittest.TestCase):
    def test_all_three_tools_read_the_same_constants(self):
        import inspect
        # Every tool imports watcher_config rather than keeping its own copy.
        for module in (guq, wr, tw):
            self.assertIs(module.wc, wc)
        wr_defaults = inspect.signature(wr.audit).parameters
        self.assertEqual(wr_defaults["min_accepted"].default, wc.MIN_ACCEPTED)
        self.assertEqual(wr_defaults["min_signed"].default, wc.MIN_SIGNED)
        self.assertEqual(wr_defaults["min_split_class"].default, wc.MIN_PER_CLASS)
        tw_defaults = inspect.signature(tw.run).parameters
        self.assertEqual(tw_defaults["min_accepted"].default, wc.MIN_ACCEPTED)
        self.assertEqual(tw_defaults["min_signed"].default, wc.MIN_SIGNED)
        self.assertEqual(tw_defaults["min_per_class"].default, wc.MIN_PER_CLASS)

    def test_phase_bc_deviation_contract_constants_are_present(self):
        self.assertGreaterEqual(wc.DIVEST_VERIFY_MIN_GAP, 1)
        self.assertGreaterEqual(wc.OPPONENT_VOLUME_UNITS_PER_DAY, 1)
        import counterfactual_dataset as cd
        self.assertIs(cd.wc, wc)

    def test_readiness_and_trainer_agree_on_one_synthetic_corpus(self):
        output_id = "product:MELON:increase"
        rng = random.Random(7)
        seed = 900_000
        rows = []
        # Comfortably over quota, two usable families, all four splits populated.
        layout = {
            ("fam_a", "train"): (24, 24), ("fam_a", "val"): (7, 7),
            ("fam_a", "test"): (9, 9), ("fam_a", "confirm"): (7, 7),
            ("fam_b", "train"): (14, 14), ("fam_b", "val"): (5, 5),
            ("fam_b", "test"): (7, 7), ("fam_b", "confirm"): (6, 6),
        }
        for (family, split), (n_pos, n_neg) in layout.items():
            for label, count in ((1, n_pos), (-1, n_neg)):
                for _ in range(count):
                    vec = [0.0] * len(tw.shared_features.FEATURE_ORDER)
                    vec[4] = (4.0 if label == 1 else -4.0) + rng.gauss(0, 0.3)
                    rows.append(make_row(rng, label, split, strategy_family=family,
                                         output_id=output_id, features=vec, seed=seed))
                    seed += 1

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

            readiness = wr.audit([corpus])
            report = tw.run([corpus], output_path=Path(tmp) / "wm.py")
            guq_accepted, guq_labels, _ = guq.counts(rows, output_id)

        ready = output_id in readiness["eligible_outputs"]
        trainer_has_quota = report["outputs"][output_id]["status"] != "unavailable"
        guq_quota = (guq_accepted >= wc.MIN_ACCEPTED and
                     guq_labels[-1] >= wc.MIN_SIGNED and guq_labels[1] >= wc.MIN_SIGNED)

        self.assertTrue(ready)
        self.assertTrue(trainer_has_quota)
        self.assertTrue(guq_quota)
        self.assertEqual(ready, trainer_has_quota)

    def test_readiness_and_trainer_agree_when_a_family_is_fit_pool_sign_imbalanced(self):
        # fam_b clears the per-sign quota overall but only via test/confirm -1
        # rows -- in train u val it has +1s and no -1s. The trainer's
        # usable_families rejects it; readiness must reach the same verdict
        # (not the pre-fix "any -1 and any +1 across all splits" rule).
        output_id = "product:MELON:avoid"
        rng = random.Random(11)
        seed = 950_000
        rows = []
        layout = {
            ("fam_a", "train"): (24, 24), ("fam_a", "val"): (7, 7),
            ("fam_a", "test"): (9, 9), ("fam_a", "confirm"): (7, 7),
            ("fam_b", "train"): (14, 0), ("fam_b", "val"): (5, 0),
            ("fam_b", "test"): (7, 14), ("fam_b", "confirm"): (6, 12),
        }
        for (family, split), (n_pos, n_neg) in layout.items():
            for label, count in ((1, n_pos), (-1, n_neg)):
                for _ in range(count):
                    vec = [0.0] * len(tw.shared_features.FEATURE_ORDER)
                    vec[4] = (4.0 if label == 1 else -4.0) + rng.gauss(0, 0.3)
                    rows.append(make_row(rng, label, split, strategy_family=family,
                                         output_id=output_id, features=vec, seed=seed))
                    seed += 1

        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            readiness = wr.audit([corpus])
            report = tw.run([corpus], output_path=Path(tmp) / "wm.py")

        ready = output_id in readiness["eligible_outputs"]
        trainer_has_quota = report["outputs"][output_id]["status"] != "unavailable"
        self.assertFalse(ready)
        self.assertFalse(trainer_has_quota)
        self.assertEqual(ready, trainer_has_quota)


if __name__ == "__main__":
    unittest.main()
