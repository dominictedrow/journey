import ast
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import counterfactual_dataset as cd
import shared_features as sf
import train_watcher as tw
import watcher_config as wc


N_FEATURES = len(sf.FEATURE_ORDER)


def make_features(rng, offset=0.0, dim=0):
    vec = [rng.gauss(0.0, 1.0) for _ in range(N_FEATURES)]
    vec[dim] += offset
    return vec


def make_row(rng, label, split, strategy_family="fam_a", output_id="product:WHEAT:increase",
             sample_weight=1.0, accepted=True, schema_hash=None, feature_order=None,
             features=None, offset_dim=0, seed=1):
    if features is None:
        offset = {1: 3.0, -1: -3.0, 0: 0.0}[label]
        features = make_features(rng, offset, offset_dim)
    row = {
        "schema_version": sf.SCHEMA_VERSION,
        "schema_hash": sf.SCHEMA_HASH if schema_hash is None else schema_hash,
        "feature_order": list(sf.FEATURE_ORDER) if feature_order is None else feature_order,
        "features": features,
        "output_id": output_id,
        "label": label,
        "accepted": accepted,
        "strategy_family": strategy_family,
        "seed": seed,
        "seed_group": str(seed),
        "intervention_audit": {"execution_verified": True},
    }
    if split is not None:
        row["split"] = split
    if sample_weight is not None:
        row["sample_weight"] = sample_weight
    return row


class SchemaValidationTests(unittest.TestCase):
    def test_schema_validation_rejects_bad_rows(self):
        rng = random.Random(0)
        good = make_row(rng, 1, "train")
        bad_hash = make_row(rng, 1, "train", schema_hash="deadbeef")
        bad_order = make_row(rng, 1, "train", feature_order=list(sf.FEATURE_ORDER)[::-1])
        bad_features = make_row(rng, 1, "train")
        bad_features["features"][0] = float("nan")
        unaccepted = make_row(rng, 1, "train", accepted=False)

        kept, rejected = tw.validate_and_filter([good, bad_hash, bad_order, bad_features, unaccepted])

        self.assertEqual(kept, [good])
        reasons_seen = [r["reasons"] for r in rejected]
        self.assertIn(["schema_hash_mismatch"], reasons_seen)
        self.assertIn(["feature_order_mismatch"], reasons_seen)
        self.assertIn(["invalid_features"], reasons_seen)
        self.assertIn(["not_accepted"], reasons_seen)


class SplitResolutionTests(unittest.TestCase):
    def test_split_missing_is_rejected_never_defaulted(self):
        rng = random.Random(1)
        row = make_row(rng, 1, split=None)

        kept, rejected = tw.resolve_splits([row])

        self.assertEqual(kept, [])
        self.assertEqual(rejected[0]["reasons"], ["split_unresolved"])

    def test_split_resolves_via_manifest_when_inline_missing(self):
        rng = random.Random(2)
        row = make_row(rng, 1, split=None, strategy_family="fam_b")

        kept, rejected = tw.resolve_splits([row], families_manifest={"fam_b": "val"})

        self.assertEqual(rejected, [])
        self.assertEqual(kept[0]["_split"], "val")

    def test_inline_split_preferred_over_manifest(self):
        rng = random.Random(3)
        row = make_row(rng, 1, split="test", strategy_family="fam_b")

        kept, rejected = tw.resolve_splits([row], families_manifest={"fam_b": "val"})

        self.assertEqual(rejected, [])
        self.assertEqual(kept[0]["_split"], "test")

    def test_seed_group_leakage_rejects_entire_group(self):
        rng = random.Random(31)
        rows = [make_row(rng, 1, "train"), make_row(rng, -1, "test")]
        for row in rows:
            row["seed_group"] = 99
            row["_split"] = row["split"]
        kept, rejected = tw.reject_seed_split_leakage(rows)
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 2)

    def test_same_raw_seed_across_family_qualified_groups_is_rejected(self):
        rng = random.Random(32)
        rows = [make_row(rng, 1, "train", strategy_family="fam_a", seed=7),
                make_row(rng, -1, "train", strategy_family="fam_b", seed=7)]
        rows[0]["seed_group"], rows[1]["seed_group"] = "fam_a:7", "fam_b:7"
        for row in rows:
            row["_split"] = row["split"]
        kept, rejected = tw.reject_seed_split_leakage(rows)
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 2)


class PartitionTests(unittest.TestCase):
    def test_partition_never_reshuffles_rows_across_splits(self):
        rng = random.Random(4)
        rows = [make_row(rng, 1, "test"), make_row(rng, -1, "train"), make_row(rng, 1, "val")]
        for row in rows:
            row["_split"] = row["split"]

        partitions = tw.partition(rows)

        self.assertEqual([r["_split"] for r in partitions["train"]], ["train"])
        self.assertEqual([r["_split"] for r in partitions["val"]], ["val"])
        self.assertEqual([r["_split"] for r in partitions["test"]], ["test"])


def full_partitions(rng, per_class, splits=wc.SPLITS):
    return {name: ([make_row(rng, 1, name) for _ in range(per_class)] +
                   [make_row(rng, -1, name) for _ in range(per_class)])
            for name in splits}


class SplitAdequacyTests(unittest.TestCase):
    def test_check_split_adequacy_marks_degenerate_output_unavailable(self):
        rng = random.Random(5)
        partitions = full_partitions(rng, 3)
        partitions["val"] = [make_row(rng, 1, "val")]

        ok, reason = tw.check_split_adequacy(partitions, min_per_class=2)

        self.assertFalse(ok)
        self.assertEqual(reason, "val_insufficient_class_counts")

    def test_check_split_adequacy_requires_the_confirm_split(self):
        rng = random.Random(51)
        partitions = full_partitions(rng, 3)
        partitions["confirm"] = []

        ok, reason = tw.check_split_adequacy(partitions, min_per_class=2)

        self.assertFalse(ok)
        self.assertEqual(reason, "confirm_insufficient_class_counts")

    def test_check_split_adequacy_passes_when_all_splits_meet_minimum(self):
        rng = random.Random(6)
        partitions = full_partitions(rng, 2)

        ok, reason = tw.check_split_adequacy(partitions, min_per_class=2)

        self.assertTrue(ok)
        self.assertIsNone(reason)


class ScalerTests(unittest.TestCase):
    def test_scaler_fit_uses_train_rows_only(self):
        rng = random.Random(7)
        train_rows = ([make_row(rng, 1, "train", offset_dim=5) for _ in range(4)] +
                      [make_row(rng, -1, "train", offset_dim=5) for _ in range(4)])
        val_rows = [make_row(rng, 1, "val", offset_dim=5) for _ in range(4)]
        for row in val_rows:
            row["features"] = [x + 1000.0 for x in row["features"]]

        X_train = tw.to_matrix(train_rows)
        scaler = tw.fit_scaler(X_train)
        manual = StandardScaler().fit(np.array(X_train, dtype=float))

        np.testing.assert_allclose(scaler.mean_, manual.mean_)
        np.testing.assert_allclose(scaler.scale_, manual.scale_)
        self.assertGreater(abs(scaler.mean_[0] - 1000.0), 500.0)


class MarginScoreTests(unittest.TestCase):
    def test_margin_scale_and_pure_score_clip_at_bounds(self):
        rng = random.Random(8)
        rows = [make_row(rng, 1, "train") for _ in range(20)] + [make_row(rng, -1, "train") for _ in range(20)]
        X = tw.to_matrix(rows)
        scaler = tw.fit_scaler(X)
        X_scaled = scaler.transform(X)
        y = tw.to_labels(rows)
        model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=0)
        model.fit(X_scaled, y)

        margin = tw.margin_scale(model, X_scaled, 95)
        self.assertGreaterEqual(margin, 1e-6)

        mean, scale = tuple(scaler.mean_), tuple(scaler.scale_)
        coef, intercept = tuple(model.coef_[0]), float(model.intercept_[0])
        huge_positive = [x * 1e6 for x in rows[0]["features"]]
        huge_negative = [-x * 1e6 for x in rows[0]["features"]]
        pos_score = tw.pure_score(huge_positive, mean, scale, coef, intercept, margin)
        neg_score = tw.pure_score(huge_negative, mean, scale, coef, intercept, margin)

        self.assertIn(pos_score, (1.0, -1.0))
        self.assertIn(neg_score, (1.0, -1.0))
        self.assertLessEqual(abs(pos_score), 1.0)
        self.assertLessEqual(abs(neg_score), 1.0)

    def test_pure_score_returns_none_for_non_finite_input(self):
        mean = scale = coef = (0.0,) * N_FEATURES
        features = [float("nan")] * N_FEATURES
        self.assertIsNone(tw.pure_score(features, mean, scale, coef, 0.0, 1.0))


class GateTests(unittest.TestCase):
    def test_gate_reads_the_bootstrap_ci_lower_bound_not_a_point_estimate(self):
        floor = wc.ACCEPT_MIN_BALANCED_ACC_CI_LOWER
        # A high point estimate with a CI lower bound below the floor fails --
        # the whole point of the change is that a lucky partition no longer passes.
        self.assertFalse(tw.gate({"balanced_accuracy_unweighted": 0.82,
                                   "bootstrap_balanced_accuracy": {
                                       "point": 0.82, "ci_lower": floor - 0.05, "ci_upper": 0.95}}))
        self.assertTrue(tw.gate({"balanced_accuracy_unweighted": 0.65,
                                  "bootstrap_balanced_accuracy": {
                                      "point": 0.65, "ci_lower": floor, "ci_upper": 0.78}}))
        # No bootstrap CI at all (e.g. an empty test split) can never pass.
        self.assertFalse(tw.gate({"balanced_accuracy_unweighted": 0.99}))

    def test_gate_cv_reads_selected_c_mean(self):
        per_c = {0.1: {"cv_mean_balanced_accuracy": wc.ACCEPT_MIN_CV_MEAN - 0.01},
                 1.0: {"cv_mean_balanced_accuracy": wc.ACCEPT_MIN_CV_MEAN + 0.03}}
        self.assertTrue(tw.gate_cv(1.0, per_c))
        self.assertFalse(tw.gate_cv(0.1, per_c))
        self.assertFalse(tw.gate_cv(None, per_c))

    def test_gate_confirm_allows_within_ci_width_and_rejects_beyond(self):
        test_metrics = {"bootstrap_balanced_accuracy": {"point": 0.70, "ci_lower": 0.60, "ci_upper": 0.80}}
        self.assertTrue(tw.gate_confirm(test_metrics, {"balanced_accuracy_unweighted": 0.62}))
        self.assertFalse(tw.gate_confirm(test_metrics, {"balanced_accuracy_unweighted": 0.45}))
        self.assertFalse(tw.gate_confirm(test_metrics, {}))

    def test_pooled_leave_one_family_out_gate(self):
        good = {"status": "evaluated",
                "pooled_balanced_accuracy": wc.ACCEPT_MIN_LOFO_POOLED + 0.02, "n_pooled": 80}
        weak = dict(good, pooled_balanced_accuracy=wc.ACCEPT_MIN_LOFO_POOLED - 0.05)
        self.assertTrue(tw.gate_leave_one_family_out(good))
        self.assertFalse(tw.gate_leave_one_family_out(weak))
        self.assertFalse(tw.gate_leave_one_family_out({"status": "not_evaluable",
                                                        "pooled_balanced_accuracy": None}))


class BootstrapTests(unittest.TestCase):
    def test_bootstrap_ci_brackets_point_and_lower_le_upper(self):
        y_true = [1] * 30 + [-1] * 30
        y_pred = [1] * 25 + [-1] * 5 + [-1] * 25 + [1] * 5   # ~0.83 bacc
        ci = tw.bootstrap_balanced_accuracy_ci(y_true, y_pred, iterations=500)
        self.assertLessEqual(ci["ci_lower"], ci["point"])
        self.assertLessEqual(ci["point"], ci["ci_upper"])
        self.assertGreater(ci["n_resamples"], 400)

    def test_bootstrap_ci_of_a_coin_flip_predictor_has_a_low_lower_bound(self):
        rng = random.Random(0)
        y_true = [rng.choice((1, -1)) for _ in range(40)]
        y_pred = [rng.choice((1, -1)) for _ in range(40)]
        ci = tw.bootstrap_balanced_accuracy_ci(y_true, y_pred, iterations=500)
        self.assertLess(ci["ci_lower"], wc.ACCEPT_MIN_BALANCED_ACC_CI_LOWER)


class ExportRoundTripTests(unittest.TestCase):
    def test_export_round_trip_matches_decision_function(self):
        rng = random.Random(9)
        rows = [make_row(rng, 1, "train", offset_dim=2) for _ in range(20)] + \
               [make_row(rng, -1, "train", offset_dim=2) for _ in range(20)]
        holdout = [make_row(rng, 1, "test", offset_dim=2) for _ in range(5)] + \
                  [make_row(rng, -1, "test", offset_dim=2) for _ in range(5)]

        X = tw.to_matrix(rows)
        scaler = tw.fit_scaler(X)
        X_scaled = scaler.transform(X)
        y = tw.to_labels(rows)
        model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=0)
        model.fit(X_scaled, y)
        margin = tw.margin_scale(model, X_scaled, 95)

        mean, scale = tuple(scaler.mean_), tuple(scaler.scale_)
        coef, intercept = tuple(model.coef_[0]), float(model.intercept_[0])
        X_hold = tw.to_matrix(holdout)
        X_hold_scaled = scaler.transform(X_hold)
        expected = np.clip(model.decision_function(X_hold_scaled) / margin, -1.0, 1.0)

        for feats, exp in zip(X_hold, expected):
            got = tw.pure_score(feats, mean, scale, coef, intercept, margin)
            self.assertAlmostEqual(got, float(exp), places=6)


class WeightTests(unittest.TestCase):
    def test_missing_sample_weight_defaults_to_one_and_counts_missing(self):
        rng = random.Random(10)
        row_with = make_row(rng, 1, "train", sample_weight=2.5)
        row_without = make_row(rng, 1, "train", sample_weight=None)

        weights, missing = tw.to_weights([row_with, row_without])

        self.assertEqual(weights, [2.5, 1.0])
        self.assertEqual(missing, 1)

    def test_renormalize_weights_rescales_to_mean_one(self):
        weights = tw.renormalize_weights([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)


class OutputGroupingTests(unittest.TestCase):
    def test_output_id_grouping_matches_counterfactual_dataset_routing(self):
        rng = random.Random(11)
        deviation = {"kind": "animal_add_one", "item": "COW"}
        output_id = cd.modeled_output_id(deviation)
        self.assertEqual(output_id, "product:MILK:increase")
        row = make_row(rng, 1, "train", output_id=output_id)

        grouped = tw.group_by_output([row])

        self.assertIn(output_id, grouped)
        self.assertEqual(grouped[output_id], [row])


class LeaveOneFamilyOutTests(unittest.TestCase):
    @staticmethod
    def _clean(rng, label, split, family):
        # Only dim 4 varies; every other feature is a constant the scaler zeroes
        # out, so the target is cleanly cross-family separable at tiny n.
        vec = [0.0] * N_FEATURES
        vec[4] = (5.0 if label == 1 else -5.0) + rng.gauss(0.0, 0.3)
        return make_row(rng, label, split, strategy_family=family, features=vec)

    def _separable(self, rng, n, family, split):
        return ([self._clean(rng, 1, split, family) for _ in range(n)] +
                [self._clean(rng, -1, split, family) for _ in range(n)])

    def test_pooled_metric_covers_train_and_val_only(self):
        rng = random.Random(12)
        train_rows = self._separable(rng, 8, "fam_a", "train") + self._separable(rng, 8, "fam_b", "train")
        val_rows = self._separable(rng, 3, "fam_a", "val") + self._separable(rng, 3, "fam_b", "val")

        lofo = tw.leave_one_family_out(train_rows, val_rows, 1.0, min_per_class=2)

        self.assertEqual(lofo["status"], "evaluated")
        self.assertEqual(lofo["n_pooled"], 44)  # 22 fam_a binary + 22 fam_b binary
        self.assertGreater(lofo["pooled_balanced_accuracy"], 0.8)
        self.assertEqual(lofo["per_family"]["fam_a"]["status"], "evaluated")

    def test_single_sign_family_is_scored_without_its_own_per_family_bacc(self):
        rng = random.Random(13)
        pool = (self._separable(rng, 8, "fam_a", "train") + self._separable(rng, 8, "fam_b", "train") +
                [self._clean(rng, 1, "train", "fam_c")])

        lofo = tw.leave_one_family_out(pool, [], 1.0, min_per_class=2)

        # fam_c's lone +1 row is still predicted and pooled; it just has no
        # two-class per-family balanced accuracy of its own.
        self.assertEqual(lofo["per_family"]["fam_c"]["status"], "evaluated")
        self.assertIsNone(lofo["per_family"]["fam_c"]["balanced_accuracy_unweighted"])
        self.assertEqual(lofo["n_pooled"], 33)

    def test_pure_noise_target_does_not_clear_the_pooled_gate(self):
        rng = random.Random(14)
        # label carries no signal: features are drawn the same regardless of sign
        def noise(n, family, split):
            return [make_row(rng, sign, split, strategy_family=family,
                             features=make_features(rng))
                    for _ in range(n) for sign in (1, -1)]
        train_rows = noise(12, "fam_a", "train") + noise(12, "fam_b", "train")
        val_rows = noise(4, "fam_a", "val") + noise(4, "fam_b", "val")

        lofo = tw.leave_one_family_out(train_rows, val_rows, 1.0, min_per_class=2)

        self.assertFalse(tw.gate_leave_one_family_out(lofo))


class RunEndToEndTests(unittest.TestCase):
    OUTPUT = "product:MELON:increase"
    # (family, split): (n_pos, n_neg) -- fam_c is deliberately sign-imbalanced,
    # the shape that made the old per-family LOFO gate structurally unpassable.
    LAYOUT = {
        ("fam_a", "train"): (20, 20), ("fam_a", "val"): (6, 6),
        ("fam_a", "test"): (8, 8), ("fam_a", "confirm"): (6, 6),
        ("fam_b", "train"): (12, 12), ("fam_b", "val"): (4, 4),
        ("fam_b", "test"): (6, 6), ("fam_b", "confirm"): (5, 5),
        ("fam_c", "train"): (10, 2), ("fam_c", "val"): (3, 0),
        ("fam_c", "test"): (2, 2), ("fam_c", "confirm"): (2, 2),
    }

    def _corpus(self, path, separable):
        rng = random.Random(2024)
        seed = 500_000
        with open(path, "w") as fh:
            for (family, split), (n_pos, n_neg) in self.LAYOUT.items():
                for label, count in ((1, n_pos), (-1, n_neg)):
                    for _ in range(count):
                        feats = None if separable else make_features(rng)
                        row = make_row(rng, label, split, strategy_family=family,
                                       output_id=self.OUTPUT, features=feats,
                                       offset_dim=4, seed=seed)
                        seed += 1
                        fh.write(json.dumps(row) + "\n")

    def test_separable_target_with_an_imbalanced_family_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            self._corpus(corpus, separable=True)
            report = tw.run([corpus], output_path=Path(tmp) / "watcher_model.py")
        result = report["outputs"][self.OUTPUT]
        self.assertEqual(result["status"], "accepted", msg=json.dumps(result, default=str)[:2000])
        self.assertEqual(result["usable_strategy_families"], ["fam_a", "fam_b"])
        self.assertTrue(result["leave_one_family_out_gate_passed"])
        self.assertTrue(result["confirm_gate_passed"])

    def test_noise_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus.jsonl"
            self._corpus(corpus, separable=False)
            report = tw.run([corpus], output_path=Path(tmp) / "watcher_model.py")
        result = report["outputs"][self.OUTPUT]
        self.assertIn(result["status"], ("rejected", "unavailable"))
        self.assertNotEqual(result.get("reason"), None)


class GeneratedModuleTests(unittest.TestCase):
    def test_generated_module_source_has_no_forbidden_imports_and_round_trips(self):
        rng = random.Random(14)
        rows = [make_row(rng, 1, "train", offset_dim=3) for _ in range(20)] + \
               [make_row(rng, -1, "train", offset_dim=3) for _ in range(20)]
        X = tw.to_matrix(rows)
        scaler = tw.fit_scaler(X)
        X_scaled = scaler.transform(X)
        y = tw.to_labels(rows)
        model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=0)
        model.fit(X_scaled, y)
        margin = tw.margin_scale(model, X_scaled, 95)

        record = tw.export_record(
            "product:WHEAT:increase", model, scaler, margin, 1.0, 0.9,
            {"balanced_accuracy_unweighted": 0.9, "accuracy_unweighted": 0.9,
             "bootstrap_balanced_accuracy": {"point": 0.9, "ci_lower": 0.7, "ci_upper": 0.98}},
            {"balanced_accuracy_unweighted": 0.88},
            {"train": 40, "val": 10, "test": 10, "confirm": 8})
        metadata = {
            "schema_version": sf.SCHEMA_VERSION, "schema_hash": sf.SCHEMA_HASH,
            "feature_order": list(sf.FEATURE_ORDER), "generated_at": "2026-01-01T00:00:00+00:00",
            "dataset_snapshot": {"row_files": [], "row_file_hashes": {}, "families_manifest": None},
        }
        source = tw.render_module({"product:WHEAT:increase": record}, metadata)

        tree = ast.parse(source)
        forbidden = {"numpy", "sklearn"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], forbidden)
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)

        namespace = {}
        exec(source, namespace)
        self.assertIn("product:WHEAT:increase", namespace["MODELS"])
        got = namespace["score"]("product:WHEAT:increase", rows[0]["features"])
        self.assertIsInstance(got, float)
        self.assertIsNone(namespace["score"]("no_such_output_id", rows[0]["features"]))


if __name__ == "__main__":
    unittest.main()
