"""Audit phase-10 causal-data quotas before phase-11 model training."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import shared_features
import watcher_config as wc


OUTPUT_IDS = ("land_expansion", "labor_expansion") + tuple(
    f"product:{product}:{direction}"
    for product in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
    for direction in ("increase", "avoid")
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(paths, min_accepted=wc.MIN_ACCEPTED, min_signed=wc.MIN_SIGNED,
          min_split_class=wc.MIN_PER_CLASS, min_families=wc.MIN_FAMILIES, max_attempts=2000):
    rows = []
    for path in paths:
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    recognized = Counter(row.get("output_id") for row in rows if row.get("output_id") in OUTPUT_IDS)
    unrecognized = Counter(row.get("output_id") for row in rows if row.get("output_id") not in OUTPUT_IDS)
    schema_versions = Counter(str(row.get("schema_version")) for row in rows)
    reports = {}
    for output_id in OUTPUT_IDS:
        attempted = [row for row in rows if row.get("output_id") == output_id]
        valid = []
        rejection = Counter()
        seed_splits, seed_families = defaultdict(set), defaultdict(set)
        for row in attempted:
            reasons = []
            if row.get("schema_version") != shared_features.SCHEMA_VERSION: reasons.append("schema_version_mismatch")
            if row.get("schema_hash") != shared_features.SCHEMA_HASH: reasons.append("schema_hash_mismatch")
            if row.get("feature_order") != list(shared_features.FEATURE_ORDER): reasons.append("feature_order_mismatch")
            features = row.get("features")
            if not isinstance(features, list) or len(features) != len(shared_features.FEATURE_ORDER): reasons.append("invalid_features")
            elif not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in features): reasons.append("invalid_features")
            if row.get("accepted") is not True: reasons.append("not_accepted")
            if not (isinstance(row.get("intervention_audit"), dict) and row["intervention_audit"].get("execution_verified")): reasons.append("execution_not_verified")
            if row.get("label") not in (-1, 0, 1): reasons.append("invalid_label")
            if row.get("split") not in wc.SPLITS: reasons.append("invalid_split")
            if not row.get("strategy_family"): reasons.append("missing_strategy_family")
            if isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int): reasons.append("missing_or_invalid_seed")
            if reasons:
                rejection.update(set(reasons))
            else:
                valid.append(row)
                seed_splits[row.get("seed")].add(row["split"])
                seed_families[row.get("seed")].add(row["strategy_family"])
        leaked = {seed for seed in seed_splits if seed is None or len(seed_splits[seed]) != 1 or len(seed_families[seed]) != 1}
        if leaked:
            rejected_rows = sum(row.get("seed") in leaked for row in valid)
            valid = [row for row in valid if row.get("seed") not in leaked]
            rejection["seed_group_split_or_family_leakage"] += rejected_rows
        labels = Counter(row["label"] for row in valid)
        split_labels = {split: Counter(row["label"] for row in valid if row["split"] == split)
                        for split in wc.SPLITS}
        family_labels = {family: Counter(row["label"] for row in valid if row["strategy_family"] == family)
                         for family in sorted({row["strategy_family"] for row in valid})}
        # "Usable family" must match train_watcher.usable_families exactly: >=
        # min_split_class of BOTH signs in train u val only (never test/confirm).
        # A family that is sign-imbalanced in the fit pool -- common for premium
        # product:*:avoid, where the scripted opponents rarely produce the rarer
        # sign -- is not usable however many signed rows it has in test/confirm.
        fitpool_family_labels = {
            family: Counter(row["label"] for row in valid
                            if row["strategy_family"] == family and row["split"] in ("train", "val"))
            for family in sorted({row["strategy_family"] for row in valid})}
        usable = [family for family, counts in fitpool_family_labels.items()
                  if counts[-1] >= min_split_class and counts[1] >= min_split_class]
        quota = (len(valid) >= min_accepted and labels[-1] >= min_signed and labels[1] >= min_signed and
                 len(usable) >= min_families and all(counts[-1] >= min_split_class and counts[1] >= min_split_class
                                                     for counts in split_labels.values()))
        reports[output_id] = {
            "attempts": len(attempted), "attempts_remaining": max(0, max_attempts - len(attempted)),
            "accepted_execution_verified": len(valid), "class_counts": dict(labels),
            "split_class_counts": {key: dict(value) for key, value in split_labels.items()},
            "family_class_counts": {key: dict(value) for key, value in family_labels.items()},
            "usable_families": usable, "rejection_counts": dict(rejection),
            "quota_complete": quota,
            "status": "ready_for_training" if quota else "deterministic_rules",
        }
    eligible = sorted(output_id for output_id, report in reports.items() if report["quota_complete"])
    return {
        "schema_version": shared_features.SCHEMA_VERSION,
        "schema_hash": shared_features.SCHEMA_HASH,
        "feature_count": len(shared_features.FEATURE_ORDER),
        "rows_loaded": len(rows),
        "input_schema_versions": dict(schema_versions),
        "recognized_output_attempts": dict(recognized),
        "unrecognized_output_attempts": dict(unrecognized),
        "dataset_files": [str(path) for path in paths],
        "dataset_hashes": {str(path): _hash(path) for path in paths},
        "thresholds": {"accepted": min_accepted, "positive": min_signed, "negative": min_signed,
                       "per_split_per_signed_class": min_split_class, "usable_families": min_families,
                       "attempt_cap": max_attempts},
        "eligible_outputs": eligible, "outputs": reports,
        "phase_11_status": "train_eligible_outputs" if eligible else "skipped_no_quota_complete_outputs",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/watcher_readiness_report.json"))
    args = parser.parse_args(argv)
    report = audit(args.rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}: {len(report['eligible_outputs'])} eligible outputs")


if __name__ == "__main__":
    main()
