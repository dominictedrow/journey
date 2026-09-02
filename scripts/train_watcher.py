"""Fit, evaluate, gate, and export per-output_id watcher LinearSVC models.

Reads counterfactual_dataset.py-shaped JSONL rows (plus the strategy_family/
split/sample_weight fields owned by Dom's preprocessing pipeline), fits one
LinearSVC per output_id, margin-scales its decision function to [-1, 1], and
exports accepted models as a pure-Python scripts/watcher_model.py -- the only
artifact strategy_follower.py's runtime is allowed to import.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import shared_features
import watcher_config as wc


DEFAULT_C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
DEFAULT_MARGIN_PERCENTILE = 95
DEFAULT_MAX_ITER = 20_000

# Phase 1 gate: a bootstrap CI lower bound on the untouched test split, a
# grouped-CV mean for C selection, and a pooled leave-one-family-out metric.
# There is no point-estimate "bacc >= 0.60" clause any more and no margin term
# (the old `(bacc - 0.5) >= 0.10` clause was algebraically identical to the
# 0.60 floor -- see git history). All thresholds live in watcher_config.
ACCEPT_MIN_BALANCED_ACC_CI_LOWER = wc.ACCEPT_MIN_BALANCED_ACC_CI_LOWER
_GATE_EPSILON = 1e-9

# Exact source templated into scripts/watcher_model.py -- pure_score() below
# must stay numerically identical to this, and the export round-trip test
# checks that against sklearn's own decision_function.
_SCORE_FUNCTION_SOURCE = '''def score(output_id, features):
    """Pure-python decision score in [-1.0, 1.0], or None if unavailable/invalid."""
    entry = MODELS.get(output_id)
    if entry is None or len(features) != len(entry["coef"]):
        return None
    total = entry["intercept"]
    for x, m, s, c in zip(features, entry["scaler_mean"], entry["scaler_scale"], entry["coef"]):
        if not math.isfinite(x):
            return None
        total += ((x - m) / s) * c
    if not math.isfinite(total):
        return None
    return max(-1.0, min(1.0, total / entry["margin_scale"]))
'''


def load_rows(paths):
    """Read JSONL rows from every path, tagging each with its source location."""
    rows = []
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_path"] = str(path)
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def _reject(row, reasons):
    return {"source_path": row.get("_source_path"), "line_no": row.get("_line_no"), "reasons": reasons}


def validate_and_filter(rows):
    """Split rows into (kept, rejected), collecting every rejection reason per row."""
    kept, rejected = [], []
    for row in rows:
        reasons = []
        if row.get("schema_version") != shared_features.SCHEMA_VERSION:
            reasons.append("schema_version_mismatch")
        if row.get("schema_hash") != shared_features.SCHEMA_HASH:
            reasons.append("schema_hash_mismatch")
        if row.get("feature_order") != list(shared_features.FEATURE_ORDER):
            reasons.append("feature_order_mismatch")
        if row.get("accepted") is not True:
            reasons.append("not_accepted")
        features = row.get("features")
        if (features is None or len(features) != len(shared_features.FEATURE_ORDER) or
                not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                        for x in features)):
            reasons.append("invalid_features")
        if row.get("label") not in (-1, 0, 1):
            reasons.append("invalid_label")
        if not row.get("output_id"):
            reasons.append("missing_output_id")
        if not row.get("strategy_family"):
            reasons.append("missing_strategy_family")
        if isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int):
            reasons.append("missing_or_invalid_seed")
        audit = row.get("intervention_audit")
        if not isinstance(audit, dict) or not audit.get("execution_verified"):
            reasons.append("execution_not_verified")
        weight = row.get("sample_weight")
        if (isinstance(weight, bool) or not isinstance(weight, (int, float)) or
                not math.isfinite(weight) or weight <= 0):
            reasons.append("invalid_sample_weight")
        if reasons:
            rejected.append(_reject(row, reasons))
        else:
            kept.append(row)
    return kept, rejected


def resolve_splits(rows, families_manifest=None):
    """Attach `_split` per row from the inline field, else the family manifest; never default."""
    kept, rejected = [], []
    for row in rows:
        split = row.get("split")
        if split not in wc.SPLITS and families_manifest is not None:
            split = families_manifest.get(row.get("strategy_family"))
        if split not in wc.SPLITS:
            rejected.append(_reject(row, ["split_unresolved"]))
            continue
        resolved = dict(row)
        resolved["_split"] = split
        kept.append(resolved)
    return kept, rejected


def reject_seed_split_leakage(rows):
    """Reject seed groups crossing either splits or focal strategy families."""
    seed_splits, seed_families = defaultdict(set), defaultdict(set)
    for row in rows:
        seed_splits[row.get("seed")].add(row["_split"])
        seed_families[row.get("seed")].add(row.get("strategy_family"))
    leaked = {seed for seed, splits in seed_splits.items()
              if seed is None or len(splits) != 1 or len(seed_families[seed]) != 1}
    kept = [row for row in rows if row.get("seed") not in leaked]
    rejected = [_reject(row, ["seed_group_split_or_family_leakage"])
                for row in rows if row.get("seed") in leaked]
    return kept, rejected


def group_by_output(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["output_id"]].append(row)
    return dict(grouped)


def partition(rows_for_output):
    """{split: [...]} for every split in watcher_config.SPLITS, strictly from stored `_split`."""
    result = {name: [] for name in wc.SPLITS}
    for row in rows_for_output:
        split = row["_split"]
        if split not in result:
            raise ValueError(f"unexpected split {split!r} for output_id {row.get('output_id')!r}")
        result[split].append(row)
    return result


def binary_only(rows):
    return [row for row in rows if row["label"] != 0]


def check_split_adequacy(partitions, min_per_class):
    """Each required split's binary_only() must have >= min_per_class of both classes.

    `confirm` is only required when watcher_config.REQUIRE_CONFIRM_SPLIT is set;
    every other split in SPLITS is always required.
    """
    for name in wc.SPLITS:
        if name == "confirm" and not wc.REQUIRE_CONFIRM_SPLIT:
            continue
        counts = Counter(row["label"] for row in binary_only(partitions[name]))
        if counts[-1] < min_per_class or counts[1] < min_per_class:
            return False, f"{name}_insufficient_class_counts"
    return True, None


def usable_families(partitions, min_per_class):
    """Families with >= min_per_class of BOTH signs in train u val (never test/confirm).

    The single "usable family" definition shared by run() eligibility and LOFO,
    so an output can no longer clear the family count on test-only signed rows
    and then fail leave-one-family-out for lack of the same rows in the fit pool.
    """
    pool = list(partitions["train"]) + list(partitions["val"])
    by_family = defaultdict(Counter)
    for row in pool:
        if row["label"] != 0:
            by_family[row["strategy_family"]][row["label"]] += 1
    return sorted(family for family, counts in by_family.items()
                  if counts[-1] >= min_per_class and counts[1] >= min_per_class)


def to_matrix(rows):
    return [list(row["features"]) for row in rows]


def constant_feature_names(rows):
    """Feature columns that never vary across `rows` -- zero information for the
    fit, reported so the effective dimensionality is visible against the row
    count (repair-plan 3.3). Pass the fit pool (train u val)."""
    matrix = to_matrix(rows)
    if not matrix:
        return []
    names = shared_features.FEATURE_ORDER
    return [names[col] for col in range(len(matrix[0]))
            if len({record[col] for record in matrix}) <= 1]


def to_labels(rows):
    return [int(row["label"]) for row in rows]


def to_weights(rows, default=1.0):
    """Extract sample_weight per row; missing/non-finite weights fall back to `default`."""
    weights, missing = [], 0
    for row in rows:
        weight = row.get("sample_weight")
        if (weight is None or isinstance(weight, bool) or
                not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0):
            weights.append(default)
            missing += 1
        else:
            weights.append(float(weight))
    return weights, missing


def renormalize_weights(weights):
    """Rescale weights to mean 1.0 over the rows passed -- call with train-split rows only."""
    if not weights:
        return list(weights)
    mean = sum(weights) / len(weights)
    if mean <= 0:
        return list(weights)
    return [w / mean for w in weights]


def fit_scaler(X_train):
    return StandardScaler().fit(np.asarray(X_train, dtype=float))


def balanced_accuracy(y_true, y_pred, weights=None):
    return float(balanced_accuracy_score(y_true, y_pred, sample_weight=weights))


def raw_accuracy(y_true, y_pred, weights=None):
    """Unweighted-scale accuracy -- the like-for-like comparator for majority_baseline (also raw-scale)."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    if weights is None:
        weights = [1.0] * len(y_true)
    total_weight = sum(weights)
    if not total_weight:
        return 0.0
    correct = sum(w for t, p, w in zip(y_true, y_pred, weights) if t == p)
    return correct / total_weight


def majority_baseline(y_true, weights=None):
    """Raw majority-class proportion -- NOT balanced accuracy of a majority predictor (always 0.5)."""
    y_true = list(y_true)
    if weights is None:
        weights = [1.0] * len(y_true)
    totals = {}
    for label, w in zip(y_true, weights):
        totals[label] = totals.get(label, 0.0) + w
    total_weight = sum(weights)
    return max(totals.values()) / total_weight if total_weight else 0.0


def _group_of(row):
    """Seed-group identity used to keep CV folds leak-free (falls back to seed)."""
    return str(row.get("seed_group", row.get("seed")))


def select_c_grouped_kfold(fit_rows, c_grid, n_folds=wc.CV_FOLDS,
                           max_iter=DEFAULT_MAX_ITER):
    """Pick C by mean balanced accuracy under grouped stratified k-fold CV.

    `fit_rows` is train u val (never test/confirm). Groups are seed groups, so a
    seed group is never split across a fold boundary. Ties in mean CV balanced
    accuracy break toward the smaller C. Returns (best_c, per_c_metrics) or
    (None, per_c_metrics) when there are too few seed groups to cross-validate.
    """
    binary = binary_only(fit_rows)
    y = np.asarray(to_labels(binary))
    groups = np.asarray([_group_of(row) for row in binary])
    n_groups = len(set(groups))
    folds = min(n_folds, n_groups)
    per_c_metrics = {}
    if folds < 2 or min(Counter(int(v) for v in y).values()) < folds:
        return None, {"reason": "insufficient_seed_groups_for_cv",
                      "n_groups": n_groups, "requested_folds": n_folds}
    X = np.asarray(to_matrix(binary), dtype=float)
    weights = np.asarray(to_weights(binary)[0], dtype=float)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=0)
    fold_indices = list(splitter.split(X, y, groups))
    best = None
    for c in c_grid:
        fold_scores = []
        for train_idx, test_idx in fold_indices:
            if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
                continue
            scaler = StandardScaler().fit(X[train_idx])
            model = LinearSVC(C=c, class_weight="balanced", max_iter=max_iter, random_state=0)
            model.fit(scaler.transform(X[train_idx]), y[train_idx],
                      sample_weight=weights[train_idx])
            pred = model.predict(scaler.transform(X[test_idx]))
            fold_scores.append(balanced_accuracy(y[test_idx], pred))
        mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
        per_c_metrics[c] = {"cv_mean_balanced_accuracy": mean_score,
                            "cv_fold_balanced_accuracies": [float(s) for s in fold_scores],
                            "cv_evaluable_folds": len(fold_scores)}
        if best is None or mean_score > best[1] + _GATE_EPSILON:
            best = (c, mean_score)
    return (best[0] if best else None), per_c_metrics


def bootstrap_balanced_accuracy_ci(y_true, y_pred, iterations=wc.BOOTSTRAP_ITERATIONS,
                                   lower_pct=wc.BOOTSTRAP_CI_LOWER_PERCENTILE,
                                   upper_pct=wc.BOOTSTRAP_CI_UPPER_PERCENTILE,
                                   seed=wc.BOOTSTRAP_SEED):
    """Percentile bootstrap CI on balanced accuracy over paired (y_true, y_pred).

    Single-class resamples are skipped rather than scored. Returns the point
    estimate plus the requested lower/upper percentiles; `ci_lower` is what the
    Phase 1 gate reads.
    """
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    point = balanced_accuracy(y_true, y_pred)
    n = len(y_true)
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(iterations):
        idx = rng.randint(0, n, size=n)
        if len(set(y_true[idx])) < 2:
            continue
        scores.append(balanced_accuracy(y_true[idx], y_pred[idx]))
    if not scores:
        return {"point": point, "ci_lower": point, "ci_upper": point,
                "n_resamples": 0, "n_rows": n}
    return {"point": point,
            "ci_lower": float(np.percentile(scores, lower_pct)),
            "ci_upper": float(np.percentile(scores, upper_pct)),
            "n_resamples": len(scores), "n_rows": n}


def margin_scale(model, X_train_scaled, percentile):
    decision = np.abs(model.decision_function(np.asarray(X_train_scaled, dtype=float)))
    return max(1e-6, float(np.percentile(decision, percentile)))


def pure_score(features, mean, scale, coef, intercept, margin_scale_value):
    """Exact formula templated into watcher_model.py's score(); tested against decision_function."""
    total = intercept
    for x, m, s, c in zip(features, mean, scale, coef):
        if not math.isfinite(x):
            return None
        total += ((x - m) / s) * c
    if not math.isfinite(total):
        return None
    return max(-1.0, min(1.0, total / margin_scale_value))


def evaluate_split(model, scaler, margin_scale_value, rows, bootstrap=False):
    """Untouched-split metrics. `majority_baseline_*` is diagnostic_only, never a gate input.

    With `bootstrap=True` (the test split), a percentile bootstrap CI on
    balanced accuracy is attached under `bootstrap_balanced_accuracy` -- its
    `ci_lower` is the Phase 1 gate input.
    """
    class_counts = Counter(row["label"] for row in rows)
    binary = binary_only(rows)
    result = {"class_counts": dict(class_counts), "n_rows": len(rows), "n_binary_rows": len(binary)}
    if not binary:
        return result
    X = scaler.transform(to_matrix(binary))
    y = to_labels(binary)
    weights, weight_missing = to_weights(binary)
    pred = model.predict(X)
    result.update({
        "weight_missing": weight_missing,
        "balanced_accuracy_unweighted": balanced_accuracy(y, pred),
        "balanced_accuracy_weighted": balanced_accuracy(y, pred, weights),
        "accuracy_unweighted": raw_accuracy(y, pred),
        "accuracy_weighted": raw_accuracy(y, pred, weights),
        "majority_baseline_unweighted_diagnostic_only": majority_baseline(y),
        "confusion_matrix": confusion_matrix(y, pred, labels=[-1, 1]).tolist(),
    })
    if bootstrap:
        result["bootstrap_balanced_accuracy"] = bootstrap_balanced_accuracy_ci(y, pred)
    return result


def leave_one_family_out(train_rows, val_rows, c, min_per_class, max_iter=DEFAULT_MAX_ITER):
    """Pooled held-family evaluation. Pool = train+val only (test cannot reach here).

    For each family: refit on all *other* families, predict that family's signed
    rows. Every held-out prediction is pooled and **one** balanced accuracy is
    computed over the pool -- a family that individually lacks both signs still
    contributes its rows without having to clear the bar alone. The per-family
    breakdown is kept as a diagnostic only.
    """
    pool = list(train_rows) + list(val_rows)
    families = sorted({row["strategy_family"] for row in pool})
    per_family = {}
    pooled_true, pooled_pred = [], []
    for family in families:
        held_out = [row for row in pool if row["strategy_family"] == family]
        fit_rows = [row for row in pool if row["strategy_family"] != family]
        fit_binary, held_binary = binary_only(fit_rows), binary_only(held_out)
        fit_counts = Counter(row["label"] for row in fit_binary)
        if fit_counts[-1] < min_per_class or fit_counts[1] < min_per_class or not held_binary:
            per_family[family] = {"status": "skipped", "reason": "insufficient_fit_or_held_rows",
                                   "n_held_out": len(held_binary)}
            continue
        scaler = fit_scaler(to_matrix(fit_rows))
        X_fit, y_fit = scaler.transform(to_matrix(fit_binary)), to_labels(fit_binary)
        weights_fit, _ = to_weights(fit_binary)
        model = LinearSVC(C=c, class_weight="balanced", max_iter=max_iter, random_state=0)
        model.fit(X_fit, y_fit, sample_weight=weights_fit)
        X_held, y_held = scaler.transform(to_matrix(held_binary)), to_labels(held_binary)
        pred = [int(v) for v in model.predict(X_held)]
        pooled_true.extend(y_held)
        pooled_pred.extend(pred)
        held_counts = Counter(y_held)
        per_family[family] = {
            "status": "evaluated", "n_held_out": len(held_binary),
            "class_counts": {str(k): v for k, v in held_counts.items()},
            "balanced_accuracy_unweighted": (
                balanced_accuracy(y_held, pred) if len(held_counts) > 1 else None),
        }
    pooled_counts = Counter(pooled_true)
    evaluable = len(pooled_true) >= wc.LOFO_MIN_POOLED_ROWS and len(pooled_counts) > 1
    return {
        "status": "evaluated" if evaluable else "not_evaluable",
        "pooled_balanced_accuracy": balanced_accuracy(pooled_true, pooled_pred) if evaluable else None,
        "n_pooled": len(pooled_true),
        "pooled_class_counts": {str(k): v for k, v in pooled_counts.items()},
        "per_family": per_family,
    }


def gate(test_metrics):
    """Accepted iff the bootstrap CI lower bound on test balanced accuracy clears the floor.

    No point-estimate clause and no margin term: the old
    `bacc >= 0.60 and (bacc - 0.5) >= 0.10` was two spellings of one 0.60
    threshold on a single tiny partition. The gate input is now the 5th
    percentile of a 2000-sample bootstrap, so a 10-per-class "pass" is honest.
    """
    ci = test_metrics.get("bootstrap_balanced_accuracy")
    if not ci:
        return False
    return ci["ci_lower"] >= ACCEPT_MIN_BALANCED_ACC_CI_LOWER - _GATE_EPSILON


def gate_leave_one_family_out(lofo):
    """Pooled held-family balanced accuracy clears watcher_config.ACCEPT_MIN_LOFO_POOLED."""
    if not lofo or lofo.get("status") != "evaluated" or lofo.get("pooled_balanced_accuracy") is None:
        return False
    return lofo["pooled_balanced_accuracy"] >= wc.ACCEPT_MIN_LOFO_POOLED - _GATE_EPSILON


def gate_cv(best_c, per_c_cv_metrics):
    """Grouped-CV mean balanced accuracy for the selected C clears ACCEPT_MIN_CV_MEAN."""
    entry = per_c_cv_metrics.get(best_c) if best_c is not None else None
    if not entry or "cv_mean_balanced_accuracy" not in entry:
        return False
    return entry["cv_mean_balanced_accuracy"] >= wc.ACCEPT_MIN_CV_MEAN - _GATE_EPSILON


def gate_confirm(test_metrics, confirm_metrics):
    """Untouched confirm split must land within the test CI width of the test point estimate.

    Opened once, after the candidate is otherwise finalized; never feeds back
    into C selection or any other decision.
    """
    if not wc.REQUIRE_CONFIRM_SPLIT:
        return True
    ci = test_metrics.get("bootstrap_balanced_accuracy")
    if not ci or "balanced_accuracy_unweighted" not in confirm_metrics:
        return False
    width = max(ci["ci_upper"] - ci["ci_lower"], _GATE_EPSILON)
    return abs(confirm_metrics["balanced_accuracy_unweighted"] - ci["point"]) <= width + _GATE_EPSILON


def export_record(output_id, model, scaler, margin_scale_value, c,
                   cv_mean_balanced_accuracy, test_metrics, confirm_metrics, split_counts):
    assert tuple(int(x) for x in model.classes_) == (-1, 1), \
        f"unexpected class order for {output_id}: {tuple(model.classes_)}"
    ci = test_metrics.get("bootstrap_balanced_accuracy", {})
    return {
        "scaler_mean": tuple(float(x) for x in scaler.mean_),
        "scaler_scale": tuple(float(x) for x in scaler.scale_),
        "coef": tuple(float(x) for x in model.coef_[0]),
        "intercept": float(model.intercept_[0]),
        "margin_scale": float(margin_scale_value),
        "C": float(c),
        "class_order": (-1, 1),
        "cv_mean_balanced_accuracy": float(cv_mean_balanced_accuracy),
        "test_balanced_accuracy_unweighted": float(test_metrics["balanced_accuracy_unweighted"]),
        "test_balanced_accuracy_ci_lower": float(ci.get("ci_lower", float("nan"))),
        "test_balanced_accuracy_ci_upper": float(ci.get("ci_upper", float("nan"))),
        "confirm_balanced_accuracy_unweighted": float(confirm_metrics.get("balanced_accuracy_unweighted", float("nan"))),
        "train_rows": int(split_counts["train"]),
        "val_rows": int(split_counts["val"]),
        "test_rows": int(split_counts["test"]),
        "confirm_rows": int(split_counts["confirm"]),
    }


def render_module(records, metadata):
    """Generate watcher_model.py source. Zero sklearn/numpy imports -- math only."""
    parts = [
        '"""Auto-generated by scripts/train_watcher.py -- do not hand-edit."""',
        "import math", "",
        f"SCHEMA_VERSION = {metadata['schema_version']!r}",
        f"SCHEMA_HASH = {metadata['schema_hash']!r}",
        f"FEATURE_ORDER = {tuple(metadata['feature_order'])!r}",
        f"GENERATED_AT = {metadata['generated_at']!r}",
        f"DATASET_SNAPSHOT = {metadata['dataset_snapshot']!r}",
        "", "MODELS = {",
    ]
    for output_id in sorted(records):
        parts.append(f"    {output_id!r}: {records[output_id]!r},")
    parts += ["}", "", "# only accepted output_ids are present; MODELS.get(output_id) is None otherwise.", ""]
    parts.append(_SCORE_FUNCTION_SOURCE.rstrip("\n"))
    return "\n".join(parts) + "\n"


def render_report(results, metadata):
    report = {"metadata": metadata, "outputs": results}
    json_str = json.dumps(report, indent=2, sort_keys=True) + "\n"

    lines = ["# Watcher training report", "",
             f"Generated at: {metadata.get('generated_at')}",
             f"Schema hash: {metadata.get('schema_hash')}",
             f"Rows loaded / kept / rejected: {metadata.get('rows_loaded')} / "
             f"{metadata.get('rows_kept')} / {metadata.get('rows_rejected')}", "",
             "| output_id | status | reason | C | cv_mean_bacc | test_bacc | test_ci_lower | "
             "lofo_pooled | confirm_bacc | accepted |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for output_id in sorted(results):
        r = results[output_id]
        test_metrics = r.get("test_metrics", {})
        ci = test_metrics.get("bootstrap_balanced_accuracy", {})
        lofo = r.get("leave_one_family_out", {})
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            output_id, r.get("status", ""), r.get("reason", ""), r.get("C", ""),
            r.get("cv_mean_balanced_accuracy", ""),
            test_metrics.get("balanced_accuracy_unweighted", ""),
            ci.get("ci_lower", ""),
            lofo.get("pooled_balanced_accuracy", ""),
            r.get("confirm_metrics", {}).get("balanced_accuracy_unweighted", ""),
            r.get("accepted", "")))
    return json_str, "\n".join(lines) + "\n"


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(rows_paths, families_path=None, output_path=None, report_path=None,
        c_grid=DEFAULT_C_GRID, margin_percentile=DEFAULT_MARGIN_PERCENTILE,
        min_per_class=wc.MIN_PER_CLASS, reweight_mode="renormalize-train",
        min_accepted=wc.MIN_ACCEPTED, min_signed=wc.MIN_SIGNED, min_families=wc.MIN_FAMILIES,
        max_iter=DEFAULT_MAX_ITER):
    rows_paths = [Path(p) for p in rows_paths]
    rows = load_rows(rows_paths)
    kept, rejected = validate_and_filter(rows)

    families_manifest = None
    if families_path is not None:
        raw_manifest = json.loads(Path(families_path).read_text(encoding="utf-8"))
        families = raw_manifest.get("families") if isinstance(raw_manifest, dict) else None
        if isinstance(families, list):
            families_manifest = {
                row.get("family_id", row.get("strategy_family")): row.get("split")
                for row in families
                if row.get("family_id", row.get("strategy_family"))
            }
        elif isinstance(raw_manifest.get("family_splits") if isinstance(raw_manifest, dict) else None, dict):
            families_manifest = raw_manifest["family_splits"]
        else:
            families_manifest = raw_manifest
    kept, split_rejected = resolve_splits(kept, families_manifest)
    rejected = rejected + split_rejected
    kept, leakage_rejected = reject_seed_split_leakage(kept)
    rejected = rejected + leakage_rejected

    by_output = group_by_output(kept)
    results, records = {}, {}
    for output_id in sorted(by_output):
        output_rows = by_output[output_id]
        partitions = partition(output_rows)
        split_counts = {name: len(rs) for name, rs in partitions.items()}
        signed_counts = Counter(row["label"] for row in output_rows)
        families = sorted({row["strategy_family"] for row in output_rows})
        usable = usable_families(partitions, min_per_class)
        if (len(output_rows) < min_accepted or signed_counts[-1] < min_signed or
                signed_counts[1] < min_signed):
            results[output_id] = {
                "status": "unavailable", "reason": "overall_quota_not_met",
                "split_counts": split_counts,
                "overall_class_counts": dict(signed_counts),
            }
            continue
        if len(usable) < min_families:
            results[output_id] = {
                "status": "unavailable", "reason": "insufficient_usable_strategy_families",
                "split_counts": split_counts, "strategy_families": families,
                "usable_strategy_families": usable,
            }
            continue
        ok, reason = check_split_adequacy(partitions, min_per_class)
        if not ok:
            results[output_id] = {"status": "unavailable", "reason": reason, "split_counts": split_counts}
            continue

        train_rows, val_rows = partitions["train"], partitions["val"]
        test_rows, confirm_rows = partitions["test"], partitions["confirm"]
        fit_rows = list(train_rows) + list(val_rows)
        fit_binary = binary_only(fit_rows)
        constant_columns = constant_feature_names(fit_binary)

        best_c, per_c_cv = select_c_grouped_kfold(fit_rows, c_grid, wc.CV_FOLDS, max_iter)
        if best_c is None:
            results[output_id] = {"status": "unavailable", "reason": "cross_validation_unavailable",
                                   "split_counts": split_counts, "cv_diagnostic": per_c_cv,
                                   "constant_feature_columns": constant_columns}
            continue

        scaler = fit_scaler(to_matrix(fit_rows))
        X_fit, y_fit = scaler.transform(to_matrix(fit_binary)), to_labels(fit_binary)
        weights_fit, weight_missing_fit = to_weights(fit_binary)
        if reweight_mode == "renormalize-train":
            weights_fit = renormalize_weights(weights_fit)
        best_model = LinearSVC(C=best_c, class_weight="balanced", max_iter=max_iter, random_state=0)
        best_model.fit(X_fit, y_fit, sample_weight=weights_fit)
        margin = margin_scale(best_model, X_fit, margin_percentile)

        test_metrics = evaluate_split(best_model, scaler, margin, test_rows, bootstrap=True)
        confirm_metrics = evaluate_split(best_model, scaler, margin, confirm_rows)
        lofo = leave_one_family_out(train_rows, val_rows, best_c, min_per_class, max_iter)

        cv_passed = gate_cv(best_c, per_c_cv)
        test_passed = gate(test_metrics)
        lofo_passed = gate_leave_one_family_out(lofo)
        confirm_passed = gate_confirm(test_metrics, confirm_metrics)
        accepted = cv_passed and test_passed and lofo_passed and confirm_passed

        result = {
            "status": "accepted" if accepted else "rejected",
            "C": best_c,
            "cv_mean_balanced_accuracy": per_c_cv[best_c]["cv_mean_balanced_accuracy"],
            "per_c_cv_metrics": per_c_cv,
            "test_metrics": test_metrics,
            "confirm_metrics": confirm_metrics,
            "split_counts": split_counts,
            "weight_missing_fit": weight_missing_fit,
            "constant_feature_columns": constant_columns,
            "effective_feature_count": len(shared_features.FEATURE_ORDER) - len(constant_columns),
            "usable_strategy_families": usable,
            "leave_one_family_out": lofo,
            "cv_gate_passed": cv_passed,
            "test_gate_passed": test_passed,
            "leave_one_family_out_gate_passed": lofo_passed,
            "confirm_gate_passed": confirm_passed,
            "reason": None if accepted else (
                "cv_gate_failed" if not cv_passed else
                "test_gate_failed" if not test_passed else
                "leave_one_family_out_gate_failed" if not lofo_passed else
                "confirm_gate_failed"
            ),
            "accepted": accepted,
        }
        results[output_id] = result
        if accepted:
            records[output_id] = export_record(
                output_id, best_model, scaler, margin, best_c,
                per_c_cv[best_c]["cv_mean_balanced_accuracy"], test_metrics,
                confirm_metrics, split_counts)

    rejection_counts = Counter(reason for entry in rejected for reason in entry["reasons"])
    metadata = {
        "schema_version": shared_features.SCHEMA_VERSION,
        "schema_hash": shared_features.SCHEMA_HASH,
        "feature_order": list(shared_features.FEATURE_ORDER),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_snapshot": {
            "row_files": [str(p) for p in rows_paths],
            "row_file_hashes": {str(p): _file_hash(p) for p in rows_paths},
            "families_manifest": str(families_path) if families_path else None,
        },
        "reweight_mode": reweight_mode,
        "c_grid": list(c_grid),
        "margin_percentile": margin_percentile,
        "max_iter": max_iter,
        "min_per_class": min_per_class,
        "min_accepted": min_accepted,
        "min_signed": min_signed,
        "min_families": min_families,
        "gate_thresholds": {
            "cv_mean_balanced_accuracy": wc.ACCEPT_MIN_CV_MEAN,
            "test_balanced_accuracy_ci_lower": wc.ACCEPT_MIN_BALANCED_ACC_CI_LOWER,
            "leave_one_family_out_pooled": wc.ACCEPT_MIN_LOFO_POOLED,
            "cv_folds": wc.CV_FOLDS,
            "bootstrap_iterations": wc.BOOTSTRAP_ITERATIONS,
            "require_confirm_split": wc.REQUIRE_CONFIRM_SPLIT,
        },
        "rows_loaded": len(rows),
        "rows_kept": len(kept),
        "rows_rejected": len(rejected),
        "rejection_reason_counts": dict(rejection_counts),
    }

    if output_path is not None:
        Path(output_path).write_text(render_module(records, metadata), encoding="utf-8")

    report = {"metadata": metadata, "outputs": results}
    if report_path is not None:
        report_path = Path(report_path)
        json_str, md_str = render_report(results, metadata)
        report_path.with_suffix(".json").write_text(json_str, encoding="utf-8")
        report_path.with_suffix(".md").write_text(md_str, encoding="utf-8")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", nargs="+", required=True, type=Path)
    p.add_argument("--families", type=Path, default=None)
    p.add_argument("--output", type=Path, default=Path("scripts/watcher_model.py"))
    p.add_argument("--report", type=Path, default=Path("watcher_training_report"))
    p.add_argument("--c-grid", nargs="+", type=float, default=list(DEFAULT_C_GRID))
    p.add_argument("--margin-percentile", type=float, default=DEFAULT_MARGIN_PERCENTILE)
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                   help=f"maximum Liblinear iterations per SVM fit (default: {DEFAULT_MAX_ITER})")
    p.add_argument("--min-per-class", type=int, default=wc.MIN_PER_CLASS)
    p.add_argument("--min-accepted", type=int, default=wc.MIN_ACCEPTED)
    p.add_argument("--min-signed", type=int, default=wc.MIN_SIGNED)
    p.add_argument("--min-families", type=int, default=wc.MIN_FAMILIES)
    p.add_argument("--reweight-mode", choices=("asis", "renormalize-train"), default="renormalize-train")
    args = p.parse_args(argv)
    if args.min_per_class < wc.SMOKE_TEST_MIN_PER_CLASS:
        print(f"!!! SMOKE TEST ONLY: --min-per-class {args.min_per_class} is below the "
              f"{wc.SMOKE_TEST_MIN_PER_CLASS}-per-class floor. Balanced accuracy and the "
              f"bootstrap CI computed from this few samples are NOT trustworthy; no model "
              f"trained this way may be promoted.", file=sys.stderr)
    report = run(args.rows, args.families, args.output, args.report,
                 tuple(args.c_grid), args.margin_percentile, args.min_per_class,
                 args.reweight_mode, args.min_accepted, args.min_signed, args.min_families,
                 args.max_iter)
    accepted = sum(1 for r in report["outputs"].values() if r.get("status") == "accepted")
    print(f"trained {len(report['outputs'])} output_ids ({accepted} accepted) -> {args.output}")


if __name__ == "__main__":
    main()
