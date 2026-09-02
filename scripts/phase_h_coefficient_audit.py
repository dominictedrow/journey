"""Phase H / 5.2 -- is the retrained SVM learning the opponent-adaptive signal?

repair.md Phase H, step 5.2: "retrain with `product:*:avoid` now trainable.
Inspect the fitted coefficients on `price_ratio_*`, `market_displacement_*` and
`opponent_*`. If they are near zero, the SVM is not learning the adaptive
signal regardless of accuracy -- same failure mode the plan already names."

For every output_id that reaches model fitting on the v6 corpus (i.e. clears
the quota + split-adequacy gates), this refits train_watcher's exact final
model (same C selection, scaler, class weights) and reports the L1 mass of the
standardized coefficient vector broken out by feature group:

  market     -- price_ratio_* + market_displacement_* (the shared-market signal)
  opponent   -- opponent_* farm-state counts (the opponent-reading signal)
  own        -- own_* farm-state
  phase/misc -- remaining_fraction, phase_*, capacity_backlog, ...

"Adaptive mass" = market + opponent share of total |coef|. Near zero => the
model is deciding from its own board, not from the opponent or the market.

    python scripts/phase_h_coefficient_audit.py \
        --rows data/counterfactual_dataset.v6.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import shared_features
import train_watcher as tw
import watcher_config as wc

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def _group(name: str) -> str:
    if name.startswith(("price_ratio_", "market_displacement_")):
        return "market"
    if name.startswith("opponent_"):
        return "opponent"
    if name.startswith(("own_", "asset_service_ratio", "capacity_backlog", "estimated_travel")):
        return "own"
    return "phase_misc"


def _fit_final_model(output_rows, c_grid, max_iter):
    """Mirror train_watcher.run()'s final-model fit for one output_id."""
    partitions = tw.partition(output_rows)
    fit_rows = list(partitions["train"]) + list(partitions["val"])
    fit_binary = tw.binary_only(fit_rows)
    best_c, per_c_cv = tw.select_c_grouped_kfold(fit_rows, c_grid, wc.CV_FOLDS, max_iter)
    if best_c is None:
        return None
    scaler = tw.fit_scaler(tw.to_matrix(fit_rows))
    X_fit = scaler.transform(tw.to_matrix(fit_binary))
    y_fit = tw.to_labels(fit_binary)
    weights_fit, _ = tw.to_weights(fit_binary)
    weights_fit = tw.renormalize_weights(weights_fit)
    model = tw.LinearSVC(C=best_c, class_weight="balanced", max_iter=max_iter, random_state=0)
    model.fit(X_fit, y_fit, sample_weight=weights_fit)
    return {
        "C": float(best_c),
        "cv_mean_balanced_accuracy": float(per_c_cv[best_c]["cv_mean_balanced_accuracy"]),
        "coef": model.coef_[0],
        "n_fit_binary": len(fit_binary),
    }


def audit(rows_paths):
    rows = tw.load_rows([Path(p) for p in rows_paths])
    kept, _ = tw.validate_and_filter(rows)
    kept, _ = tw.resolve_splits(kept, None)
    kept, _ = tw.reject_seed_split_leakage(kept)
    by_output = tw.group_by_output(kept)

    feature_order = list(shared_features.FEATURE_ORDER)
    groups = [_group(n) for n in feature_order]

    report = {
        "corpus": [str(p) for p in rows_paths],
        "schema_hash": shared_features.SCHEMA_HASH,
        "min_accepted": wc.MIN_ACCEPTED,
        "min_signed": wc.MIN_SIGNED,
        "outputs": {},
    }

    for output_id in sorted(by_output):
        output_rows = by_output[output_id]
        partitions = tw.partition(output_rows)
        signed = Counter(row["label"] for row in output_rows)
        usable = tw.usable_families(partitions, wc.MIN_PER_CLASS)
        quota_ok = (len(output_rows) >= wc.MIN_ACCEPTED
                    and signed[-1] >= wc.MIN_SIGNED and signed[1] >= wc.MIN_SIGNED)
        adequacy_ok, adequacy_reason = tw.check_split_adequacy(partitions, wc.MIN_PER_CLASS)
        if not (quota_ok and len(usable) >= wc.MIN_FAMILIES and adequacy_ok):
            report["outputs"][output_id] = {
                "fitted": False,
                "reason": ("quota_not_met" if not quota_ok
                           else "insufficient_usable_families" if len(usable) < wc.MIN_FAMILIES
                           else adequacy_reason),
                "signed_class_counts": {str(k): v for k, v in signed.items()},
            }
            continue

        fit = _fit_final_model(output_rows, tw.DEFAULT_C_GRID, tw.DEFAULT_MAX_ITER)
        if fit is None:
            report["outputs"][output_id] = {"fitted": False, "reason": "cv_unavailable"}
            continue

        coef = np.abs(fit["coef"])
        total = float(coef.sum()) or 1.0
        by_group = {}
        for g in ("market", "opponent", "own", "phase_misc"):
            mask = np.array([gr == g for gr in groups])
            by_group[g] = {
                "l1_mass": float(coef[mask].sum()),
                "l1_share": float(coef[mask].sum() / total),
                "max_abs_coef": float(coef[mask].max()) if mask.any() else 0.0,
            }
        ranked = sorted(zip(feature_order, fit["coef"]), key=lambda t: abs(t[1]), reverse=True)
        report["outputs"][output_id] = {
            "fitted": True,
            "C": fit["C"],
            "cv_mean_balanced_accuracy": fit["cv_mean_balanced_accuracy"],
            "n_fit_binary": fit["n_fit_binary"],
            "signed_class_counts": {str(k): v for k, v in signed.items()},
            "l1_total": total,
            "group_l1": by_group,
            "adaptive_l1_share": by_group["market"]["l1_share"] + by_group["opponent"]["l1_share"],
            "top_15_coefficients": [
                {"feature": f, "coef": float(c), "group": _group(f)} for f, c in ranked[:15]
            ],
            "market_coefficients": [
                {"feature": f, "coef": float(c)} for f, c in zip(feature_order, fit["coef"])
                if _group(f) == "market"
            ],
        }

    return report


def _render_md(report) -> str:
    lines = ["# Phase H / 5.2 -- SVM coefficient audit", ""]
    lines.append(f"Corpus: `{', '.join(report['corpus'])}`")
    lines.append(f"Schema hash: `{report['schema_hash']}`")
    lines.append("")
    lines.append("\"Adaptive L1 share\" = fraction of the standardized |coef| mass on the "
                 "shared-market (`price_ratio_*`, `market_displacement_*`) plus "
                 "opponent-state (`opponent_*`) features. Near zero => the model decides "
                 "from its own board, not the opponent or the market.")
    lines.append("")
    lines.append("| output | fitted | CV bacc | market share | opponent share | adaptive share |")
    lines.append("|---|---|---|---|---|---|")
    for oid, o in report["outputs"].items():
        if not o.get("fitted"):
            lines.append(f"| {oid} | no ({o['reason']}) | | | | |")
            continue
        g = o["group_l1"]
        lines.append(
            f"| {oid} | yes | {o['cv_mean_balanced_accuracy']:.3f} | "
            f"{g['market']['l1_share']:.1%} | {g['opponent']['l1_share']:.1%} | "
            f"{o['adaptive_l1_share']:.1%} |")
    lines.append("")
    for oid, o in report["outputs"].items():
        if not o.get("fitted"):
            continue
        lines.append(f"## {oid}")
        lines.append("")
        lines.append(f"C = {o['C']}, CV mean balanced accuracy = {o['cv_mean_balanced_accuracy']:.3f}, "
                     f"fit rows (binary) = {o['n_fit_binary']}, signed = {o['signed_class_counts']}")
        lines.append("")
        lines.append("Top coefficients by |value|:")
        lines.append("")
        lines.append("| feature | coef | group |")
        lines.append("|---|---|---|")
        for c in o["top_15_coefficients"]:
            lines.append(f"| {c['feature']} | {c['coef']:+.3f} | {c['group']} |")
        lines.append("")
        lines.append("Shared-market coefficients:")
        lines.append("")
        lines.append("| feature | coef |")
        lines.append("|---|---|")
        for c in o["market_coefficients"]:
            lines.append(f"| {c['feature']} | {c['coef']:+.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", nargs="+", default=["data/counterfactual_dataset.v6.jsonl"])
    p.add_argument("--out", default=str(ARTIFACTS / "phase_h_coefficient_audit"))
    args = p.parse_args(argv)

    report = audit(args.rows)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    out.with_suffix(".md").write_text(_render_md(report))
    print(_render_md(report))
    print(f"wrote {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
