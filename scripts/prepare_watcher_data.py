"""Add auditable provenance, seed-safe splits, and explicit weights to JSONL rows."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from counterfactual_dataset import split_for_seed_group


def enrich_row(row, strategy_family, sample_weight):
    result = dict(row)
    result["strategy_family"] = result.get("strategy_family") or strategy_family
    result["split"] = split_for_seed_group(result["seed_group"])
    result["sample_weight"] = float(result.get("sample_weight", sample_weight))
    return result


def validate(rows):
    seed_splits = defaultdict(set)
    counts = defaultdict(Counter)
    errors = []
    for index, row in enumerate(rows, 1):
        seed_splits[row.get("seed_group")].add(row.get("split"))
        weight = row.get("sample_weight")
        if (isinstance(weight, bool) or not isinstance(weight, (int, float)) or
                not math.isfinite(weight) or weight <= 0):
            errors.append(f"row {index}: invalid sample_weight")
        if row.get("accepted"):
            counts[row.get("output_id")][row.get("label")] += 1
    for seed, splits in seed_splits.items():
        if seed is None or len(splits) != 1:
            errors.append(f"seed_group {seed!r}: split leakage {sorted(splits)}")
    return errors, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy-family", default="strategy_follower")
    parser.add_argument("--sample-weight", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.sample_weight) or args.sample_weight <= 0:
        parser.error("--sample-weight must be finite and positive")

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    rows = [enrich_row(row, args.strategy_family, args.sample_weight) for row in rows]
    errors, counts = validate(rows)
    if errors:
        raise SystemExit("\n".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"wrote {len(rows)} enriched rows to {args.output}")
    for output_id in sorted(counts):
        print(output_id, dict(sorted(counts[output_id].items())))


if __name__ == "__main__":
    main()
