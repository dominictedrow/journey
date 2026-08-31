"""Run targeted counterfactual generation until buffered SVM quotas are met."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import watcher_config as wc


PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL")
OUTPUTS = tuple(
    f"product:{product}:{direction}"
    for product in PRODUCTS
    for direction in ("increase", "avoid")
) + ("land_expansion", "labor_expansion")
FAMILIES = (
    "strategy_follower",
    "strategies_v2.ring.agent",
    "strategies_v2.leader_clone.agent",
)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def counts(rows: list[dict], output_id: str) -> tuple[int, Counter, int]:
    attempted = [row for row in rows if row.get("output_id") == output_id]
    accepted = [row for row in attempted if row.get("accepted")]
    return len(accepted), Counter(row.get("label") for row in accepted), len(attempted)


def family_counts(rows: list[dict], output_id: str) -> tuple[Counter, Counter]:
    """Per-family attempt and accepted counts for one output_id."""
    attempts: Counter = Counter()
    accepted: Counter = Counter()
    for row in rows:
        if row.get("output_id") != output_id:
            continue
        family = row.get("strategy_family")
        attempts[family] += 1
        if row.get("accepted"):
            accepted[family] += 1
    return attempts, accepted


def dead_families(rows: list[dict], output_id: str, dead_min_attempts: int) -> set[str]:
    """Families already proven unable to produce an accepted row for this output.

    A family that has been attempted at least `dead_min_attempts` times for this
    output with zero acceptances is not going to start working with more of the
    same attempts -- the harness is verifying that the family's scripted policy
    never lets this specific intervention register as having changed anything.
    """
    attempts, accepted = family_counts(rows, output_id)
    return {family for family in FAMILIES if attempts[family] >= dead_min_attempts and accepted[family] == 0}


def choose_family(rows: list[dict], output_id: str, dead_min_attempts: int) -> str | None:
    """Least-attempted family for this output that isn't already proven dead.

    Returns None when every family is dead, meaning this output cannot progress
    (and, if it only ever had one live family, can never reach the two-usable-
    family requirement) regardless of how many more pairs are generated.
    """
    dead = dead_families(rows, output_id, dead_min_attempts)
    live = [family for family in FAMILIES if family not in dead]
    if not live:
        return None
    attempts, _ = family_counts(rows, output_id)
    return min(live, key=lambda item: (attempts[item], FAMILIES.index(item)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=200000)
    parser.add_argument("--min-accepted", type=int, default=wc.MIN_ACCEPTED)
    parser.add_argument("--min-signed", type=int, default=wc.MIN_SIGNED)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--dead-min-attempts", type=int, default=15,
                         help="attempts at zero acceptance before a family is treated as dead for an output")
    args = parser.parse_args()

    seed_base = args.seed_base
    blocked: dict[str, str] = {}
    while True:
        rows = load_rows(args.output)
        pending = []
        for output_id in OUTPUTS:
            accepted, labels, attempted = counts(rows, output_id)
            if accepted < args.min_accepted or labels[-1] < args.min_signed or labels[1] < args.min_signed:
                if attempted >= args.max_attempts:
                    blocked.setdefault(output_id, "attempt-cap")
                    continue
                family = choose_family(rows, output_id, args.dead_min_attempts)
                if family is None:
                    blocked.setdefault(output_id, "all-families-dead")
                    continue
                deficit = max(args.min_accepted - accepted, args.min_signed - labels[-1], args.min_signed - labels[1])
                pending.append((deficit, output_id, accepted, labels, attempted, family))
        if not pending:
            break

        # Work on the largest remaining quota deficit first.  Choose the least-used
        # family for that output so leave-one-family-out diagnostics get real data,
        # skipping any family already shown dead for this specific output.
        _, output_id, accepted, labels, attempted, family = max(pending)
        batch = min(args.batch_size, args.max_attempts - attempted)
        print(
            f"TARGET {output_id}: accepted={accepted}, neg={labels[-1]}, "
            f"pos={labels[1]}, attempts={attempted}; next={batch} family={family}",
            flush=True,
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("counterfactual_dataset.py")),
            "--output", str(args.output),
            "--append",
            "--pairs", str(batch),
            "--seed-base", str(seed_base),
            "--workers", str(args.workers),
            "--target-output", output_id,
            "--strategy-family", family,
        ]
        subprocess.run(command, check=True)
        seed_base += batch

    rows = load_rows(args.output)
    print("FINAL QUOTAS", flush=True)
    for output_id in OUTPUTS:
        accepted, labels, attempted = counts(rows, output_id)
        if accepted >= args.min_accepted and labels[-1] >= args.min_signed and labels[1] >= args.min_signed:
            status = "met"
        else:
            status = blocked.get(output_id, "attempt-cap")
        print(f"{output_id}: {status}; accepted={accepted}, neg={labels[-1]}, pos={labels[1]}, attempts={attempted}")
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
