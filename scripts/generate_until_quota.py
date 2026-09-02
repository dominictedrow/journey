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

# Phase 5.2: the market-glutting opponents replace ring/leader_clone as the
# *opponent* set (the focal families above are unchanged). The scripted
# opponents never over-expose melon/milk/wool, so removing that exposure
# never helps near-term money -> product:*:avoid can't produce its rarer
# sign. The glutters put the shared-market price-collapse regime
# (MD files/CLAUDE.md "Why the watcher matters") in-distribution.
DEFAULT_OPPONENTS = (
    "starter",
    "strategies_v2.glut_melon.agent",
    "strategies_v2.glut_livestock.agent",
    "strategies_v2.glut_mixed.agent",
)


PROBE_CACHE_SEED = 0


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_probe_cache(path: Path, families, opponents, episode_steps: int) -> None:
    """One probe replay per (family, opponent) -> sidecar of emission days and
    opponent-volume days (repair.md Phase F). The focal families are
    opponent-blind, so this is reused across every seed instead of re-run per
    pair -- roughly a third fewer replays in the Phase G regeneration."""
    import counterfactual_dataset as cd

    entries: dict[str, dict] = {}
    for family in families:
        base = cd.resolve_opponent(family)
        for opponent_name in opponents:
            probe = cd._run(base, cd.resolve_opponent(opponent_name), 0,
                            PROBE_CACHE_SEED, episode_steps, episode_steps + 1)
            entries[f"{family}|{opponent_name}"] = cd._probe_summary(probe[1], 0, probe[4])
            print(f"  probe {family} vs {opponent_name}", flush=True)
    payload = {"opponent_volume_units_per_day": wc.OPPONENT_VOLUME_UNITS_PER_DAY,
               "episode_steps": episode_steps, "entries": entries}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def signed_quota_unreachable(rows: list[dict], output_id: str, min_attempts: int,
                             max_attempts: int = 2000, min_signed: int = wc.MIN_SIGNED,
                             projection_floor: int = 30) -> bool:
    """True when an output cannot plausibly reach ``min_signed`` of *both* signs
    within ``max_attempts``, so every further pair is wasted.

    This is the low-value ``product:*:avoid`` / cheap-crop failure mode: the
    one-shot bounded nudge never moves near-term money past the label band, so
    accepted rows pile up neutral (or one sign never appears) regardless of
    focal family or attempt count. The rarer sign's arrival rate is projected
    forward over the remaining attempt budget; if the projected final count
    falls short of ``projection_floor`` the output is abandoned. The
    ``min_attempts`` floor keeps a genuinely slow-starting output alive.
    """
    accepted, labels, attempted = counts(rows, output_id)
    if attempted < min_attempts or accepted < 60:
        return False
    rarer = min(labels[-1], labels[1])
    projected = rarer * max_attempts / attempted
    return projected < projection_floor and projection_floor <= min_signed


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
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS),
                         help="opponent agents each pair is run against (round-robined)")
    parser.add_argument("--outputs", nargs="+", default=None,
                         help="restrict generation to these output_ids (default: all). "
                              "Phase 5.2 skips product:{MELON,MILK}:avoid -- the small "
                              "remove-two deviation can't move the label even under a glut.")
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--probe-cache", type=Path, default=None,
                         help="path to the Phase F per-(family, opponent) probe sidecar; "
                              "built here if missing, then passed through to every generation batch")
    parser.add_argument("--dead-min-attempts", type=int, default=15,
                         help="attempts at zero acceptance before a family is treated as dead for an output")
    parser.add_argument("--unsigned-dead-attempts", type=int, default=400,
                         help="attempt floor before an output whose rarer signed class is projected "
                              "to fall short of the per-sign quota is abandoned (e.g. low-value "
                              "product:*:avoid, whose one-shot nudge never clears the label band)")
    args = parser.parse_args()

    seed_base = args.seed_base
    target_outputs = tuple(args.outputs) if args.outputs else OUTPUTS
    unknown = set(target_outputs) - set(OUTPUTS)
    if unknown:
        parser.error(f"unknown output_id(s): {sorted(unknown)}")
    if args.probe_cache and not args.probe_cache.exists():
        print(f"building probe cache -> {args.probe_cache}", flush=True)
        build_probe_cache(args.probe_cache, FAMILIES, args.opponents, args.episode_steps)

    blocked: dict[str, str] = {}
    while True:
        rows = load_rows(args.output)
        pending = []
        for output_id in target_outputs:
            accepted, labels, attempted = counts(rows, output_id)
            if accepted < args.min_accepted or labels[-1] < args.min_signed or labels[1] < args.min_signed:
                if attempted >= args.max_attempts:
                    blocked.setdefault(output_id, "attempt-cap")
                    continue
                if signed_quota_unreachable(rows, output_id, args.unsigned_dead_attempts,
                                            args.max_attempts, args.min_signed):
                    blocked.setdefault(output_id, "signed-quota-unreachable")
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
            "--episode-steps", str(args.episode_steps),
            "--opponents", *args.opponents,
        ]
        if args.probe_cache:
            command += ["--probe-cache", str(args.probe_cache)]
        subprocess.run(command, check=True)
        seed_base += batch

    rows = load_rows(args.output)
    print("FINAL QUOTAS", flush=True)
    for output_id in target_outputs:
        accepted, labels, attempted = counts(rows, output_id)
        if accepted >= args.min_accepted and labels[-1] >= args.min_signed and labels[1] >= args.min_signed:
            status = "met"
        else:
            status = blocked.get(output_id, "attempt-cap")
        print(f"{output_id}: {status}; accepted={accepted}, neg={labels[-1]}, pos={labels[1]}, attempts={attempted}")
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
