"""Split data/raw_landmarks/ into train/val/test, grouped by signer.

Splitting per word alone isn't enough: one signer contributes many clips
across many different words. A per-word file shuffle scatters those clips
across splits with no regard for that, so the same signer's hand geometry,
background, and camera setup end up on both sides of the train/test boundary
-- a model can partly learn to recognize the *signer* instead of the *sign*,
and val/test accuracy overstates how well it'll generalize to someone new.

Every clip is grouped by participant_id, which extract_kaggle_landmarks.py
already bakes into the filename (Kaggle's train.csv tags every sequence with
its own participant_id directly -- no separate metadata lookup needed to
recover it). Whole signer groups, never individual clips, are then handed to
one split, so no signer's clips ever appear in more than one of
train/val/test.

ponytail: exact per-word 60/20/20 balance isn't possible once splitting is
signer-disjoint -- a word signed by only one or two people necessarily lands
wherever their signer group lands, sometimes entirely out of val or test.
The old per-file shuffle faked a clean per-word ratio by ignoring this.
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw_landmarks"
SPLIT_PATH = Path(__file__).parent.parent / "data" / "split.json"
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
SEED = 42

# extract_kaggle_landmarks.py's naming: "{sign}__{participant_id}-{sequence_id}_landmarks.json"
FILENAME_RE = re.compile(r"^(.+?)__(\d+)-(\d+)_landmarks\.json$")


def _signer_key(filename: str) -> tuple:
    """landmarks filename -> ("Kaggle", "signer", participant_id)."""
    m = FILENAME_RE.match(filename)
    participant_id = m.group(2)
    return ("Kaggle", "signer", participant_id)


def _split_groups(groups: dict[tuple, list[str]], train_frac: float = TRAIN_FRAC,
                   val_frac: float = VAL_FRAC, seed: int = SEED) -> dict[str, str]:
    """group key -> list of filenames, all assigned to the SAME split -> filename -> split.

    Groups (not individual files) are hard constraints: every file in a
    group must land together. Whole groups are handed out via greedy
    proportional fill -- shuffle the groups, then drop each one into
    whichever split is currently furthest below its target share of files
    seen so far. That converges close to train_frac/val_frac/test_frac
    overall without ever splitting a group, which an exact bin-packing
    wouldn't buy much over for this few thousand groups.
    """
    rng = random.Random(seed)
    keys = sorted(groups)  # fixed order before shuffling, reproducible under a given seed
    rng.shuffle(keys)

    targets = {"train": train_frac, "val": val_frac, "test": 1 - train_frac - val_frac}
    counts = {"train": 0, "val": 0, "test": 0}
    split = {}
    for key in keys:
        files = groups[key]
        total = sum(counts.values()) or 1
        chosen = max(counts, key=lambda s: targets[s] - counts[s] / total)
        for f in files:
            split[f] = chosen
        counts[chosen] += len(files)
    return split


def stratified_split(filenames: list[str], train_frac: float = TRAIN_FRAC,
                      val_frac: float = VAL_FRAC, seed: int = SEED) -> dict[str, str]:
    """filename -> "train"/"val"/"test", grouped so every signer's clips land in one split."""
    groups = defaultdict(list)
    for f in filenames:
        groups[_signer_key(f)].append(f)
    return _split_groups(groups, train_frac, val_frac, seed)


def _self_check():
    from collections import Counter

    # --- group-allocation logic in isolation: synthetic groups, no real
    # metadata lookup, just checking the allocator itself. ---
    groups = {("src", "signer", i): [f"w{i}_a", f"w{i}_b", f"w{i}_c"] for i in range(30)}
    split = _split_groups(groups, seed=0)
    assert set(split) == {f for files in groups.values() for f in files}

    for files in groups.values():  # every group's files land in ONE split -- the whole point
        assert len({split[f] for f in files}) == 1

    counts = Counter(split.values())
    total = sum(counts.values())
    assert 0.5 < counts["train"] / total < 0.7
    assert 0.1 < counts["val"] / total < 0.3
    assert 0.1 < counts["test"] / total < 0.3
    assert _split_groups(groups, seed=0) == split  # reproducible under a fixed seed

    # a handful of large, uneven groups should still land close to target,
    # same as the many-small-groups case above -- greedy fill shouldn't
    # depend on groups being roughly equal-sized.
    uneven = {("src", "signer", i): [f"g{i}_{j}" for j in range(size)]
              for i, size in enumerate([50, 40, 30, 20, 15, 10, 8, 6, 4, 2] * 3)}
    uneven_split = _split_groups(uneven, seed=1)
    for files in uneven.values():
        assert len({uneven_split[f] for f in files}) == 1
    uneven_counts = Counter(uneven_split.values())
    uneven_total = sum(uneven_counts.values())
    assert 0.45 < uneven_counts["train"] / uneven_total < 0.75

    # --- _signer_key against extract_kaggle_landmarks.py's filename convention. ---
    assert _signer_key("book__16069-100015657_landmarks.json") == ("Kaggle", "signer", "16069")

    # two clips from the same participant (different signs/sequences) must
    # share a signer key -- that's what keeps a signer's clips together.
    key_a = _signer_key("book__16069-100015657_landmarks.json")
    key_b = _signer_key("cloud__16069-1002113535_landmarks.json")
    assert key_a == key_b

    # a different participant must get a different key.
    key_c = _signer_key("book__28656-1000106739_landmarks.json")
    assert key_c != key_a

    print("self-check passed: signer groups never get split across train/val/test, "
          "greedy fill lands close to target ratios for both even and uneven group sizes, "
          "and signer_key parses participant_id straight out of the Kaggle filename convention")


if __name__ == "__main__":
    import sys
    if "--self-test" in sys.argv:
        _self_check()
        sys.exit(0)

    filenames = sorted(f.name for f in DATA_DIR.glob("*_landmarks.json"))
    split = stratified_split(filenames)
    SPLIT_PATH.write_text(json.dumps(split, indent=2))

    counts = defaultdict(int)
    for s in split.values():
        counts[s] += 1
    print(f"{len(filenames)} files -> train={counts['train']} val={counts['val']} test={counts['test']}")

    # diagnostic: signer-disjoint splitting can leave a rare word with zero
    # examples in val or test (see the ponytail note up top) -- surface how
    # often that actually happens on this corpus instead of hiding it.
    by_word = defaultdict(set)
    for f, s in split.items():
        word = f.split("__", 1)[0]
        by_word[word].add(s)
    starved = {w: present for w, present in by_word.items() if len(present) < 3}
    print(f"{len(by_word)} words total, {len(starved)} missing from at least one split "
          f"(unavoidable when a word's clips all belong to the same signer group)")
    print(f"wrote {SPLIT_PATH}")
