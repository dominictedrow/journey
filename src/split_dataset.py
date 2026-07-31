"""Split data/raw_landmarks/ into train/val/test, grouped by signer.

Splitting per word alone isn't enough: many WLASL/MS-ASL clips share a
source recording -- one YouTube video where a single signer performs dozens
of different words in one sitting. A per-word file shuffle scatters those
clips across splits with no regard for that, so the same signer's hand
geometry, background, and camera setup end up on both sides of the
train/test boundary -- a model can partly learn to recognize the *signer*
instead of the *sign*, and val/test accuracy overstates how well it'll
generalize to someone new.

Every clip is grouped by signer_id (namespaced by source -- WLASL and
MS-ASL each start their own signer_id numbering at 0, so signer 0 in one
isn't signer 0 in the other) via label_lookup, which already resolves a
clip back to the WLASL/MS-ASL metadata that recorded it. Whole signer
groups, never individual clips, are then handed to one split, so no
signer's clips ever appear in more than one of train/val/test. Where
signer_id isn't recorded (doesn't happen in the current corpus, but
label_lookup treats it as optional), the clip's source video is used as the
grouping key instead -- weaker (two different videos of the same signer
could still land in different splits) but still keeps a single recording
session from being split apart.

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

from label_lookup import lookup

DATA_DIR = Path(__file__).parent.parent / "data" / "raw_landmarks"
SPLIT_PATH = Path(__file__).parent.parent / "data" / "split.json"
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
SEED = 42

# extract_raw_videos.py's naming: "{label}__{video_id}_landmarks.json", where
# video_id is "{source_video_stem}__{instance_index}" for a multi-label
# source video, or just the bare source_video_stem for a single-label one.
FILENAME_RE = re.compile(r"^(.+?)__(.+)_landmarks\.json$")
MULTI_INSTANCE_RE = re.compile(r"^(.+)__(\d+)$")


def _signer_key(filename: str) -> tuple:
    """landmarks filename -> (source, "signer", signer_id), or a same-source-
    video fallback key when signer_id is unavailable. Also double-checks the
    label recovered from label_lookup against the one in the filename itself
    -- if those ever disagree, the video_id/instance-index parsing above has
    drifted out of sync with extract_raw_videos.py's naming and the split
    would silently group the wrong clips.
    """
    m = FILENAME_RE.match(filename)
    label, video_id = m.group(1), m.group(2)

    multi = MULTI_INSTANCE_RE.match(video_id)
    raw_stem, index = (multi.group(1), int(multi.group(2))) if multi else (video_id, 0)

    inst = lookup(raw_stem + ".mp4")[index]
    assert inst["label"] == label, (
        f"{filename}: filename says label={label!r}, but label_lookup resolved "
        f"instance {index} of {raw_stem!r} to {inst['label']!r}"
    )

    if inst["signer_id"] is not None:
        return (inst["source"], "signer", inst["signer_id"])
    return (inst["source"], "video", raw_stem)


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

    # --- _signer_key against real WLASL/MS-ASL metadata (same clips
    # label_lookup._self_check already exercises), proving the
    # filename<->lookup mapping actually lines up. ---
    book_key = _signer_key("book__69241_landmarks.json")
    assert book_key[:2] == ("WLASL", "signer")

    # a real WLASL source video known (see label_lookup._self_check) to carry
    # more than one gloss -- each instance must resolve to whatever signer_id
    # label_lookup itself reports for that instance, not silently swapped.
    shared = lookup("1_jXxjd0lKE.mp4")
    assert len(shared) >= 2
    for i, inst in enumerate(shared):
        fname = "{}__1_jXxjd0lKE__{}_landmarks.json".format(inst["label"], i)
        assert _signer_key(fname) == ("WLASL", "signer", inst["signer_id"])

    # a mismatched label must be caught, not silently mis-grouped
    try:
        _signer_key("not_the_real_label__69241_landmarks.json")
        assert False, "expected an AssertionError on label mismatch"
    except AssertionError as e:
        assert "filename says label" in str(e)

    print("self-check passed: signer groups never get split across train/val/test, "
          "greedy fill lands close to target ratios for both even and uneven group sizes, "
          "and signer_key's filename<->metadata mapping matches real WLASL data")


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
