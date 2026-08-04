"""Map a data/raw_videos/ filename back to the word(s) it shows, using the
WLASL/MS-ASL/ASL-Citizen annotation files the downloader/migration scripts
already read.

A raw file's name alone doesn't say what sign it is -- four different
filename schemes are mixed together in data/raw_videos/ (see the July 30
analysis): WLASL dictionary clips keyed by WLASL's own video_id, WLASL
YouTube-sourced clips keyed by the raw YouTube id, MS-ASL clips keyed by
"{youtube_id}_{start_time_ms}", and ASL-Citizen clips keyed by their own
"{hash}-{GLOSS}.mp4" filename (the CSV's "Video file" column, not a filename
parse -- ASL-Citizen glosses can contain a space the actual filename doesn't,
e.g. filename "...-SOCCER 2.mp4" vs CSV Gloss "SOCCER2"). A single WLASL
YouTube file can also carry more than one label: some are long source videos
where several different glosses each occupy their own frame range inside the
same file. lookup() always returns a list for that reason -- usually one
instance, sometimes many.

Labels are filtered (and, for synonyms, rewritten) against vocab.txt the same
way the downloaders decided what to fetch, so every label lookup() returns is
guaranteed to already be a spelling that exists in vocab.txt -- e.g. a clip
whose own JSON entry says "daddy" comes back labeled "father" if that's the
vocab.txt word its synonym group matched on. Without that, dataset.one_hot
would crash the moment it hit a synonym spelling vocab.txt doesn't have.
ASL-Citizen glosses go through the same canonical map after stripping a
trailing variant digit and lowercasing (e.g. "BACKPACK1" -> "backpack",
"ABOUT2" -> "about") -- ASL-Citizen numbers repeat *takes* of the same sign,
unlike WLASL/MS-ASL where no such suffix convention exists.
"""
import csv
import json
import re
from pathlib import Path

from dataset import load_vocab

HERE = Path(__file__).parent.parent
WLASL_JSON = HERE / "data" / "WLASL" / "start_kit" / "WLASL_v0.3.json"
MSASL_DIR = HERE / "data" / "MS-ASL"
MSASL_SPLIT_FILES = ("train", "val", "test")
SYNONYM_PATH = MSASL_DIR / "MSASL_synonym.json"
ASLCITIZEN_DIR = HERE / "data" / "ASL_Citizen"
ASLCITIZEN_SPLIT_FILES = ("train", "val", "test")
TRAILING_DIGIT_RE = re.compile(r'\d+$')

YT_ID_RE = re.compile(r'(?:[?&]v=|youtu\.be/)([\w-]{11})')


def _extract_yt_id(url: str) -> str | None:
    m = YT_ID_RE.search(url)
    return m.group(1) if m else None


def _load_vocab_with_synonyms() -> dict:
    """word -> canonical vocab.txt spelling, for every word vocab.txt's
    synonym groups make eligible. vocab.txt words map to themselves; a
    synonym maps to whichever vocab.txt word its group contains.
    """
    vocab = load_vocab()
    canonical = {w: w for w in vocab}
    vocab_set = set(vocab)
    for group in json.loads(SYNONYM_PATH.read_text(encoding="utf-8")):
        owners = [w for w in group if w in vocab_set]
        if not owners:
            continue
        for w in group:
            canonical.setdefault(w, owners[0])
    return canonical


def _load_wlasl_index(canonical: dict) -> tuple[dict, dict]:
    """video_id -> [instance...], yt_id -> [instance...]; instance = one gloss occurrence."""
    by_video_id, by_yt_id = {}, {}
    for entry in json.loads(WLASL_JSON.read_text(encoding="utf-8")):
        label = canonical.get(entry["gloss"])
        if label is None:
            continue
        for inst in entry["instances"]:
            record = {
                "label": label,
                "frame_start": inst["frame_start"],
                "frame_end": inst["frame_end"],  # -1 means "to the end of the file"
                "fps": inst["fps"],
                "signer_id": inst["signer_id"],
                "split": inst["split"],
                "source": "WLASL",
            }
            by_video_id.setdefault(inst["video_id"], []).append(record)
            yt_id = _extract_yt_id(inst["url"])
            if yt_id:
                by_yt_id.setdefault(yt_id, []).append(record)
    return by_video_id, by_yt_id


def _load_msasl_index(canonical: dict) -> dict:
    '''"{yt_id}_{start_time_ms}" -> [instance] (always exactly one -- MS-ASL clips are pre-trimmed).'''
    by_key = {}
    for split in MSASL_SPLIT_FILES:
        path = MSASL_DIR / "MSASL_{}.json".format(split)
        for e in json.loads(path.read_text(encoding="utf-8")):
            label = canonical.get(e["clean_text"])
            if label is None:
                continue
            url = e["url"] if e["url"].startswith("http") else "https://" + e["url"]
            yt_id = _extract_yt_id(url)
            if not yt_id:
                continue
            key = "{}_{}.mp4".format(yt_id, int(e["start_time"] * 1000))
            by_key[key] = [{
                "label": label,
                # e["start"]/e["end"] are frame numbers in the ORIGINAL,
                # untrimmed source video (end-start matches (end_time -
                # start_time) * fps exactly) -- meaningless as an offset into
                # this file. msasl_downloader.py already downloads just the
                # [start_time, end_time] slice via --download-sections, so
                # the saved file's own frames start at 0: the whole file is
                # the instance, same convention as a WLASL dictionary clip.
                "frame_start": 0,
                "frame_end": -1,
                "fps": e["fps"],
                "signer_id": e.get("signer_id"),
                "split": split,
                "source": "MS-ASL",
            }]
    return by_key


def _load_aslcitizen_index(canonical: dict) -> dict:
    '''"{hash}-{GLOSS}.mp4" (the CSV's own "Video file" column) -> [instance]
    (always exactly one -- ASL-Citizen clips are pre-trimmed to just the sign).
    '''
    by_key = {}
    for split in ASLCITIZEN_SPLIT_FILES:
        path = ASLCITIZEN_DIR / "splits" / "{}.csv".format(split)
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gloss = TRAILING_DIGIT_RE.sub("", row["Gloss"]).strip().lower()
                label = canonical.get(gloss)
                if label is None:
                    continue
                by_key[row["Video file"]] = [{
                    "label": label,
                    "frame_start": 0,
                    "frame_end": -1,
                    "fps": None,
                    "signer_id": row["Participant ID"],
                    "split": split,
                    "source": "ASL_Citizen",
                }]
    return by_key


_index = None  # (by_video_id, by_yt_id, by_msasl_key, by_aslcitizen_key), built lazily on first lookup()


def _ensure_index():
    global _index
    if _index is None:
        canonical = _load_vocab_with_synonyms()
        by_video_id, by_yt_id = _load_wlasl_index(canonical)
        by_msasl_key = _load_msasl_index(canonical)
        by_aslcitizen_key = _load_aslcitizen_index(canonical)
        _index = (by_video_id, by_yt_id, by_msasl_key, by_aslcitizen_key)
    return _index


def lookup(filename: str) -> list[dict]:
    """filename is a bare name from data/raw_videos/ (e.g. "69241.mp4",
    "1_jXxjd0lKE.mp4", "-0GelCb31wA_0.mp4", "15890366051589533-APPLE.mp4").
    Returns every labeled instance that file contains -- usually one,
    sometimes many for a shared WLASL YouTube source video. Empty list for
    anything not a recognized clip (leftover .f137.mp4/.m4a partial-stream
    artifacts, or a word that isn't in vocab.txt).
    """
    if not filename.endswith(".mp4"):
        return []
    by_video_id, by_yt_id, by_msasl_key, by_aslcitizen_key = _ensure_index()
    stem = filename[:-4]
    if stem in by_video_id:
        return by_video_id[stem]
    if stem in by_yt_id:
        return by_yt_id[stem]
    if filename in by_msasl_key:
        return by_msasl_key[filename]
    return by_aslcitizen_key.get(filename, [])


def _self_check():
    vocab = set(load_vocab())

    assert lookup("not_a_real_file_12345.mp4") == []
    assert lookup("some_video.f137.mp4") == []  # partial-stream artifact, not a real clip

    book = lookup("69241.mp4")  # WLASL dictionary clip, aslbricks.org/.../book.mp4
    assert len(book) == 1
    assert book[0]["label"] == "book" and book[0]["source"] == "WLASL"
    assert book[0]["frame_end"] == -1  # whole file is the sign, no trim needed

    # WLASL YouTube source video known to carry many different glosses at
    # different frame ranges -- every one of them must come back labeled
    # with a real vocab.txt word (proves the synonym-canonicalization and
    # vocab filtering actually ran, not just glosses copied verbatim).
    shared = lookup("1_jXxjd0lKE.mp4")
    assert len(shared) >= 1
    for inst in shared:
        assert inst["label"] in vocab
        assert inst["source"] == "WLASL"
        assert inst["frame_end"] == -1 or inst["frame_start"] <= inst["frame_end"]

    match = lookup("-0GelCb31wA_0.mp4")  # MS-ASL clip, pre-trimmed
    assert len(match) == 1
    assert match[0]["source"] == "MS-ASL"
    assert match[0]["label"] in vocab
    # the downloaded file already IS the [start_time, end_time] slice --
    # whole file is the instance, not a further trim by e["start"]/e["end"]
    # (those are frame numbers in the original untrimmed source video).
    assert match[0]["frame_start"] == 0 and match[0]["frame_end"] == -1

    aslcitizen = lookup("15890366051589533-APPLE.mp4")  # ASL-Citizen clip, pre-trimmed
    assert len(aslcitizen) == 1
    assert aslcitizen[0]["source"] == "ASL_Citizen" and aslcitizen[0]["label"] == "apple"
    assert aslcitizen[0]["frame_start"] == 0 and aslcitizen[0]["frame_end"] == -1
    assert aslcitizen[0]["signer_id"] == "P1"

    # ASL-Citizen's own trailing-digit variant-take suffix ("BACKPACK1")
    # must canonicalize down to the plain vocab.txt spelling ("backpack"),
    # not get rejected as an unrecognized word.
    variant = lookup("4520498201410337-BACKPACK.mp4")
    assert len(variant) == 1 and variant[0]["label"] == "backpack"

    print("self-check passed: all four raw_videos filename schemes resolve to "
          "vocab.txt-canonical labels, multi-label WLASL source videos return every instance, "
          "and ASL-Citizen's trailing-digit variant takes canonicalize to the plain vocab spelling")


if __name__ == "__main__":
    _self_check()
