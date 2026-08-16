"""Load MediaPipe hand-landmark JSON files (written by process_video.py) into
records ready for the rest of the ingestion pipeline. This is the connection
point to the MediaPipe output -- no encoding, scaling, or batching happens
here, that's separate steps.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_LANDMARKS_DIR = Path(__file__).parent.parent / "data" / "raw_landmarks"
SPLIT_PATH = Path(__file__).parent.parent / "data" / "split.json"


def load_video(path: Path, label: str) -> list[dict]:
    """Read one *_landmarks.json file and tag each detection with its label.

    Two record shapes exist depending on who wrote the file:
    - process_video.py (personal recordings in data/) writes bare detections
      with no "video"/"label" -- those get inferred from the filename, one
      video/word per file (e.g. "hello_landmarks.json" -> "hello").
    - extract_kaggle_landmarks.py (the bulk Kaggle asl-signs corpus in
      data/raw_landmarks/) already tags every detection with its own "video"
      and "label", since a source file's own name doesn't encode the word
      the way a personal recording's does. Those pass through unchanged.
    Returns one dict per detected body part per frame: video, label, frame, part
    ("right_hand"/"left_hand"/"face"/"pose", or None if nothing was detected
    that frame -- see dataset.fuse_landmarks), landmarks, plus timestamp_ms
    when the source is a live MediaPipe run (process_video.py/mac.py).
    """
    records = []
    for detection in json.loads(path.read_text()):
        if "label" in detection:
            records.append(detection)
        else:
            records.append({"video": label, "label": label, **detection})
    return records


def load_records(data_dir: Path = DATA_DIR, split: str | None = None) -> list[dict]:
    """Load every *_landmarks.json file in data_dir and tag each detection with its label.

    split: "train"/"val"/"test" to load only the files split_dataset.py
    assigned to that split (data/split.json). Files with no entry there
    (e.g. personal recordings, which aren't part of the split) are skipped
    whenever a split filter is given.

    Eagerly reads and json.loads's every matching file into memory -- fine
    for a small corpus, but not the full Kaggle-scale one. For that, use
    list_video_files + dataset.LazySignVideoDataset instead, which reads and
    fuses one video at a time.
    """
    split_map = json.loads(SPLIT_PATH.read_text()) if split else {}
    records = []
    for path in sorted(Path(data_dir).glob("*_landmarks.json")):
        if split and split_map.get(path.name) != split:
            continue
        label = path.stem.removesuffix("_landmarks").split("__", 1)[0]
        records.extend(load_video(path, label))
    return records


def list_video_files(data_dir: Path = DATA_DIR, split: str | None = None) -> list[tuple[Path, str]]:
    """Same file discovery + split filtering as load_records, but returns
    (path, label) pairs without reading any file contents -- no
    path.read_text() or json.loads() here, that's the whole point. Lets a
    caller build a per-video lazy Dataset (dataset.LazySignVideoDataset)
    whose memory use doesn't scale with corpus size.
    """
    split_map = json.loads(SPLIT_PATH.read_text()) if split else {}
    files = []
    for path in sorted(Path(data_dir).glob("*_landmarks.json")):
        if split and split_map.get(path.name) != split:
            continue
        label = path.stem.removesuffix("_landmarks").split("__", 1)[0]
        files.append((path, label))
    return files


def cache_path_for(path: Path) -> Path:
    """A *_landmarks.json file's sibling .npy cache path (same directory,
    same stem) -- shared by build_landmark_cache.py (writer) and
    dataset.LazySignVideoDataset (reader) so the two never disagree on where
    a video's cached array lives."""
    return path.with_suffix(".npy")


def _self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sample = [{
            "frame": 0,
            "timestamp_ms": 0,
            "part": "right_hand",
            "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.0}] * 21,
        }]
        (tmp_path / "hello_landmarks.json").write_text(json.dumps(sample))

        records = load_records(tmp_path)

        assert len(records) == 1
        assert records[0]["video"] == "hello"
        assert records[0]["label"] == "hello"
        assert records[0]["part"] == "right_hand"
        assert len(records[0]["landmarks"]) == 21

    # raw_landmarks-style files: detections already carry "video"/"label"
    # (extract_kaggle_landmarks.py tags them) -- must pass through as-is, not
    # get re-tagged from the filename (which would wreck a name like
    # "about__16069-100015657_landmarks.json", where the label is only the
    # part before the first "__").
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tagged = [{"video": "about__abc123__0", "label": "about", "frame": 0,
                   "handedness": "Left", "landmarks": []}]
        (tmp_path / "about__abc123__0_landmarks.json").write_text(json.dumps(tagged))

        records = load_records(tmp_path)
        assert len(records) == 1
        assert records[0]["video"] == "about__abc123__0"
        assert records[0]["label"] == "about"

    # list_video_files must never touch file contents -- garbage JSON that
    # would blow up json.loads has to still yield the correct (path, label)
    # pair, for both filename conventions (personal-recording: whole stem is
    # the label; Kaggle: label is only the part before the first "__").
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "hello_landmarks.json").write_text("not valid json{{{")
        (tmp_path / "book__1-1_landmarks.json").write_text("also not valid json")

        files = dict(list_video_files(tmp_path))
        assert files[tmp_path / "hello_landmarks.json"] == "hello"
        assert files[tmp_path / "book__1-1_landmarks.json"] == "book"
        assert cache_path_for(tmp_path / "hello_landmarks.json") == tmp_path / "hello_landmarks.npy"

    # split filtering: only files data/split.json assigns to the requested
    # split load, the rest (including files split.json doesn't mention at
    # all) are skipped. list_video_files must filter identically.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for name, video in [("cat__1_landmarks.json", "cat__1"), ("cat__2_landmarks.json", "cat__2"),
                             ("dog__1_landmarks.json", "dog__1")]:
            (tmp_path / name).write_text(json.dumps(
                [{"video": video, "label": video.split("__")[0], "frame": 0, "handedness": None, "landmarks": []}]))

        global SPLIT_PATH
        original_split_path = SPLIT_PATH
        SPLIT_PATH = tmp_path / "split.json"
        SPLIT_PATH.write_text(json.dumps({
            "cat__1_landmarks.json": "train",
            "cat__2_landmarks.json": "test",
            # "dog__1_landmarks.json" deliberately absent -- must be skipped, not default to some split
        }))
        try:
            train_records = load_records(tmp_path, split="train")
            assert [r["video"] for r in train_records] == ["cat__1"]
            test_records = load_records(tmp_path, split="test")
            assert [r["video"] for r in test_records] == ["cat__2"]
            assert load_records(tmp_path) != train_records  # no split filter -> everything, unaffected by split.json

            assert list_video_files(tmp_path, split="train") == [(tmp_path / "cat__1_landmarks.json", "cat")]
            assert list_video_files(tmp_path, split="test") == [(tmp_path / "cat__2_landmarks.json", "cat")]
            assert len(list_video_files(tmp_path)) == 3  # no split filter -> everything
        finally:
            SPLIT_PATH = original_split_path

    print("self-check passed: records load with correct label and shape, raw_landmarks "
          "pre-tagged detections pass through unchanged, list_video_files matches load_records' "
          "label/split logic without ever reading file contents, and split filtering only loads "
          "files split.json assigns to the requested split")


if __name__ == "__main__":
    _self_check()
