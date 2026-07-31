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


def load_records(data_dir: Path = DATA_DIR, split: str | None = None) -> list[dict]:
    """Load every *_landmarks.json file in data_dir and tag each detection with its label.

    Two record shapes exist depending on who wrote the file:
    - process_video.py (personal recordings in data/) writes bare detections
      with no "video"/"label" -- those get inferred from the filename, one
      video/word per file (e.g. "hello_landmarks.json" -> "hello").
    - extract_raw_videos.py (the bulk WLASL/MS-ASL corpus in
      data/raw_landmarks/) already tags every detection with its own "video"
      and "label", since a source file's own name doesn't encode the word
      the way a personal recording's does. Those pass through unchanged.
    Returns one dict per detected hand per frame: video, label, frame, timestamp_ms, handedness, landmarks, world_landmarks.

    split: "train"/"val"/"test" to load only the files split_dataset.py
    assigned to that split (data/split.json). Files with no entry there
    (e.g. personal recordings, which aren't part of the split) are skipped
    whenever a split filter is given.
    """
    split_map = json.loads(SPLIT_PATH.read_text()) if split else {}
    records = []
    for path in sorted(Path(data_dir).glob("*_landmarks.json")):
        if split and split_map.get(path.name) != split:
            continue
        label = path.stem.removesuffix("_landmarks")
        for detection in json.loads(path.read_text()):
            if "label" in detection:
                records.append(detection)
            else:
                records.append({"video": label, "label": label, **detection})
    return records


def _self_check():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        sample = [{
            "frame": 0,
            "timestamp_ms": 0,
            "handedness": "Right",
            "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.0}] * 21,
            "world_landmarks": [{"x": 0.01, "y": 0.02, "z": 0.0}] * 21,
        }]
        (tmp_path / "hello_landmarks.json").write_text(json.dumps(sample))

        records = load_records(tmp_path)

        assert len(records) == 1
        assert records[0]["video"] == "hello"
        assert records[0]["label"] == "hello"
        assert records[0]["handedness"] == "Right"
        assert len(records[0]["landmarks"]) == 21

    # raw_landmarks-style files: detections already carry "video"/"label"
    # (extract_raw_videos.py tags them) -- must pass through as-is, not get
    # re-tagged from the filename (which would wreck a multi-instance name
    # like "about__jPn0FQxkdBc__5_landmarks.json").
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tagged = [{"video": "about__abc123__0", "label": "about", "frame": 0,
                   "handedness": "Left", "landmarks": []}]
        (tmp_path / "about__abc123__0_landmarks.json").write_text(json.dumps(tagged))

        records = load_records(tmp_path)
        assert len(records) == 1
        assert records[0]["video"] == "about__abc123__0"
        assert records[0]["label"] == "about"

    # split filtering: only files data/split.json assigns to the requested
    # split load, the rest (including files split.json doesn't mention at
    # all) are skipped.
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
        finally:
            SPLIT_PATH = original_split_path

    print("self-check passed: records load with correct label and shape, raw_landmarks "
          "pre-tagged detections pass through unchanged, and split filtering only loads "
          "files split.json assigns to the requested split")


if __name__ == "__main__":
    _self_check()
