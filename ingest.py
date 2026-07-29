"""Load MediaPipe hand-landmark JSON files (written by process_video.py) into
records ready for the rest of the ingestion pipeline. This is the connection
point to the MediaPipe output -- no encoding, scaling, or batching happens
here, that's separate steps.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_records(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load every *_landmarks.json file in data_dir and tag each detection with its label.

    process_video.py writes one JSON file per video, named <video_stem>_landmarks.json.
    The label is that stem with the "_landmarks" suffix stripped (e.g. "hello_landmarks.json" -> "hello").
    Returns one dict per detected hand per frame: label, frame, timestamp_ms, handedness, landmarks, world_landmarks.
    """
    records = []
    for path in sorted(Path(data_dir).glob("*_landmarks.json")):
        label = path.stem.removesuffix("_landmarks")
        for detection in json.loads(path.read_text()):
            records.append({"label": label, **detection})
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
        assert records[0]["label"] == "hello"
        assert records[0]["handedness"] == "Right"
        assert len(records[0]["landmarks"]) == 21
    print("self-check passed: records load with correct label and shape")


if __name__ == "__main__":
    _self_check()
