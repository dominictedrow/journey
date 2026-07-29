"""Run MediaPipe HandLandmarker over a video file, frame by frame, in VIDEO mode.

Outputs one JSON record per detected hand per frame: frame index, timestamp,
handedness, and 21 landmark (x, y, z) coordinates. Does not classify signs --
that's a separate downstream step.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
DATA_DIR = Path(__file__).parent / "data"


def make_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def process_video(video_path: str) -> list[dict]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    landmarker = make_landmarker()
    results = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = int(frame_idx * 1000 / fps)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for hand_landmarks, world_landmarks, handedness in zip(
                    result.hand_landmarks, result.hand_world_landmarks, result.handedness):
                results.append({
                    "frame": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "handedness": handedness[0].category_name,
                    "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks],
                    "world_landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in world_landmarks],
                })
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()

    return results


def _self_check():
    """ponytail: smoke test with a synthetic blank frame, no real video file needed."""
    import numpy as np
    landmarker = make_landmarker()
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank)
    result = landmarker.detect_for_video(mp_image, 0)
    assert hasattr(result, "hand_landmarks")
    assert hasattr(result, "hand_world_landmarks")
    landmarker.close()
    print("self-check passed: pipeline wiring OK (no hands expected in a blank frame)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", help="path to input video file")
    parser.add_argument("-o", "--output", default=None, help="output JSON path")
    parser.add_argument("--self-test", action="store_true", help="run wiring smoke test instead of processing a video")
    args = parser.parse_args()

    if args.self_test or not args.video:
        _self_check()
        sys.exit(0)

    data = process_video(args.video)
    if args.output:
        out_path = Path(args.output)
    else:
        DATA_DIR.mkdir(exist_ok=True)
        out_path = DATA_DIR / (Path(args.video).stem + "_landmarks.json")
    out_path.write_text(json.dumps(data, indent=2))
    print(f"wrote {len(data)} hand detections to {out_path}")
