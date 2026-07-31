"""Run MediaPipe HandLandmarker over a video file, frame by frame, in VIDEO mode.

Outputs one JSON record per detected hand per frame: frame index, timestamp,
handedness, and 21 landmark (x, y, z) coordinates. Does not classify signs --
that's a separate downstream step.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"
DATA_DIR = Path(__file__).parent.parent / "data"


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
            if not result.hand_landmarks:
                # No hand detected at all this frame -- still record the frame
                # (handedness=None, no landmarks) so it isn't silently dropped
                # from the sequence. dataset.fuse_hands treats a record with no
                # matching handedness as a no-op, i.e. zero-filled -- the same
                # outcome mac.py's live path already gets for a handless frame.
                results.append({
                    "frame": frame_idx,
                    "timestamp_ms": timestamp_ms,
                    "handedness": None,
                    "landmarks": [],
                    "world_landmarks": [],
                })
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()

    return results


def process_and_save(video_path: str, out_dir: Path) -> str:
    """Unit of work for one worker process: run MediaPipe on one video, write its JSON.

    Runs inside its own process (see process_dir), so it builds its own
    HandLandmarker -- that per-process instance is what makes this parallel
    instead of just queued.
    """
    data = process_video(video_path)
    out_path = out_dir / (Path(video_path).stem + "_landmarks.json")
    out_path.write_text(json.dumps(data, indent=2))
    return out_path.name


def _pending_videos(dir_path: Path, out_dir: Path) -> list[Path]:
    """Every *.mp4 in dir_path that doesn't already have a landmarks JSON in out_dir."""
    return [v for v in sorted(dir_path.glob("*.mp4"))
            if not (out_dir / (v.stem + "_landmarks.json")).exists()]


def process_dir(dir_path: Path, out_dir: Path, workers: int) -> None:
    """Fan every *.mp4 in dir_path out across `workers` processes, each running
    its own MediaPipe instance. Already-processed videos (landmarks JSON
    already on disk) are skipped, so an interrupted run can just be re-run.
    """
    pending = _pending_videos(dir_path, out_dir)
    print(f"{len(pending)} videos pending in {dir_path} ({workers} workers)")

    out_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_and_save, str(v), out_dir): v for v in pending}
        for i, future in enumerate(as_completed(futures), 1):
            video = futures[future]
            try:
                name = future.result()
                print(f"[{i}/{len(pending)}] wrote {name}")
            except Exception as e:
                print(f"[{i}/{len(pending)}] FAILED {video.name}: {e}")


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

    # A blank (handless) frame must still show up in process_video's output --
    # as a handedness=None placeholder -- instead of being silently dropped.
    # Every frame surviving is what lets dataset.fuse_hands zero-fill it
    # instead of the sequence quietly losing a timestep.
    import tempfile
    NUM_FRAMES = 5
    with tempfile.TemporaryDirectory() as tmp:
        video_path = str(Path(tmp) / "blank.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (224, 224))
        for _ in range(NUM_FRAMES):
            writer.write(blank)
        writer.release()

        results = process_video(video_path)
        assert len(results) == NUM_FRAMES
        assert [r["frame"] for r in results] == list(range(NUM_FRAMES))
        assert all(r["handedness"] is None and r["landmarks"] == [] for r in results)

    # _pending_videos: a video with an existing landmarks JSON should be skipped,
    # one without should show up as pending -- this is the logic that makes
    # re-running process_dir after an interrupt resume instead of redo everything.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "done.mp4").touch()
        (tmp_path / "todo.mp4").touch()
        (tmp_path / "done_landmarks.json").write_text("[]")
        pending = _pending_videos(tmp_path, tmp_path)
        assert [p.name for p in pending] == ["todo.mp4"]

    print("self-check passed: pipeline wiring OK (no hands expected in a blank frame), "
          "pending-video resume logic skips already-processed videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", help="path to a single input video file")
    parser.add_argument("--dir", type=Path, default=None,
                         help="process every *.mp4 in this directory in parallel instead of one video")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                         help="parallel MediaPipe worker processes (--dir mode only)")
    parser.add_argument("-o", "--output", default=None, help="output JSON path (single-video mode only)")
    parser.add_argument("--self-test", action="store_true", help="run wiring smoke test instead of processing a video")
    args = parser.parse_args()

    if args.self_test or not (args.video or args.dir):
        _self_check()
        sys.exit(0)

    if args.dir:
        process_dir(args.dir, DATA_DIR, args.workers)
        sys.exit(0)

    data = process_video(args.video)
    if args.output:
        out_path = Path(args.output)
    else:
        DATA_DIR.mkdir(exist_ok=True)
        out_path = DATA_DIR / (Path(args.video).stem + "_landmarks.json")
    out_path.write_text(json.dumps(data, indent=2))
    print(f"wrote {len(data)} hand detections to {out_path}")
