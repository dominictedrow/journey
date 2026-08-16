"""Live camera test of the sign translator.

Connects MediaPipe HolisticLandmarker to a webcam feed instead of a video
file, fuses each frame the same way dataset.py does, and buffers frames for
one gesture at a time (hands present -> absent), resampling each completed
gesture to SEQUENCE_LENGTH steps the same way training clips were built
before running it through a trained SignLSTM checkpoint. This is the
real-life usability check -- everywhere else in the pipeline only ever sees
pre-recorded, pre-labeled clips.

Uses Holistic, not process_video.py's hand-only HandLandmarker, and that is now
load-bearing rather than merely nice: dataset.fuse_landmarks normalizes every
part into one shared body reference frame anchored on the pose shoulders (or,
failing that, the face). Hand-only capture resolves no such frame, so the whole
body-frame block -- hand position, lip shape, pose -- would come through as
zeros at inference time while the model was trained on real values.
"""
import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dataset import COORD_DIM, finalize_sequence, fuse_landmarks, load_vocab, resample_sequence
from model import build_model
from speak import speak
from train import SEQUENCE_LENGTH

MODEL_PATH = Path(__file__).parent.parent / "models" / "holistic_landmarker.task"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "models" / "checkpoint.pt"

# dataset.py: "Clips run 14-384 frames (median ~80)" -- training never saw a
# gesture outside that range, so live segmentation is bounded the same way.
MIN_GESTURE_FRAMES = 14
MAX_GESTURE_FRAMES = 384


def make_landmarker():
    options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def frame_to_holistic_records(result) -> list[dict]:
    """HolisticLandmarkerResult -> the same "part"-tagged shape
    extract_kaggle_landmarks.py produces from the Kaggle corpus, so
    fuse_landmarks treats live frames and training clips identically. Unlike
    HandLandmarker's hand_landmarks/handedness (one entry per detected hand),
    Holistic returns one flat landmark list per part, already split into
    left_hand/right_hand -- no handedness lookup needed."""
    parts = {
        "face": result.face_landmarks,
        "pose": result.pose_landmarks,
        "left_hand": result.left_hand_landmarks,
        "right_hand": result.right_hand_landmarks,
    }
    return [
        {"part": part, "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]}
        for part, landmarks in parts.items()
        if landmarks
    ]


def run_live(checkpoint: Path, camera_index: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab()
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    landmarker = make_landmarker()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError(f"Could not open camera {camera_index}")

    start_time = time.time()
    predicted_word = ""
    gesture_buffer: list[np.ndarray] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # Segment the continuous stream into one-gesture-at-a-time chunks
            # instead of classifying a raw fixed-size window every frame: the
            # model only ever saw whole gestures resampled to SEQUENCE_LENGTH
            # steps, never an arbitrary slice of continuous, unsegmented video.
            # Gesture boundaries are still keyed on hands specifically (not
            # face/pose, which can stay visible with hands at rest) -- same
            # segmentation signal as before, now alongside the richer per-frame
            # feature vector.
            hands_present = bool(result.left_hand_landmarks) or bool(result.right_hand_landmarks)
            if hands_present:
                gesture_buffer.append(fuse_landmarks(frame_to_holistic_records(result)))

            gesture_ended = not hands_present and gesture_buffer
            gesture_overlong = hands_present and len(gesture_buffer) >= MAX_GESTURE_FRAMES
            if gesture_ended or gesture_overlong:
                if len(gesture_buffer) >= MIN_GESTURE_FRAMES:
                    # finalize_sequence (velocity + presence blocks) is the
                    # same call the training datasets make, so live inference
                    # and training cannot drift apart on feature layout.
                    coords = finalize_sequence(resample_sequence(gesture_buffer, SEQUENCE_LENGTH))
                    x = torch.from_numpy(coords).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                    predicted_word = vocab[logits[0].argmax(dim=-1).item()]
                    # say blocks while it plays -- run it off the main
                    # thread so the camera loop doesn't stall mid-word.
                    threading.Thread(target=speak, args=(predicted_word,), daemon=True).start()
                gesture_buffer = []
            elif not hands_present:
                predicted_word = ""

            cv2.putText(frame, predicted_word, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("sign translator (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


def _self_check():
    """ponytail: wiring smoke test with a synthetic blank frame, no real camera needed (mirrors process_video.py's self-check)."""
    landmarker = make_landmarker()
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=blank)
    result = landmarker.detect_for_video(mp_image, 0)
    landmarker.close()

    fused = fuse_landmarks(frame_to_holistic_records(result))
    assert fused.shape == (COORD_DIM,)  # blank frame -> nothing detected -> all slots zero

    window = deque([fused] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    coords = finalize_sequence(resample_sequence(list(window), SEQUENCE_LENGTH))
    x = torch.from_numpy(coords).unsqueeze(0)
    model = build_model()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, model.classifier.out_features)
    print("self-check passed: camera-frame -> landmarks -> fused vector -> buffered gesture -> "
          "model forward pass all wired correctly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="checkpoint to use (default: models/checkpoint.pt)")
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument("--self-test", action="store_true", help="run wiring smoke test instead of opening the camera")
    args = parser.parse_args()

    if args.self_test:
        _self_check()
        sys.exit(0)

    run_live(args.checkpoint, args.camera)
