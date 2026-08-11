"""Live camera test of the sign translator.

Connects process_video.py's MediaPipe HandLandmarker to a webcam feed instead
of a video file, fuses each frame the same way dataset.py does, and buffers
frames for one gesture at a time (hands present -> absent), resampling each
completed gesture to SEQUENCE_LENGTH steps the same way training clips were
built before running it through a trained SignLSTM checkpoint. This is the
real-life usability check -- everywhere else in the pipeline only ever sees
pre-recorded, pre-labeled clips.
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

from dataset import TOTAL_FEATURE_DIM, fuse_landmarks, load_vocab, resample_sequence
from model import build_model
from process_video import make_landmarker
from speak import speak
from train import SEQUENCE_LENGTH

# dataset.py: "Clips run 14-384 frames (median ~80)" -- training never saw a
# gesture outside that range, so live segmentation is bounded the same way.
MIN_GESTURE_FRAMES = 14
MAX_GESTURE_FRAMES = 384


def frame_to_hand_records(result) -> list[dict]:
    """Same per-hand shape process_video.py writes, minus frame/timestamp bookkeeping."""
    return [
        {
            "handedness": handedness[0].category_name,
            "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks],
        }
        for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness)
    ]


def run_live(checkpoint: Path, camera_index: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab()
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
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
            hands_present = len(result.hand_landmarks) > 0
            if hands_present:
                # This still only ever runs MediaPipe HandLandmarker, not
                # Holistic, so fuse_landmarks' face/pose slots stay zero here
                # -- live inference won't see those features until this
                # capture is upgraded to Holistic too.
                gesture_buffer.append(fuse_landmarks(frame_to_hand_records(result)))

            gesture_ended = not hands_present and gesture_buffer
            gesture_overlong = hands_present and len(gesture_buffer) >= MAX_GESTURE_FRAMES
            if gesture_ended or gesture_overlong:
                if len(gesture_buffer) >= MIN_GESTURE_FRAMES:
                    coords = resample_sequence(gesture_buffer, SEQUENCE_LENGTH)
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

    fused = fuse_landmarks(frame_to_hand_records(result))
    assert fused.shape == (TOTAL_FEATURE_DIM,)  # face/pose slots zero -- HandLandmarker only

    window = deque([fused] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    x = torch.from_numpy(np.stack(window)).unsqueeze(0)
    model = build_model()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, model.classifier.out_features)
    print("self-check passed: camera-frame -> landmarks -> fused vector -> buffered gesture -> "
          "model forward pass all wired correctly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None, help="path to a checkpoint saved by train.py")
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument("--self-test", action="store_true", help="run wiring smoke test instead of opening the camera")
    args = parser.parse_args()

    if args.self_test or not args.checkpoint:
        _self_check()
        sys.exit(0)

    run_live(args.checkpoint, args.camera)
