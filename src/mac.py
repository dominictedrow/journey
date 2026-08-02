"""Live camera test of the sign translator.

Connects process_video.py's MediaPipe HandLandmarker to a webcam feed instead
of a video file, fuses each frame the same way dataset.py does, and feeds a
rolling window of the last SEQUENCE_LENGTH frames through a trained SignLSTM
checkpoint. This is the real-life usability check -- everywhere else in the
pipeline only ever sees pre-recorded, pre-labeled clips.
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

from dataset import HAND_FEATURE_DIM, fuse_hands, load_vocab
from model import build_model
from process_video import make_landmarker
from speak import speak
from train import SEQUENCE_LENGTH


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

    window = deque(maxlen=SEQUENCE_LENGTH)
    start_time = time.time()
    predicted_word = ""
    last_spoken = ""

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int((time.time() - start_time) * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            window.append(fuse_hands(frame_to_hand_records(result)))

            if len(window) == SEQUENCE_LENGTH:
                x = torch.from_numpy(np.stack(window)).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                predicted_word = vocab[logits[0].argmax(dim=-1).item()]
                if predicted_word != last_spoken:
                    # say blocks while it plays -- run it off the main
                    # thread so the camera loop doesn't stall mid-word.
                    threading.Thread(target=speak, args=(predicted_word,), daemon=True).start()
                    last_spoken = predicted_word

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

    fused = fuse_hands(frame_to_hand_records(result))
    assert fused.shape == (2 * HAND_FEATURE_DIM,)

    window = deque([fused] * SEQUENCE_LENGTH, maxlen=SEQUENCE_LENGTH)
    x = torch.from_numpy(np.stack(window)).unsqueeze(0)
    model = build_model()
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, model.classifier.out_features)
    print("self-check passed: camera-frame -> landmarks -> fused vector -> rolling window -> "
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
