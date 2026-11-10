"""PLAN.md step 0.4 — throwaway benchmark: capture FPS and HUD OCR latency. Not part of the harness.

Run with a BTD6+ game open (any map, HUD visible):
    .venv/bin/python scripts/bench_capture_ocr.py
HUD boxes are in window points for the 1470x830 windowed layout (title bar included).
"""
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mss
import mss.tools
import Vision
from Foundation import NSData

sys.path.insert(0, str(Path(__file__).parent))
from feasibility_mac import activate, find_window  # noqa: E402

HUD = {  # name: (x0, y0, x1, y1) in window points
    "lives": (125, 42, 200, 84),
    "money": (280, 42, 445, 84),
    "round": (1100, 52, 1182, 88),
}
N_CAPTURE = 60
N_OCR = 20


def ms(samples):
    return f"median {statistics.median(samples) * 1000:.1f} ms, p90 {sorted(samples)[int(len(samples) * 0.9)] * 1000:.1f} ms"


def png_bytes(img):
    return mss.tools.to_png(img.rgb, img.size)


def vision_ocr(png):
    """Apple Vision text recognition on PNG bytes. Returns the joined strings."""
    data = NSData.dataWithBytes_length_(png, len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    handler.performRequests_error_([req], None)
    return " ".join(r.topCandidates_(1)[0].string() for r in (req.results() or []))


def vision_ocr_fast(png):
    data = NSData.dataWithBytes_length_(png, len(png))
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelFast)
    req.setUsesLanguageCorrection_(False)
    handler.performRequests_error_([req], None)
    return " ".join(r.topCandidates_(1)[0].string() for r in (req.results() or []))


def tesseract_ocr(png):
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(png)
        f.flush()
        out = subprocess.run(["tesseract", f.name, "-", "--psm", "7",
                              "-c", "tessedit_char_whitelist=0123456789$,/"],
                             capture_output=True, text=True)
    return out.stdout.strip()


def main():
    w = find_window()
    if not w:
        raise SystemExit("Game window not found.")
    b, owner = w["kCGWindowBounds"], w["kCGWindowOwnerName"]
    print(f"window bounds (points): {dict(b)}")
    activate(owner, 0.8)
    win = {"left": int(b["X"]), "top": int(b["Y"]), "width": int(b["Width"]), "height": int(b["Height"])}

    with mss.MSS() as sct:
        sct.grab(win)  # warm-up
        t = []
        for _ in range(N_CAPTURE):
            t0 = time.perf_counter(); sct.grab(win); t.append(time.perf_counter() - t0)
        img = sct.grab(win)
        print(f"full-window grab {img.size}: {ms(t)} -> ~{1 / statistics.median(t):.0f} FPS")

        x0, y0, x1, y1 = HUD["money"]
        box = {"left": win["left"] + x0, "top": win["top"] + y0, "width": x1 - x0, "height": y1 - y0}
        t = []
        for _ in range(N_CAPTURE):
            t0 = time.perf_counter(); sct.grab(box); t.append(time.perf_counter() - t0)
        print(f"HUD-box grab: {ms(t)}")

        crops = {}
        for name, (x0, y0, x1, y1) in HUD.items():
            c = sct.grab({"left": win["left"] + x0, "top": win["top"] + y0, "width": x1 - x0, "height": y1 - y0})
            crops[name] = png_bytes(c)
            Path(tempfile.gettempdir(), f"hud_{name}.png").write_bytes(crops[name]) if "--save" in sys.argv else None

    for label, fn in (("Vision accurate", vision_ocr), ("Vision fast", vision_ocr_fast), ("tesseract", tesseract_ocr)):
        fn(crops["money"])  # warm-up
        for name, png in crops.items():
            t, text = [], ""
            for _ in range(N_OCR):
                t0 = time.perf_counter(); text = fn(png); t.append(time.perf_counter() - t0)
            print(f"{label:16} {name:6} -> {text!r:16} {ms(t)}")


if __name__ == "__main__":
    main()
