"""macOS platform layer: find the BTD6+ window, capture it, send input, OCR. The only OS-specific file.

What Phase 0 established (RESEARCH.md, 2026-11-04), and how this module follows it:
- Input only reaches the game while it is the frontmost app, so every input call
  brings it to the front first (`ensure_front`).
- Keys go through CGEventPostToPid. Clicks and drags go through the HID tap (it moves
  the real cursor). Never send modifier events: combos don't register, and any modifier
  key event makes the game ignore plain keys until `refocus()`.
- Capture uses CGWindowListCreateImage on the game window. It works while the window
  is covered or in the background, at Retina resolution (2 px per point). The API is
  deprecated; ScreenCaptureKit is the replacement if a macOS update removes it.
- OCR uses Apple Vision (accurate, no language correction): ~16 ms per HUD field.
  Parsing the text into numbers is portable and lives in bloons/hud.py.

Coordinates are normalized 0-1 over the content area (see bloons/coords.py and data/ui.json).
Methods return True/False or None for game-side problems; they raise only when the
window can't be found at all.
"""
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import Quartz as Q
import Vision
from AppKit import NSDate, NSRunLoop, NSRunningApplication, NSWorkspace

from bloons.coords import box_to_pixels, to_norm, to_screen
from bloons.hud import parse_hud

UI = json.loads((Path(__file__).parent / "data" / "ui.json").read_text())

# macOS virtual keycodes (ANSI layout).
KEYCODES = {
    **dict(zip("asdfhgzxcv", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))),
    "b": 11, "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17,
    "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25, "7": 26, "-": 27,
    "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38,
    "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "`": 50,
    "tab": 48, "space": 49, "backspace": 51, "escape": 53,
    "pageup": 116, "pagedown": 121,
}


class GameNotFound(RuntimeError):
    pass


@dataclass
class Frame:
    """One capture of the full game window (title bar included)."""
    image: object        # CGImageRef
    width: int           # pixels
    height: int
    scale: float         # pixels per point
    title_bar_px: int
    t: float             # time.monotonic() at capture

    def crop(self, box):
        """Normalized [x0, y0, x1, y1] -> CGImage of that region."""
        x, y, w, h = box_to_pixels(box, self.width, self.height, self.title_bar_px)
        return Q.CGImageCreateWithImageInRect(self.image, Q.CGRectMake(x, y, w, h))

    def content_bgr(self):
        """Content area (title bar cut off) as an HxWx3 BGR numpy array, for bloons/vision.py."""
        return cgimage_bgr(self.image)[self.title_bar_px:]

    def save_png(self, path, image=None):
        """Write the frame (or another CGImage) as a zlib-compressed RGB PNG. macOS's own PNG
        writer barely compresses (~7.5 MB per 2x frame); this gives ~2-3 MB."""
        Path(path).write_bytes(png_bytes(image or self.image))
        return True


def _rgbx(image):
    """CGImage -> (bytes, w, h): tightly packed 8-bit RGBX, whatever the source pixel format."""
    w, h = Q.CGImageGetWidth(image), Q.CGImageGetHeight(image)
    ctx = Q.CGBitmapContextCreate(None, w, h, 8, w * 4, Q.CGColorSpaceCreateDeviceRGB(),
                                  Q.kCGImageAlphaNoneSkipLast)
    Q.CGContextDrawImage(ctx, Q.CGRectMake(0, 0, w, h), image)
    return bytes(Q.CGDataProviderCopyData(Q.CGImageGetDataProvider(Q.CGBitmapContextCreateImage(ctx)))), w, h


def cgimage_bgr(image):
    """CGImage -> HxWx3 uint8 BGR numpy array (OpenCV's layout, for bloons/vision.py)."""
    import numpy as np
    data, w, h = _rgbx(image)
    return np.frombuffer(data, np.uint8).reshape(h, w, 4)[:, :, 2::-1].copy()


def png_bytes(image, level=9):
    """CGImage -> PNG bytes (8-bit RGB, no alpha). Redraws into a known RGBX layout first, so
    it works for any source pixel format."""
    import struct
    import zlib
    rgbx, w, h = _rgbx(image)
    raw = bytearray()
    for y in range(h):
        row = rgbx[y * w * 4:(y + 1) * w * 4]
        rgb = bytearray(w * 3)
        rgb[0::3], rgb[1::3], rgb[2::3] = row[0::4], row[1::4], row[2::4]
        raw += b"\x00" + rgb

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), level)) + chunk(b"IEND", b""))


def load_png(path):
    """PNG file -> CGImage (for offline tests on fixtures)."""
    url = Q.CFURLCreateWithFileSystemPath(None, str(path), Q.kCFURLPOSIXPathStyle, False)
    return Q.CGImageSourceCreateImageAtIndex(Q.CGImageSourceCreateWithURL(url, None), 0, None)


OCR_FALLBACK_REVISION = 2


def _recognize(image, revision=None):
    """One accurate Vision text request -> (ok, results)."""
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    if revision is not None:
        req.setRevision_(revision)
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    ok, _err = handler.performRequests_error_([req], None)
    return ok, req.results() or []


def ocr_image(image):
    """Text in a CGImage via Apple Vision (accurate, no language correction), lines joined
    by spaces. '' if nothing found.

    macOS 27.0.1: the default revision (3) fails on and off with an e5rt (Neural Engine)
    error, sometimes on every call in a process; revisions 1 and 2 don't (RESEARCH
    2026-11-07). A failed request falls back to OCR_FALLBACK_REVISION."""
    ok, results = _recognize(image)
    if not ok:
        ok, results = _recognize(image, revision=OCR_FALLBACK_REVISION)
    return " ".join(r.topCandidates_(1)[0].string() for r in results)


def find_window(owner_substr="bloons"):
    """The game's main window info dict, or None. Works in full screen and windowed mode.

    Lists all windows (a full-screen game sits in its own Space, off-screen while you're
    elsewhere) and picks the largest layer-0 window (the game also owns small menu-bar strips).
    """
    wins = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionAll, Q.kCGNullWindowID) or []
    game = [w for w in wins
            if owner_substr in (w.get("kCGWindowOwnerName") or "").lower() and w.get("kCGWindowLayer") == 0]
    if not game:
        return None
    return max(game, key=lambda w: w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"])


class Game:
    """Handle on the running BTD6+ window."""

    def __init__(self, owner_substr="bloons"):
        w = find_window(owner_substr)
        if not w:
            raise GameNotFound(f"No window owned by an app matching {owner_substr!r}. Is the game open?")
        self.pid = int(w["kCGWindowOwnerPID"])
        self.wid = int(w["kCGWindowNumber"])
        self.owner = w["kCGWindowOwnerName"]
        self.title_bar_pt = UI["window"]["title_bar_pt"]
        self.bounds = self._read_bounds()

    # --- window -------------------------------------------------------------------------
    def _read_bounds(self):
        info = Q.CGWindowListCopyWindowInfo(Q.kCGWindowListOptionIncludingWindow, self.wid) or []
        if not info:
            raise GameNotFound("Game window disappeared (closed or relaunched?).")
        b = info[0]["kCGWindowBounds"]
        return (float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"]))

    def reattach(self, timeout=10.0):
        """Find the game's window again (loading a map can drop it from the window list for a
        moment, and it may come back as a new window). True once found."""
        deadline = time.monotonic() + timeout
        while True:
            w = find_window()
            if w and w["kCGWindowBounds"]["Width"] > 600:
                self.pid, self.wid, self.owner = int(w["kCGWindowOwnerPID"]), int(w["kCGWindowNumber"]), w["kCGWindowOwnerName"]
                try:
                    self.bounds = self._read_bounds()
                    return True
                except GameNotFound:
                    pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)

    def refresh(self):
        """Re-read the window position (it can move between actions)."""
        self.bounds = self._read_bounds()
        return self.bounds

    @staticmethod
    def screen_locked():
        """True while the Mac is locked or the display is asleep. Captures then come back
        plain white and input is ignored (seen 2026-11-04 when the lid was closed mid-run)."""
        d = Q.CGSessionCopyCurrentDictionary() or {}
        return bool(d.get("CGSSessionScreenIsLocked")) or bool(Q.CGDisplayIsAsleep(Q.CGMainDisplayID()))

    def size_ok(self):
        """Is the window at the calibrated size (data/ui.json)?"""
        exp = UI["window"]["expected_pt"]
        return (round(self.bounds[2]), round(self.bounds[3])) == tuple(exp)

    def is_front(self):
        """Is the game the active app? NSWorkspace only learns about app switches through its
        run loop, so spin it for 10 ms first (a plain script has no run loop and would see a
        stale answer). The on-screen window order isn't enough: an active app with no windows
        (e.g. Finder) leaves the game's window on top while keys go elsewhere."""
        NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.01))
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        return bool(app) and app.processIdentifier() == self.pid

    def ensure_front(self, timeout=2.0):
        """Bring the game to the front; True once it is. Input is ignored otherwise."""
        if self.screen_locked():
            return False
        if self.is_front():
            return True
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(self.pid)
        if app:
            app.activateWithOptions_(0)
        if not self._wait_front(0.3):
            # macOS may refuse activation requests from a background process; AppleScript isn't refused.
            subprocess.run(["osascript", "-e", f'tell application "{self.owner}" to activate'], check=False)
            if not self._wait_front(timeout):
                return False
        return self._wait_settled()

    def _wait_settled(self, min_wait=0.25, timeout=1.5):
        """After activation, wait until the window stops moving. Stage Manager (on for this
        Mac) slides inactive windows aside and back, so bounds read mid-animation would aim
        clicks at the wrong place, and keys sent during the switch can be dropped."""
        start = time.monotonic()
        time.sleep(min_wait)
        last = self.refresh()
        while time.monotonic() - start < timeout:
            time.sleep(0.05)
            if self.refresh() == last:
                return True
            last = self.bounds
        return True  # still moving after the timeout: carry on with the latest bounds

    def _wait_front(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_front():
                return True
            time.sleep(0.05)
        return False

    # --- capture ------------------------------------------------------------------------
    def grab(self):
        """Capture the game window (works while covered). None if it can't be captured
        (minimized, or the screen is locked)."""
        if self.screen_locked():
            return None
        img = Q.CGWindowListCreateImage(Q.CGRectNull, Q.kCGWindowListOptionIncludingWindow, self.wid,
                                        Q.kCGWindowImageBoundsIgnoreFraming)
        if img is None:
            return None
        w, h = Q.CGImageGetWidth(img), Q.CGImageGetHeight(img)
        scale = w / self.bounds[2] if self.bounds[2] else 1.0
        return Frame(img, w, h, scale, round(self.title_bar_pt * scale), time.monotonic())

    def ocr(self, frame, box):
        """Text in a normalized box of a frame, via Apple Vision. '' if nothing found."""
        return ocr_image(frame.crop(box))

    def read_hud_raw(self, frame=None):
        """Raw OCR text of each HUD field in data/ui.json: {'lives': '178', 'money': '$1,552', ...}.
        {} if the window can't be captured."""
        frame = frame or self.grab()
        if frame is None:
            return {}
        return {name: self.ocr(frame, box) for name, box in UI["hud"].items()}

    def read_hud(self, frame=None):
        """Parsed HUD of one frame: Hud(lives, money, round, rounds_total), None per unreadable
        field. Feed successive readings through hud.HudFilter to smooth out misreads."""
        return parse_hud(self.read_hud_raw(frame))

    # --- input --------------------------------------------------------------------------
    def _screen(self, nx, ny):
        return to_screen(nx, ny, self.bounds, self.title_bar_pt)

    def _mouse(self, kind, pt):
        Q.CGEventPost(Q.kCGHIDEventTap, Q.CGEventCreateMouseEvent(None, kind, pt, Q.kCGMouseButtonLeft))

    def cursor(self):
        """The real mouse cursor as normalized content coords (outside 0-1 = off the content)."""
        self.refresh()
        loc = Q.CGEventGetLocation(Q.CGEventCreate(None))
        return to_norm(loc.x, loc.y, self.bounds, self.title_bar_pt)

    def move(self, nx, ny):
        if not self.ensure_front():
            return False
        self._mouse(Q.kCGEventMouseMoved, self._screen(nx, ny))
        return True

    def click(self, nx, ny):
        """Left click at a normalized point (HID tap: moves the real cursor)."""
        if not self.ensure_front():
            return False
        pt = self._screen(nx, ny)
        for kind in (Q.kCGEventMouseMoved, Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp):
            self._mouse(kind, pt)
            time.sleep(0.05)
        return True

    def drag(self, a, b, steps=12, duration=0.3):
        """Press at normalized point a, move to b, release."""
        if not self.ensure_front():
            return False
        pa, pb = self._screen(*a), self._screen(*b)
        self._mouse(Q.kCGEventMouseMoved, pa)
        time.sleep(0.03)
        self._mouse(Q.kCGEventLeftMouseDown, pa)
        for i in range(1, steps + 1):
            f = i / steps
            self._mouse(Q.kCGEventLeftMouseDragged, (pa[0] + (pb[0] - pa[0]) * f, pa[1] + (pb[1] - pa[1]) * f))
            time.sleep(duration / steps)
        self._mouse(Q.kCGEventLeftMouseUp, pb)
        return True

    def scroll(self, nx, ny, lines):
        """Scroll wheel over a normalized point: `lines` > 0 scrolls up, < 0 down. Sent as HID
        events of at most 5 lines each (the shop needs ~30 to go end to end, RESEARCH 3.1)."""
        if not self.move(nx, ny):
            return False
        step = 5 if lines > 0 else -5
        left = int(lines)
        while left:
            n = step if abs(left) >= 5 else left
            Q.CGEventPost(Q.kCGHIDEventTap, Q.CGEventCreateScrollWheelEvent(None, Q.kCGScrollEventUnitLine, 1, n))
            left -= n
            time.sleep(0.02)
        return True

    def key(self, name):
        """Tap one plain key ('q', ',', 'space', 'escape', ...). No modifier combos (they don't
        register in BTD6+)."""
        code = KEYCODES.get(name.lower())
        if code is None:
            raise ValueError(f"Unknown key {name!r}")
        if not self.ensure_front():
            return False
        for down in (True, False):
            e = Q.CGEventCreateKeyboardEvent(None, code, down)
            Q.CGEventSetFlags(e, 0)
            Q.CGEventPostToPid(self.pid, e)
            time.sleep(0.04)
        return True

    def refocus(self):
        """Recover the game's keyboard handling: switch to Finder and back. Any synthetic
        modifier event (Ctrl/Shift/Cmd/Option key-up, or a flagged combo) leaves BTD6+ ignoring
        plain keys until it loses and regains focus. So never send modifiers; call this if
        hotkeys stop registering."""
        subprocess.run(["osascript", "-e", 'tell application "Finder" to activate'], check=False)
        time.sleep(0.5)
        return self.ensure_front()
