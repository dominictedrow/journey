"""Windows platform layer for Steam BTD6: find the window, capture it, send input, OCR.
Same interface as platform_mac.py (Game, Frame, UI, load_png, ocr_image, png_bytes); with it,
the only OS-specific files.

- Window: the Unity window titled "BloonsTD6". Coordinates use the client area (no title
  bar), in physical pixels (the process is made per-monitor DPI-aware on import).
- Capture: PrintWindow(PW_RENDERFULLCONTENT) works while covered; if it comes back black
  (some DX11 builds), a screen grab of the client rect (needs the window visible).
- Input: SendInput (real cursor, real keys) with the game in front. Window messages don't
  work: Unity takes a posted click at the real cursor's position (RESEARCH 2026-11-09).
- OCR: the HUD numbers go through bloons/glyphs.py (Windows OCR misreads that font);
  other text uses the built-in Windows OCR engine (winrt), crops upscaled.
"""
import asyncio
import ctypes
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import win32api
import win32con
import win32gui
import win32process
import win32ui

from bloons import calib, glyphs
from bloons.hud import parse_hud

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except OSError:
    pass  # already set for this process

calib.use("win")   # importing this backend directly (scripts) means win data too
UI = json.loads(calib.path("ui.json").read_text())
TITLE = "BloonsTD6"
STEAM_URL = "steam://rungameid/960090"
PW_RENDERFULLCONTENT = 2
CLICK_HOVER_S = 0.12   # calibration knobs: raise if clicks get dropped
CLICK_HOLD_S = 0.08
WHEEL_PER_LINE = 80

VK = {**{c: ord(c.upper()) for c in "abcdefghijklmnopqrstuvwxyz0123456789"},
      "tab": 0x09, "space": 0x20, "backspace": 0x08, "escape": 0x1B, "enter": 0x0D,
      "pageup": 0x21, "pagedown": 0x22, ",": 0xBC, ".": 0xBE, "/": 0xBF, ";": 0xBA,
      "[": 0xDB, "]": 0xDD, "\\": 0xDC, "-": 0xBD, "=": 0xBB, "`": 0xC0}


class GameNotFound(RuntimeError):
    pass


@dataclass
class Frame:
    """One capture of the client area as an HxWx3 BGR array."""
    image: np.ndarray
    t: float
    title_bar_px: int = 0
    scale: float = 1.0

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    def crop(self, box):
        h, w = self.image.shape[:2]
        return self.image[round(box[1] * h):round(box[3] * h), round(box[0] * w):round(box[2] * w)]

    def content_bgr(self):
        return self.image

    def save_png(self, path, image=None):
        return cv2.imwrite(str(path), self.image if image is None else image)


def load_png(path):
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def png_bytes(image, level=9):
    return cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, level])[1].tobytes()


def cgimage_bgr(image):
    """Name kept for callers written against platform_mac: images are already BGR arrays."""
    return image


# --- OCR -----------------------------------------------------------------------------------
_ocr = {}
OCR_MIN_H = 60   # px: crops are upscaled to at least this height (Windows OCR misses small text)


def _engine():
    if not _ocr:
        from winrt.windows.media.ocr import OcrEngine
        _ocr["engine"] = OcrEngine.try_create_from_user_profile_languages()
        _ocr["loop"] = asyncio.new_event_loop()
    return _ocr["engine"], _ocr["loop"]


def ocr_image(image):
    """Text in a BGR array via Windows OCR, lines joined by spaces. '' if nothing found."""
    if image is None or image.size == 0:
        return ""
    from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
    from winrt.windows.storage.streams import DataWriter
    h = image.shape[0]
    if h < OCR_MIN_H:
        f = OCR_MIN_H / h
        image = cv2.resize(image, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    w = DataWriter()
    w.write_bytes(bgra.tobytes())
    bmp = SoftwareBitmap.create_copy_from_buffer(w.detach_buffer(), BitmapPixelFormat.BGRA8,
                                                 bgra.shape[1], bgra.shape[0], BitmapAlphaMode.PREMULTIPLIED)
    engine, loop = _engine()
    res = loop.run_until_complete(engine.recognize_async(bmp))
    return " ".join(line.text for line in res.lines)


# --- window --------------------------------------------------------------------------------
def find_window(title=TITLE):
    hwnd = win32gui.FindWindow(None, title)
    return hwnd if hwnd and win32gui.IsWindow(hwnd) else None


ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x1, 0x2


def keep_awake(on=True):
    """Stop the PC sleeping or blanking the display while the bot runs."""
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


class Game:
    """Handle on the running BTD6 window (Steam)."""

    def __init__(self, title=TITLE):
        self.title = title
        self.hwnd = find_window(title)
        if not self.hwnd:
            raise GameNotFound(f"No window titled {title!r}. Is the game open?")
        self.pid = win32process.GetWindowThreadProcessId(self.hwnd)[1]
        self.title_bar_pt = 0
        self.bounds = self._read_bounds()
        keep_awake()

    def _read_bounds(self):
        if not win32gui.IsWindow(self.hwnd):
            raise GameNotFound("Game window disappeared (closed or relaunched?).")
        _, _, w, h = win32gui.GetClientRect(self.hwnd)
        x, y = win32gui.ClientToScreen(self.hwnd, (0, 0))
        return (float(x), float(y), float(w), float(h))

    def refresh(self):
        self.bounds = self._read_bounds()
        return self.bounds

    def reattach(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while True:
            hwnd = find_window(self.title)
            if hwnd:
                self.hwnd = hwnd
                self.pid = win32process.GetWindowThreadProcessId(hwnd)[1]
                try:
                    if self.refresh()[2] > 600:
                        return True
                except GameNotFound:
                    pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.25)

    def relaunch(self, timeout=120.0):
        """Start the game through Steam (after a crash) and reattach."""
        os.startfile(STEAM_URL)
        return self.reattach(timeout)

    @staticmethod
    def screen_locked():
        """True while the workstation is locked (input desktop isn't ours)."""
        h = ctypes.windll.user32.OpenInputDesktop(0, False, 0x0100)  # DESKTOP_SWITCHDESKTOP
        if not h:
            return True
        ok = ctypes.windll.user32.SwitchDesktop(h)
        ctypes.windll.user32.CloseDesktop(h)
        return not ok

    def size_ok(self):
        exp = UI["window"]["expected_pt"]
        return (round(self.bounds[2]), round(self.bounds[3])) == tuple(exp)

    def is_front(self):
        return win32gui.GetForegroundWindow() == self.hwnd

    def ensure_front(self, timeout=2.0):
        if self.screen_locked():
            return False
        if self.is_front():
            return True
        if win32gui.IsIconic(self.hwnd):
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
        # Windows refuses SetForegroundWindow from a background process unless it just got input:
        # a lone Alt tap counts.
        win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
        win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
        try:
            win32gui.SetForegroundWindow(self.hwnd)
        except Exception:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_front():
                time.sleep(0.15)
                self.refresh()
                return True
            time.sleep(0.05)
        return False

    # --- capture ---------------------------------------------------------------------------
    def grab(self):
        """Capture the client area. None if it can't be captured (minimized, locked)."""
        if self.screen_locked() or win32gui.IsIconic(self.hwnd):
            return None
        try:
            self.refresh()
        except GameNotFound:
            return None
        try:
            img = self._print_window()
        except win32ui.error as e:              # the window died mid-capture (game crashed)
            if not win32gui.IsWindow(self.hwnd):
                raise GameNotFound("Game window disappeared mid-capture.") from e
            img = None
        if img is None or img.max() < 8:
            img = self._screen_grab()
        return None if img is None else Frame(img, time.monotonic())

    def _print_window(self):
        _, _, w, h = win32gui.GetClientRect(self.hwnd)
        if w <= 0 or h <= 0:
            return None
        hdc = win32gui.GetWindowDC(self.hwnd)
        src = win32ui.CreateDCFromHandle(hdc)
        mem = src.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(src, w, h)
        mem.SelectObject(bmp)
        try:
            ok = ctypes.windll.user32.PrintWindow(self.hwnd, mem.GetSafeHdc(), PW_RENDERFULLCONTENT | 1)  # 1 = client only
            buf = bmp.GetBitmapBits(True)
        finally:
            win32gui.DeleteObject(bmp.GetHandle())
            mem.DeleteDC()
            src.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hdc)
        if not ok:
            return None
        return np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    def _screen_grab(self):
        import mss
        x, y, w, h = map(int, self.bounds)
        with mss.mss() as s:
            shot = s.grab({"left": x, "top": y, "width": w, "height": h})
        return np.asarray(shot)[:, :, :3].copy()

    def ocr(self, frame, box):
        return ocr_image(frame.crop(box))

    def read_number(self, frame, box):
        """Prices and other outlined numbers ('$1,530', '200XP'): glyph templates."""
        return glyphs.read(frame.crop(box))

    def read_hud_raw(self, frame=None):
        frame = frame or self.grab()
        if frame is None:
            return {}
        return {name: glyphs.read(frame.crop(box)) for name, box in UI["hud"].items()}

    def read_hud(self, frame=None):
        return parse_hud(self.read_hud_raw(frame))

    # --- input -----------------------------------------------------------------------------
    def _client(self, nx, ny):
        return round(nx * self.bounds[2]), round(ny * self.bounds[3])

    def _screen(self, nx, ny):
        cx, cy = self._client(nx, ny)
        return int(self.bounds[0]) + cx, int(self.bounds[1]) + cy

    def cursor(self):
        self.refresh()
        x, y = win32api.GetCursorPos()
        return (x - self.bounds[0]) / self.bounds[2], (y - self.bounds[1]) / self.bounds[3]

    def move(self, nx, ny):
        if not self.ensure_front():
            return False
        win32api.SetCursorPos(self._screen(nx, ny))
        return True

    def click(self, nx, ny):
        if not self.move(nx, ny):
            return False
        time.sleep(CLICK_HOVER_S)   # Unity drops a press that lands in the same frame as the jump
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(CLICK_HOLD_S)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        time.sleep(0.05)
        return True

    def drag(self, a, b, steps=12, duration=0.3):
        if not self.move(*a):
            return False
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        for i in range(1, steps + 1):
            f = i / steps
            win32api.SetCursorPos(self._screen(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f))
            time.sleep(duration / steps)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        return True

    def scroll(self, nx, ny, lines):
        """Wheel over a normalized point: lines > 0 scrolls up. The game only takes wheel input
        after a relative mouse move over the target (a bare SetCursorPos isn't enough), so each
        chunk is preceded by a 2 px jiggle. WHEEL_PER_LINE: the shop is ~30 lines end to end,
        as on the Mac (2026-11-09)."""
        if not self.move(nx, ny):
            return False
        time.sleep(CLICK_HOVER_S)
        left = int(lines)
        while left:
            n = max(-30, min(30, left))
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 2, 0)
            time.sleep(0.02)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -2, 0)
            time.sleep(0.02)
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, n * WHEEL_PER_LINE)
            left -= n
            time.sleep(0.05)
        return True

    def key(self, name):
        vk = VK.get(name.lower())
        if vk is None:
            raise ValueError(f"Unknown key {name!r}")
        if not self.ensure_front():
            return False
        scan = win32api.MapVirtualKey(vk, 0)
        win32api.keybd_event(vk, scan, 0, 0)
        time.sleep(0.04)
        win32api.keybd_event(vk, scan, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.04)
        return True

    def refocus(self):
        return self.ensure_front()
