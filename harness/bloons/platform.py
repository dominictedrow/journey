"""The platform layer for calib.PLATFORM: platform_mac (BTD6+) or platform_win (Steam BTD6).

Each backend only runs on its own OS, so a mode that doesn't match the OS fails here with a
clear message. Offline code that never imports this module is unaffected."""
import sys

from . import calib

_NEEDS = {"mac": ("darwin", "macOS (BTD6+)"), "win": ("win32", "Windows (Steam BTD6)")}


class PlatformMismatch(RuntimeError):
    pass


def check(platform=None, os_name=sys.platform):
    platform = platform or calib.PLATFORM
    want, label = _NEEDS[platform]
    if os_name != want:
        other = "win" if os_name == "win32" else "mac"
        raise PlatformMismatch(f"platform={platform!r} needs {label}; this machine is {os_name}. "
                               f"Use Bot(platform={other!r}) or --platform {other} "
                               "(the default is win).")


check()

if calib.PLATFORM == "win":
    from .platform_win import *  # noqa: F401,F403
    from .platform_win import UI, Frame, Game, GameNotFound, cgimage_bgr, load_png, ocr_image, png_bytes  # noqa: F401
else:
    from .platform_mac import *  # noqa: F401,F403
    from .platform_mac import UI, Frame, Game, GameNotFound, cgimage_bgr, load_png, ocr_image, png_bytes  # noqa: F401
