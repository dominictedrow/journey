"""The platform layer for this OS: platform_win (Steam BTD6) on Windows, platform_mac (BTD6+) on macOS."""
import sys

if sys.platform == "win32":
    from .platform_win import *  # noqa: F401,F403
    from .platform_win import UI, Frame, Game, GameNotFound, cgimage_bgr, load_png, ocr_image, png_bytes  # noqa: F401
else:
    from .platform_mac import *  # noqa: F401,F403
    from .platform_mac import UI, Frame, Game, GameNotFound, cgimage_bgr, load_png, ocr_image, png_bytes  # noqa: F401
