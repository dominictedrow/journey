"""Generic, minimal Playwright persistent-context helper.

Not skill-specific — just a thin wrapper around
`chromium.launch_persistent_context`. Because the context is persistent,
cookies/localStorage are written straight to `user_data_dir` on disk and
auto-persist across process runs; there's no manual storage_state
export/import step required.
"""

from contextlib import contextmanager
from pathlib import Path


def default_profile_dir():
    path = Path.home() / "Library" / "Application Support" / "cwork" / "x_profile"
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


@contextmanager
def persistent_context(headless=True, user_data_dir=None):
    """Launch a persistent Chromium context and yield it.

    Usage:
        with persistent_context(headless=False) as ctx:
            page = ctx.new_page()
            page.goto("https://x.com")
    """
    from playwright.sync_api import sync_playwright

    user_data_dir = user_data_dir or default_profile_dir()
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir), headless=headless
        )
        try:
            yield ctx
        finally:
            ctx.close()
