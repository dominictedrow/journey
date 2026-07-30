"""One-time, human-run interactive X login.

Run directly (not as a module):

    python scripts/auth_x.py

Opens a real, headed Chromium window against the persistent profile used by
the `concepts` skill, navigates to x.com, and waits for Dom to log in by
hand. This script never touches credentials or automates the login itself —
it just opens the browser and waits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills._lib.browser import persistent_context


def main():
    with persistent_context(headless=False) as ctx:
        page = ctx.new_page()
        page.goto("https://x.com")

        print(
            "\nA browser window has opened to x.com.\n"
            "Log into X manually in that window (username/password, 2FA, etc.).\n"
            "This script does not touch your credentials — it just waits for you.\n",
            file=sys.stderr,
        )
        input("Press Enter once you're logged in... ")

    print("Session saved. You can close this terminal.", file=sys.stderr)


if __name__ == "__main__":
    main()
