"""Standalone X-timeline fetch, always run as a subprocess (never imported
directly) by skills.concepts's __main__.

Why a subprocess: Playwright's own per-call timeouts (`goto`, `wait_for_selector`)
were observed to NOT fire reliably when X redirects an unauthenticated/expired
session to its login page — the call can hang well past its stated timeout,
apparently blocked in a native call that doesn't yield back to Python for
signal handling either. A `subprocess.run(..., timeout=...)` is OS-enforced
(SIGKILL on timeout) and is the only bound that reliably holds regardless of
what Playwright is stuck on. This is the concrete confirmation, called for by
plan.md, that X's session needs this kind of hard guard before concepts can
rely on it in a daily unattended run.

Prints a JSON list of candidate post texts to stdout on success. Any failure
exits non-zero with a message on stderr and no stdout output.
"""

import sys
import json

from .._lib import browser
from ._common import CANDIDATE_LIMIT, load_x_lists


def _extract_post_texts(page, limit):
    page.wait_for_selector("article", timeout=15000)
    texts = page.eval_on_selector_all(
        "article div[data-testid='tweetText']",
        "els => els.map(e => e.innerText)",
    )
    seen = []
    for t in texts:
        t = t.strip()
        if t and t not in seen:
            seen.append(t)
        if len(seen) >= limit:
            break
    return seen


def main():
    urls = ["https://x.com/home"] + load_x_lists()
    candidates = []
    with browser.persistent_context(headless=True) as ctx:
        for url in urls:
            page = ctx.new_page()
            try:
                page.goto(url, timeout=20000)
                candidates.extend(_extract_post_texts(page, CANDIDATE_LIMIT))
            finally:
                page.close()
            if len(candidates) >= CANDIDATE_LIMIT:
                break

    if not candidates:
        print("no posts extracted (session likely invalid or expired)", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(candidates[:CANDIDATE_LIMIT]))


if __name__ == "__main__":
    main()
