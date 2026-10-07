"""Subprocess wrapper around `claude -p` — the only way skills talk to Claude.

Token-discipline rules enforced here (see plan.md): truncate oversized input,
always request JSON output, and cache identical prompts for 24h in SQLite so
repeat runs don't re-spend tokens on unchanged input.
"""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from . import store

_MAX_INPUT_BYTES = 8 * 1024
_CACHE_TTL_HOURS = 24

# launchd agents and Finder/Dock-launched Electron get a minimal PATH that
# doesn't include where the `claude` CLI is usually installed, so fall back
# to its common install locations when it isn't on PATH.
_CLAUDE_FALLBACKS = (
    Path.home() / ".local" / "bin" / "claude",
    Path("/opt/homebrew/bin/claude"),
    Path("/usr/local/bin/claude"),
)


def _claude_bin():
    override = os.environ.get("CWORK_CLAUDE_BIN")
    if override:
        return override
    found = shutil.which("claude")
    if found:
        return found
    for candidate in _CLAUDE_FALLBACKS:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError(
        "claude CLI not found on PATH or in ~/.local/bin, /opt/homebrew/bin, /usr/local/bin "
        "(set CWORK_CLAUDE_BIN to its absolute path)"
    )


def _truncate(prompt):
    encoded = prompt.encode("utf-8")
    if len(encoded) <= _MAX_INPUT_BYTES:
        return prompt
    return encoded[:_MAX_INPUT_BYTES].decode("utf-8", errors="ignore")


def _cache_key(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _cache_get(key):
    conn = store.connect()
    row = conn.execute(
        "SELECT response FROM claude_cache "
        "WHERE cache_key = ? AND created_at > datetime('now', ?)",
        (key, f"-{_CACHE_TTL_HOURS} hours"),
    ).fetchone()
    conn.close()
    return json.loads(row["response"]) if row else None


def _cache_put(key, response):
    conn = store.connect()
    conn.execute(
        "INSERT OR REPLACE INTO claude_cache (cache_key, response, created_at) "
        "VALUES (?, ?, datetime('now'))",
        (key, json.dumps(response)),
    )
    conn.commit()
    conn.close()


def ask(prompt, use_cache=True):
    """Send `prompt` to `claude -p --output-format json` and return the parsed JSON.

    `prompt` must already be reasoned-down input (never raw HTML) — callers
    are responsible for parsing/filtering before this is called.
    """
    prompt = _truncate(prompt)
    key = _cache_key(prompt)

    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            return cached

    result = subprocess.run(
        [_claude_bin(), "-p", prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
        check=True,
    )
    response = json.loads(result.stdout)

    if use_cache:
        _cache_put(key, response)

    return response
