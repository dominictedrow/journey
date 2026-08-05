"""Speak a word aloud through the system voice.

mac.py's whole point is testing the translator against a real camera, so a
prediction needs to be audible, not just printed. macOS ships a `say`
command with instantly usable voices -- no extra dependency, no network
call, no API key.
"""
import shutil
import subprocess
import sys
import threading
import time
from unittest import mock

VOICE = "Rishi"  # macOS's built-in Indian-English voice
MIN_INTERVAL_SEC = 1.5  # ponytail: fixed cooldown, tune if words feel clipped or laggy

_lock = threading.Lock()
_last_spoken_at = 0.0


def speak(word: str) -> None:
    """Cap: the lock serializes playback so two calls can never overlap, and a call
    within MIN_INTERVAL_SEC of the last one that actually spoke is dropped instead of
    queued -- a burst of rapid predictions plays the first and skips the rest."""
    global _last_spoken_at
    with _lock:
        now = time.monotonic()
        if now - _last_spoken_at < MIN_INTERVAL_SEC:
            return
        _last_spoken_at = now
        subprocess.run(["say", "-v", VOICE, word], check=True)


def _self_check():
    global _last_spoken_at
    _last_spoken_at = 0.0
    with mock.patch("subprocess.run") as run:
        speak("hello")
        speak("world")  # inside the cooldown -- should be dropped
        assert run.call_count == 1, "cooldown should have suppressed the second call"
    _last_spoken_at = 0.0

    """ponytail: `say` only exists on macOS -- skips actual playback elsewhere instead of failing."""
    if shutil.which("say") is None:
        print("self-check skipped real playback: `say` is macOS-only, not available on this machine")
        return
    speak("test")
    print(f"self-check passed: cooldown suppresses rapid repeats, spoke a word through the {VOICE} voice")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        speak(" ".join(sys.argv[1:]))
    else:
        _self_check()
