"""Speak a word aloud through the system voice.

mac.py's whole point is testing the translator against a real camera, so a
prediction needs to be audible, not just printed. macOS ships a `say`
command with instantly usable voices -- no extra dependency, no network
call, no API key.
"""
import shutil
import subprocess
import sys

VOICE = "Rishi"  # macOS's built-in Indian-English voice


def speak(word: str) -> None:
    subprocess.run(["say", "-v", VOICE, word], check=True)


def _self_check():
    """ponytail: `say` only exists on macOS -- skips actual playback elsewhere instead of failing."""
    if shutil.which("say") is None:
        print("self-check skipped: `say` is macOS-only, not available on this machine")
        return
    speak("test")
    print(f"self-check passed: spoke a word through the {VOICE} voice")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        speak(" ".join(sys.argv[1:]))
    else:
        _self_check()
