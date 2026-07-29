"""Build and validate the one JSON envelope every skill emits on stdout."""

import json
import sys
from datetime import datetime, timezone

SECTION_TYPES = {"stat", "list", "table", "text"}
REQUIRED_KEYS = {"skill", "ok", "ran_at", "summary", "sections", "error"}


def build(skill, ok, summary, sections=None, error=None):
    return {
        "skill": skill,
        "ok": ok,
        "ran_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "summary": summary,
        "sections": sections or [],
        "error": error,
    }


def validate(envelope):
    """Raise ValueError with a specific reason if envelope doesn't match the contract."""
    if not isinstance(envelope, dict):
        raise ValueError(f"envelope must be a dict, got {type(envelope).__name__}")

    missing = REQUIRED_KEYS - envelope.keys()
    if missing:
        raise ValueError(f"envelope missing required keys: {sorted(missing)}")

    if not isinstance(envelope["skill"], str) or not envelope["skill"]:
        raise ValueError("envelope['skill'] must be a non-empty string")
    if not isinstance(envelope["ok"], bool):
        raise ValueError("envelope['ok'] must be a bool")
    if not isinstance(envelope["summary"], str):
        raise ValueError("envelope['summary'] must be a string")
    if not isinstance(envelope["sections"], list):
        raise ValueError("envelope['sections'] must be a list")
    if envelope["error"] is not None and not isinstance(envelope["error"], str):
        raise ValueError("envelope['error'] must be a string or null")

    for i, section in enumerate(envelope["sections"]):
        if not isinstance(section, dict):
            raise ValueError(f"sections[{i}] must be a dict")
        if section.get("type") not in SECTION_TYPES:
            raise ValueError(f"sections[{i}]['type'] must be one of {sorted(SECTION_TYPES)}")
        if not isinstance(section.get("title"), str):
            raise ValueError(f"sections[{i}]['title'] must be a string")

    return envelope


def emit(skill, ok, summary, sections=None, error=None):
    """Validate and print the envelope to stdout as the skill's sole stdout output."""
    envelope = build(skill, ok, summary, sections, error)
    validate(envelope)
    print(json.dumps(envelope))
    return envelope


def progress(*args):
    """Human-readable progress line — goes to stderr, never stdout."""
    print(*args, file=sys.stderr, flush=True)
