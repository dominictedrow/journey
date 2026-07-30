"""Shared between __main__.py (orchestrator) and _x_fetch.py (subprocess)."""

import json
from pathlib import Path

CANDIDATE_LIMIT = 40


def config_path():
    return Path.home() / "Library" / "Application Support" / "cwork" / "config.json"


def load_x_lists():
    try:
        with open(config_path(), encoding="utf-8") as f:
            data = json.load(f)
        lists = (data.get("concepts") or {}).get("x_lists") or []
        return [str(u) for u in lists if u]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return []
