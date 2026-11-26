"""Repo-relative paths. Everything is built with pathlib from REPO_ROOT (cross-platform rule)."""

from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    override = os.environ.get("QT_HOME")
    if override:
        return Path(override).resolve()
    # src/qt/common/paths.py -> parents[3] is the repo root
    return Path(__file__).resolve().parents[3]


REPO_ROOT: Path = _find_root()
CONFIG_DIR: Path = REPO_ROOT / "configs"
ENV_FILE: Path = REPO_ROOT / ".env"
MIGRATIONS_DIR: Path = REPO_ROOT / "src" / "qt" / "db" / "migrations"


def repo_path(rel: str | Path) -> Path:
    """Resolve a config-supplied relative path (e.g. ``logs``) against the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def visible_files(directory: Path, pattern: str) -> list[Path]:
    """Sorted glob that skips macOS AppleDouble files (``._*``), which exFAT keeps creating."""
    return sorted(p for p in directory.glob(pattern) if not p.name.startswith("._"))
