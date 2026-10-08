"""Machine-specific paths, read from configs/paths.yaml (gitignored) with env-var overrides.

No drive letters are hard-coded anywhere else in the code base (02 §G).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "configs" / "paths.yaml"

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def _need(val: Path | None, key: str) -> Path:
    if val is None:
        raise KeyError(f"'{key}' not set: add it to {CONFIG_FILE} or set KNEE_{key.upper()}")
    return val


@dataclass(frozen=True)
class Paths:
    """cache_root is required. data_root and audit_out are checked when read, not at load: only the tools that use
    them need them (a Kaggle training run has no audit_out; review 2026-09-30)."""
    _data_root: Path | None
    cache_root: Path
    _audit_out: Path | None = None

    @property
    def data_root(self) -> Path:
        return _need(self._data_root, "data_root")

    @property
    def audit_out(self) -> Path:
        return _need(self._audit_out, "audit_out")

    @property
    def train_csv(self) -> Path:
        return self.data_root / "train.csv"

    @property
    def train_series_csv(self) -> Path:
        return self.data_root / "train_series.csv"

    @property
    def test_series_csv(self) -> Path:
        return self.data_root / "test_series.csv"


def load_paths(config_file: Path | None = None) -> Paths:
    cfg: dict = {}
    f = Path(config_file) if config_file else CONFIG_FILE
    if f.exists():
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}

    def get(key: str, required: bool = True) -> Path | None:
        val = os.environ.get(f"KNEE_{key.upper()}") or cfg.get(key)
        if not val:
            if required:
                raise KeyError(f"'{key}' not set: add it to {CONFIG_FILE} or set KNEE_{key.upper()}")
            return None
        return Path(val)

    return Paths(get("data_root", required=False), get("cache_root"), get("audit_out", required=False))
