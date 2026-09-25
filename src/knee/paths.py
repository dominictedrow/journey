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


@dataclass(frozen=True)
class Paths:
    data_root: Path
    cache_root: Path
    audit_out: Path

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
        cfg = yaml.safe_load(f.read_text()) or {}

    def get(key: str) -> Path:
        val = os.environ.get(f"KNEE_{key.upper()}") or cfg.get(key)
        if not val:
            raise KeyError(f"'{key}' not set: add it to {CONFIG_FILE} or set KNEE_{key.upper()}")
        return Path(val)

    return Paths(get("data_root"), get("cache_root"), get("audit_out"))
