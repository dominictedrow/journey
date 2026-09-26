"""Run plumbing shared by pretrain_mae.py and train_sup.py: seeding, device/precision, a resumable sampler, atomic
checkpoints, a JSONL logger, weight averaging, and the STOP-file pause.

Pause/resume a long job (HANDOFF §4): create an empty file named STOP in the run dir. The trainer notices it at the
next step, saves `last.pt`, removes STOP and exits. Re-running the same command resumes from `last.pt`.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Sampler

from knee.paths import REPO_ROOT


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # MPS is not used: the Mac is only for CPU tests (CLAUDE.md, compute)


def autocast_ctx(device: torch.device, precision: str):
    """bf16 autocast on CUDA (decision 2026-09-24); fp32 elsewhere. fp16 is for inference on the T4 only."""
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "bf16_cpu" and device.type == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


class ResumableRandomSampler(Sampler[int]):
    """A fresh permutation per epoch from (seed, epoch); `start` skips the first items (mid-epoch resume) without
    loading them."""

    def __init__(self, n: int, seed: int = 0, shuffle: bool = True):
        self.n, self.seed, self.shuffle = n, seed, shuffle
        self.epoch, self.start = 0, 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        self.epoch, self.start = epoch, start

    def __iter__(self):
        order = np.random.default_rng((self.seed, self.epoch)).permutation(self.n) if self.shuffle \
            else np.arange(self.n)
        return iter(order[self.start:].tolist())

    def __len__(self) -> int:
        return self.n - self.start


def save_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # a crash mid-save never leaves a half-written checkpoint


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


class JsonlLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **rec) -> None:
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **rec}
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")


def stop_requested(run_dir: Path) -> bool:
    f = run_dir / "STOP"
    if f.exists():
        f.unlink()
        return True
    return False


def average_state_dicts(states: list[dict]) -> dict:
    """Uniform average of the float tensors (top-k epoch averaging / SWA-style); integer buffers from the first."""
    out = {}
    for k, v in states[0].items():
        if torch.is_floating_point(v):
            out[k] = sum(s[k].float() for s in states) / len(states)
            out[k] = out[k].to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def cuda_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
