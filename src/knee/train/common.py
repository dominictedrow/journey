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


def resolve_precision(device: torch.device, precision: str) -> str:
    """"auto" → bf16 on GPUs with native bf16 tensor cores (compute capability ≥ 8.0: the RTX 5070), fp16 below that
    (the Kaggle T4 is sm_75: no bf16 tensor cores, and torch may still report bf16 as "supported" through emulation),
    fp32 on CPU. Explicit values pass through."""
    if precision != "auto":
        return precision
    if device.type != "cuda":
        return "fp32"
    return "bf16" if torch.cuda.get_device_capability(device)[0] >= 8 else "fp16"


def autocast_ctx(device: torch.device, precision: str):
    """bf16 autocast on CUDA (decision 2026-09-24); fp16 autocast for the T4 (pair it with make_scaler); fp32 elsewhere.
    `precision` should already be resolved (resolve_precision)."""
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    if precision == "bf16_cpu" and device.type == "cpu":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return torch.autocast(device.type, enabled=False)


def make_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler:
    """Loss scaling, needed for fp16 only: fp16 gradients underflow to 0 below ~6e-8, so the loss is multiplied by a
    large factor before backward and the gradients divided back before the optimizer step. A step whose gradients
    overflowed (inf/NaN) is skipped and the factor halved. bf16 has fp32's exponent range and needs none, so the
    scaler is disabled there (every call is then a pass-through)."""
    return torch.amp.GradScaler(device.type, enabled=precision == "fp16" and device.type == "cuda")


class GpuMeter:
    """Running sums kept on the device, so the training loop never waits for the GPU (no .item() per micro-batch).
    `read()` syncs once; call it at log/checkpoint time only."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.loss_sum = torch.zeros((), device=self.device, dtype=torch.float64)
        self.n_bad = torch.zeros((), device=self.device, dtype=torch.int64)
        self.n = 0

    def add(self, loss: torch.Tensor, weight: int = 1) -> None:
        ok = torch.isfinite(loss.detach())
        self.loss_sum += torch.where(ok, loss.detach().double() * weight, 0.0)
        self.n_bad += (~ok).long()
        self.n += weight

    def read(self) -> tuple[float, int]:
        """(mean finite loss, number of non-finite micro-batches) since the last reset."""
        return float(self.loss_sum) / max(self.n, 1), int(self.n_bad)


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


class ResumableRandomSampler(Sampler[int]):
    """A fresh permutation per epoch from (seed, epoch); `start` skips the first items (mid-epoch resume) without
    loading them; `num_samples` stops the epoch early (train_sup: at the last full optimizer step, so no micro-batch
    is computed and then thrown away).

    encode_epoch=True yields `epoch * n + index` instead of `index`. The datasets decode it
    (knee.data.dataset.decode_index), so the epoch reaches the DataLoader workers through the index itself. That lets
    the loader keep persistent workers: a worker's dataset copy never sees `set_epoch`, and doesn't need to (review
    2026-09-30, bug.md #10)."""

    def __init__(self, n: int, seed: int = 0, shuffle: bool = True, encode_epoch: bool = False):
        self.n, self.seed, self.shuffle, self.encode_epoch = n, seed, shuffle, encode_epoch
        self.epoch, self.start, self.num_samples = 0, 0, None

    def set_epoch(self, epoch: int, start: int = 0, num_samples: int | None = None) -> None:
        self.epoch, self.start, self.num_samples = epoch, start, num_samples

    def _stop(self) -> int:
        return self.n if self.num_samples is None else min(self.n, self.num_samples)

    def __iter__(self):
        order = np.random.default_rng((self.seed, self.epoch)).permutation(self.n) if self.shuffle \
            else np.arange(self.n)
        order = order[self.start:self._stop()]
        if self.encode_epoch:
            order = order + self.epoch * self.n
        return iter(order.tolist())

    def __len__(self) -> int:
        return max(0, self._stop() - self.start)


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
    """Uniform average of the float tensors (top-k epoch averaging / SWA-style); integer buffers from the first.
    BatchNorm running statistics are averaged too, which is only exact if they didn't change between the epochs
    (Track A freezes BN for that reason; trackA.py)."""
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
