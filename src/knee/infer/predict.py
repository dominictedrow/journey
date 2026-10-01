"""Inference: test study folders → submission.csv (the model side of the S06 notebook; 01 §7, 02 §F).

    study dir ─preprocess_study (the same code as the cache build)─▶ series
              ─bag_sample / slots_sample (the same code as training, train=False)─▶ one batch per arm
              ─model, fp16 autocast on CUDA, fp32 retry on failure / non-finite output─▶ 12 probabilities per arm
    arms combined per finding (probability mean by default, or rank blend) ─▶ submission.csv

Preprocessing runs in DataLoader workers so the CPU (DICOM decode) overlaps the GPU (02 §F). Any study that fails
anywhere gets the fallback row (the train prior if given, else 0.5) and is logged: every test study always gets a row.
Arms run sequentially on one GPU (no DataParallel); a checkpoint is a `model_avg.pt` from train_sup.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from knee.data.dataset import SPAN, bag_sample, collate_bag, slots_sample
from knee.data.preprocess import PreprocParams, preprocess_study
from knee.paths import LABELS
from knee.utils.metrics import rank_normalize

log = logging.getLogger(__name__)
SUBMISSION_COLUMNS = ["StudyInstanceUID", *LABELS]


PERF_DEFAULTS = {"half": False, "compile": False, "compile_mode": None, "channels_last": False, "min_bucket": 4}


def load_arm(ckpt: str | Path, device, perf: dict | None = None) -> tuple[torch.nn.Module, dict]:
    """A train_sup model_avg.pt → (model in eval mode with inference folds applied, its data config). Never downloads
    weights.

    perf (arms.json "perf", all off by default; docs/plan/03_performance_plan.md):
    - half: the whole model in fp16, run without autocast (B1). Changes numerics: check the OOF AUC before using it.
    - compile: torch.compile the window encoder at fixed chunk shapes (B4). CUDA/Linux only; exact.
    - channels_last: Track A backbone in NHWC (B3). Exact."""
    from knee.model.compiled import to_half
    from knee.model.fuse import prepare_for_inference
    from knee.train.train_sup import build_model
    perf = {**PERF_DEFAULTS, **(perf or {})}
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = ck["cfg"]["model"]
    if mcfg["kind"] == "trackA":
        mcfg = {**mcfg, "cfg": {**mcfg.get("cfg", {}), "pretrained": False, "grad_checkpointing": False}}
    model = build_model(mcfg)
    model.load_state_dict(ck["model"])
    folded = prepare_for_inference(model)  # exact LayerScale fold (conv+BN off: slower in eager on the T4)
    model.to(device)
    if perf["half"] and device.type == "cuda":
        to_half(model)
    if perf["channels_last"] and hasattr(model, "use_channels_last"):
        model.use_channels_last()
    if perf["compile"] and device.type == "cuda":
        model.compile_encoder(perf["compile_mode"], min_bucket=perf["min_bucket"])
    log.info("arm %s: folded %s, perf %s", Path(ckpt).name, folded, perf)
    return model, ck["cfg"]["data"]


def arm_input(series: list, data_cfg: dict) -> dict:
    """ProcessedSeries list → one study's model input for an arm (batch of 1), built by the training code."""
    pairs = [(s.pixels, {**s.meta, "positions_mm": s.positions_mm, "n_slices": len(s.pixels)}) for s in series]
    rng = np.random.default_rng(0)  # unused with train=False; eval sampling is deterministic
    if data_cfg["format"] == "bag":
        out = bag_sample(pairs, data_cfg["train_windows"], data_cfg["eval_windows"], data_cfg["out_px"], False, rng,
                         SPAN, max_series=data_cfg.get("max_series"))
        b = {k: torch.from_numpy(v) for k, v in out.items()}
        b.update({"labels": torch.zeros(len(LABELS)), "label_mask": torch.zeros(len(LABELS), dtype=torch.bool),
                  "study_uid": ""})
        return collate_bag([b])
    out = slots_sample(pairs, data_cfg["eval_windows"], data_cfg["out_px"], False, rng, SPAN)
    return {k: torch.from_numpy(v)[None] for k, v in out.items()}


def n_windows(batch: dict) -> int:
    return int(batch["window_mask"].sum()) if "window_mask" in batch else int(batch["slot_mask"].sum())


class TestStudies(Dataset):
    """Worker-side preprocessing: one item = (uid, per-arm inputs or None, error, seconds)."""

    def __init__(self, study_dirs: list[Path], series_info: dict, params: PreprocParams, data_cfgs: list[dict]):
        self.dirs, self.info, self.params, self.data_cfgs = study_dirs, series_info, params, data_cfgs

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i: int):
        d = Path(self.dirs[i])
        t0 = time.time()
        try:
            series, _ = preprocess_study(d, self.info, self.params)
            inputs = [arm_input(series, c) for c in self.data_cfgs]
            if all(n_windows(b) == 0 for b in inputs):
                raise ValueError("no usable windows")
            return d.name, inputs, None, time.time() - t0
        except Exception as e:  # noqa: BLE001 - a bad study must not kill the submission
            return d.name, None, f"{type(e).__name__}: {e}"[:300], time.time() - t0


@torch.no_grad()
def run_arm(model, batch: dict, device) -> np.ndarray:
    """fp16 on CUDA (the T4 has no bf16 tensor cores [W]): autocast, or plain fp16 if the model was loaded with
    perf.half (autocast would bring back the fp32 LayerNorm casts). fp32 retry if it fails or gives NaN/inf."""
    from knee.model.compiled import is_half
    b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    if device.type == "cuda":
        half = is_half(model)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=not half):
                p = torch.sigmoid(model(b).float())
            if torch.isfinite(p).all():
                return p[0].cpu().numpy()
            log.warning("non-finite fp16 output; retrying in fp32")
        except RuntimeError as e:
            log.warning("fp16 forward failed (%s); retrying in fp32", e)
            torch.cuda.empty_cache()
        if half:  # fp16 → fp32 → fp16 is lossless, so the retry leaves the weights exactly as they were
            model.float()
            try:
                return torch.sigmoid(model(b).float())[0].cpu().numpy()
            finally:
                model.half()
    return torch.sigmoid(model(b).float())[0].cpu().numpy()


def combine(arm_probs: list[np.ndarray], weights: list[float] | None, how: str) -> np.ndarray:
    """(n_studies, 12) per arm → one (n_studies, 12). 'mean' of probabilities or 'rank' blend (AUC reads order)."""
    w = np.ones(len(arm_probs)) if weights is None else np.asarray(weights, dtype=np.float64)
    if how == "rank":
        return sum(wi * rank_normalize(p) for wi, p in zip(w, arm_probs)) / w.sum()
    return sum(wi * p for wi, p in zip(w, arm_probs)) / w.sum()


def predict_dirs(study_dirs: list[Path], series_info: dict, arms: list[tuple[torch.nn.Module, dict]], device,
                 params: PreprocParams = PreprocParams(), workers: int = 2, weights: list[float] | None = None,
                 how: str = "mean", fallback: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (predictions: StudyInstanceUID + 12, per-study timing/error log)."""
    fallback = np.full(len(LABELS), 0.5) if fallback is None else np.asarray(fallback, dtype=np.float64)
    ds = TestStudies(study_dirs, series_info, params, [c for _, c in arms])
    dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers,
                    prefetch_factor=2 if workers else None)
    uids, per_arm, rows = [], [[] for _ in arms], []
    for uid, inputs, err, t_pre in dl:
        t0 = time.time()
        probs = None
        if inputs is not None:
            try:
                probs = [run_arm(m, b, device) for (m, _), b in zip(arms, inputs)]
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"[:300]
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        if probs is None:
            log.warning("study %s → fallback row (%s)", uid, err)
            probs = [fallback] * len(arms)
        uids.append(uid)
        for k, p in enumerate(probs):
            per_arm[k].append(p)
        rows.append({"StudyInstanceUID": uid, "error": err, "preprocess_s": t_pre, "model_s": time.time() - t0})
    final = combine([np.stack(p) for p in per_arm], weights, how) if uids else np.zeros((0, len(LABELS)))
    pred = pd.DataFrame(final, columns=LABELS)
    pred.insert(0, "StudyInstanceUID", uids)
    return pred, pd.DataFrame(rows)


def write_submission(pred: pd.DataFrame, expected_uids: list[str], out: str | Path,
                     fallback: np.ndarray | None = None) -> pd.DataFrame:
    """Exactly one row per expected study, in the expected order, submission column order; missing → fallback."""
    fallback = np.full(len(LABELS), 0.5) if fallback is None else np.asarray(fallback, dtype=np.float64)
    sub = pd.DataFrame({"StudyInstanceUID": list(expected_uids)}).merge(pred, on="StudyInstanceUID", how="left")
    missing = sub[LABELS].isna().any(axis=1)
    if missing.any():
        log.warning("%d studies without predictions → fallback", int(missing.sum()))
        sub.loc[missing, LABELS] = fallback
    sub = sub[SUBMISSION_COLUMNS]
    assert list(sub.columns) == SUBMISSION_COLUMNS
    assert sub.StudyInstanceUID.tolist() == list(expected_uids) and sub.StudyInstanceUID.is_unique
    assert np.isfinite(sub[LABELS].to_numpy()).all()
    sub.to_csv(out, index=False)
    return sub
