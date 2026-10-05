"""Inference: test study folders → submission.csv (the model side of the S06 notebook; 01 §7, 02 §F).

    study dir ─preprocess_study (the same code as the cache build)─▶ series
              ─bag_sample / slots_sample (the same code as training, train=False)─▶ one batch per arm
              ─model, fp16 autocast on CUDA, fp32 retry on failure / non-finite output─▶ 12 probabilities per arm
    arms combined per finding (probability mean by default, or rank blend) ─▶ submission.csv

Preprocessing runs in DataLoader workers so the CPU (DICOM decode) overlaps the GPU (02 §F). Any study that fails
anywhere gets the fallback row (the train prior if given, else 0.5) and is logged: every test study always gets a row.
Arms run sequentially on one GPU (no DataParallel); a checkpoint is a `model_avg.pt` from train_sup. With several
GPUs (Kaggle's T4 ×2), `predict_dirs_multi` runs one process per GPU over a cost-balanced share of the studies.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from knee.data.dataset import SPAN, bag_sample, collate_bag, slots_sample, worker_init
from knee.data.preprocess import PreprocParams, preprocess_study
from knee.paths import LABELS
from knee.utils.metrics import rank_normalize

log = logging.getLogger(__name__)
SUBMISSION_COLUMNS = ["StudyInstanceUID", *LABELS]


PERF_DEFAULTS = {"half": False, "compile": False, "compile_mode": None, "channels_last": False, "min_bucket": 4,
                 "window_chunk": None, "studies_per_forward": 1}  # the last one is read by notebooks/infer.py


def load_arm(ckpt: str | Path, device, perf: dict | None = None) -> tuple[torch.nn.Module, dict]:
    """A train_sup model_avg.pt → (model in eval mode with inference folds applied, its data config). Never downloads
    weights.

    perf (arms.json "perf", all off by default; docs/plan/03_performance_plan.md):
    - half: the whole model in fp16, run without autocast (B1). Changes numerics: check the OOF AUC before using it.
    - compile: torch.compile the window encoder at fixed chunk shapes (B4). CUDA/Linux only; exact.
    - channels_last: Track A backbone in NHWC (B3). Exact.
    - window_chunk: windows per encoder call at inference (plan §4 item 3: bigger matmuls). Training's value is set by
      its VRAM (activations kept for backward); inference keeps none, so it can be larger. Exact for SliceViT and for
      Track A in eval (BatchNorm uses running statistics)."""
    from knee.model.compiled import load_compile_cache, to_half
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
    if device.type == "cuda" and perf["compile"]:
        # cuDNN autotunes each conv shape once and reuses the fastest kernel. Only with compile: its buckets bound the
        # chunk shapes. Eager, a bag's last chunk can be any size up to window_chunk (64 shapes for Track B), each tuned
        # anew inside the scored runtime (review 2026-10-01). Exact up to float rounding.
        torch.backends.cudnn.benchmark = True
    if perf["window_chunk"]:
        model.window_chunk = int(perf["window_chunk"])
    if perf["half"] and device.type == "cuda":
        to_half(model)
    if perf["channels_last"] and hasattr(model, "use_channels_last"):
        model.use_channels_last()
    if perf["compile"] and device.type == "cuda":
        load_compile_cache()  # once per process; a no-op without $KNEE_COMPILE_CACHE_IN
        model.compile_encoder(perf["compile_mode"], min_bucket=perf["min_bucket"])
    log.info("arm %s: folded %s, perf %s", Path(ckpt).name, folded, perf)
    return model, ck["cfg"]["data"]


def sample_key(data_cfg: dict) -> tuple:
    """What an arm's input depends on. Arms with the same key (e.g. the 5 folds of one recipe) get the same windows,
    so each study's windows are built, sent from the worker and copied to the GPU once, not once per arm (review
    2026-09-30, bug.md #7)."""
    if data_cfg["format"] == "bag":
        return ("bag", data_cfg["train_windows"], data_cfg["eval_windows"], data_cfg["out_px"],
                data_cfg.get("max_series"))
    return ("slots", data_cfg["eval_windows"], data_cfg["out_px"])


def arm_sample(series: list, data_cfg: dict) -> dict:
    """ProcessedSeries list → one study's (uncollated) model input for an arm, built by the training code."""
    pairs = [(s.pixels, {**s.meta, "positions_mm": s.positions_mm, "n_slices": len(s.pixels)}) for s in series]
    rng = np.random.default_rng(0)  # unused with train=False; eval sampling is deterministic
    if data_cfg["format"] == "bag":
        out = bag_sample(pairs, data_cfg["train_windows"], data_cfg["eval_windows"], data_cfg["out_px"], False, rng,
                         SPAN, max_series=data_cfg.get("max_series"))
        b = {k: torch.from_numpy(v) for k, v in out.items()}
        b.update({"labels": torch.zeros(len(LABELS)), "label_mask": torch.zeros(len(LABELS), dtype=torch.bool),
                  "study_uid": ""})
        return b
    out = slots_sample(pairs, data_cfg["eval_windows"], data_cfg["out_px"], False, rng, SPAN)
    return {k: torch.from_numpy(v) for k, v in out.items()}


def collate_arm(samples: list[dict], data_cfg: dict) -> dict:
    """Several studies' arm_sample outputs → one batch (bags padded to the longest; slots have a fixed shape)."""
    if data_cfg["format"] == "bag":
        return collate_bag(samples)
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


def arm_input(series: list, data_cfg: dict) -> dict:
    """One study's model input for an arm (batch of 1)."""
    return collate_arm([arm_sample(series, data_cfg)], data_cfg)


def n_windows(sample: dict) -> int:
    m = sample["window_mask"] if "window_mask" in sample else sample.get("slot_mask")
    if m is None:  # an uncollated bag sample has no mask: every window is real
        return len(sample["x"])
    return int(m.sum())


class TestStudies(Dataset):
    """Worker-side preprocessing: one item = (uid, {sample key: input} or None, error, seconds).

    Only the series some arm can use are decoded: with slots-only arms (Track A), the unslotted series are skipped
    (slot assignment needs only the headers; review 2026-09-30)."""

    def __init__(self, study_dirs: list[Path], series_info: dict, params: PreprocParams, data_cfgs: list[dict]):
        self.dirs, self.info, self.params = study_dirs, series_info, params
        self.cfgs = {}
        for c in data_cfgs:
            self.cfgs.setdefault(sample_key(c), c)
        self.render = "slots" if all(c["format"] == "slots" for c in data_cfgs) else "all"

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i: int):
        d = Path(self.dirs[i])
        t0 = time.time()
        try:
            series, _ = preprocess_study(d, self.info, self.params, render=self.render)
            inputs = {k: arm_sample(series, c) for k, c in self.cfgs.items()}
            if all(n_windows(b) == 0 for b in inputs.values()):
                raise ValueError("no usable windows")
            return d.name, inputs, None, time.time() - t0
        except Exception as e:  # noqa: BLE001 - a bad study must not kill the submission
            return d.name, None, f"{type(e).__name__}: {e}"[:300], time.time() - t0


@torch.no_grad()
def run_arm(model, batch: dict, device) -> np.ndarray:
    """→ [B, 12] probabilities for a batch of B studies (rows may still be non-finite after the retry: the caller
    replaces those). fp16 on CUDA (the T4 has no bf16 tensor cores [W]): autocast, or plain fp16 if the model was loaded with
    perf.half (autocast would bring back the fp32 LayerNorm casts). fp32 retry if it fails or gives NaN/inf."""
    from knee.model.compiled import is_half
    b = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
    if device.type == "cuda":
        half = is_half(model)
        try:
            with torch.autocast("cuda", dtype=torch.float16, enabled=not half):
                p = torch.sigmoid(model(b).float())
            if torch.isfinite(p).all():
                return p.cpu().numpy()
            log.warning("non-finite fp16 output; retrying in fp32")
        except RuntimeError as e:
            log.warning("fp16 forward failed (%s); retrying in fp32", e)
        # empty_cache here, not in the except: there the traceback still holds the failed forward's activations
        torch.cuda.empty_cache()
        if half:  # fp16 → fp32 → fp16 is lossless, so the retry leaves the weights exactly as they were
            model.float()
            try:
                return torch.sigmoid(model(b).float()).cpu().numpy()
            finally:
                model.half()
    return torch.sigmoid(model(b).float()).cpu().numpy()


def combine(arm_probs: list[np.ndarray], weights: list[float] | None, how: str) -> np.ndarray:
    """(n_studies, 12) per arm → one (n_studies, 12). 'mean' of probabilities or 'rank' blend (AUC reads order)."""
    w = np.ones(len(arm_probs)) if weights is None else np.asarray(weights, dtype=np.float64)
    if how == "rank":
        return sum(wi * rank_normalize(p) for wi, p in zip(w, arm_probs)) / w.sum()
    return sum(wi * p for wi, p in zip(w, arm_probs)) / w.sum()


def predict_dirs(study_dirs: list[Path], series_info: dict, arms: list[tuple[torch.nn.Module, dict]], device,
                 params: PreprocParams = PreprocParams(), workers: int = 2, weights: list[float] | None = None,
                 how: str = "mean", fallback: np.ndarray | None = None, studies_per_forward: int = 1
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (predictions: StudyInstanceUID + 12, per-study timing/error log).

    studies_per_forward > 1 runs several preprocessed studies through each arm in one forward (bigger matmuls, fewer
    ragged chunks and launches; plan §4 item 3). The predictions don't depend on it. If a grouped forward fails, its
    studies are retried one at a time, so one bad study can't take its neighbours down to the fallback row."""
    uids, per_arm, timing = predict_raw(study_dirs, series_info, arms, device, params, workers, fallback,
                                        studies_per_forward)
    return combine_raw(uids, per_arm, weights, how), timing


def combine_raw(uids: list[str], per_arm: list[np.ndarray], weights: list[float] | None, how: str) -> pd.DataFrame:
    """Per-arm (n_studies, 12) → the predictions frame. Runs once over *all* studies: a rank blend is only correct
    over the full test set, so the multi-GPU shards return raw per-arm probabilities and are combined here."""
    final = combine(per_arm, weights, how) if uids else np.zeros((0, len(LABELS)))
    pred = pd.DataFrame(final, columns=LABELS)
    pred.insert(0, "StudyInstanceUID", list(uids))
    return pred


def predict_raw(study_dirs: list[Path], series_info: dict, arms: list[tuple[torch.nn.Module, dict]], device,
                params: PreprocParams = PreprocParams(), workers: int = 2, fallback: np.ndarray | None = None,
                studies_per_forward: int = 1) -> tuple[list[str], list[np.ndarray], pd.DataFrame]:
    """predict_dirs without the arm blend → (uids, per arm an (n, 12) array, timing log)."""
    fallback = np.full(len(LABELS), 0.5) if fallback is None else np.asarray(fallback, dtype=np.float64)
    k = max(1, int(studies_per_forward))
    ds = TestStudies(study_dirs, series_info, params, [c for _, c in arms])
    dl = DataLoader(ds, batch_size=None, shuffle=False, num_workers=workers,
                    prefetch_factor=max(2, k) if workers else None, worker_init_fn=worker_init if workers else None)
    # The `non_blocking=True` copies below are only asynchronous from page-locked memory. collate_arm builds new
    # (pageable) tensors, so the batch is pinned after collate; DataLoader(pin_memory=True) would pin the per-study
    # inputs before the stack and leave the copy synchronous (review 2026-10-01).
    pin = torch.device(device).type == "cuda"
    uids, per_arm, rows = [], [[] for _ in arms], []
    keys = [sample_key(c) for _, c in arms]

    def forward(group: list[tuple[str, dict]]) -> list[list[np.ndarray]]:
        """[(uid, {sample key: input})] → per study, per arm, 12 probabilities. One collate + host→device copy per
        distinct sample key, shared by every arm with that key."""
        out = [[] for _ in group]
        on_dev: dict = {}
        for a, (m, cfg) in enumerate(arms):
            if keys[a] not in on_dev:
                batch = collate_arm([inp[keys[a]] for _, inp in group], cfg)
                on_dev[keys[a]] = {kk: (v.pin_memory() if pin else v).to(device, non_blocking=True)
                                   if torch.is_tensor(v) else v for kk, v in batch.items()}
            p = run_arm(m, on_dev[keys[a]], device)
            for i in range(len(group)):
                out[i].append(p[i])
        return out

    def finite_or_fallback(uid: str, probs: list[np.ndarray], err):
        """A study whose output is still non-finite after the fp32 retry gets the fallback row for that arm (and the
        error column says so) instead of reaching write_submission (review 2026-09-30, bug.md #8)."""
        bad = [a for a, p in enumerate(probs) if not np.isfinite(p).all()]
        if not bad:
            return probs, err
        log.warning("study %s: non-finite output from arm(s) %s after the fp32 retry → fallback row for them", uid, bad)
        probs = [fallback if a in bad else p for a, p in enumerate(probs)]
        return probs, (err + "; " if err else "") + f"non-finite output from arm(s) {bad}"

    def flush(group: list[tuple[str, dict, float]]) -> None:
        t0 = time.time()
        probs, errs = None, [None] * len(group)
        try:
            probs = forward([(u, inp) for u, inp, _ in group])
        except Exception as e:  # noqa: BLE001
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if len(group) == 1:
                errs[0] = f"{type(e).__name__}: {e}"[:300]
            else:
                log.warning("grouped forward of %d studies failed (%s); retrying one by one", len(group), e)
                probs = []
                for i, (u, inp, _) in enumerate(group):
                    try:
                        probs.append(forward([(u, inp)])[0])
                    except Exception as e1:  # noqa: BLE001
                        errs[i] = f"{type(e1).__name__}: {e1}"[:300]
                        probs.append(None)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
        dt = (time.time() - t0) / len(group)  # model time, shared equally by the group's studies
        for i, (uid, _, t_pre) in enumerate(group):
            p = probs[i] if probs is not None else None
            if p is None:
                log.warning("study %s → fallback row (%s)", uid, errs[i])
                p = [fallback] * len(arms)
            else:
                p, errs[i] = finite_or_fallback(uid, p, errs[i])
            emit(uid, p, errs[i], t_pre, dt)

    def emit(uid, probs, err, t_pre, t_model) -> None:
        uids.append(uid)
        for a, p in enumerate(probs):
            per_arm[a].append(p)
        rows.append({"StudyInstanceUID": uid, "error": err, "preprocess_s": t_pre, "model_s": t_model})

    group: list[tuple[str, dict, float]] = []
    for uid, inputs, err, t_pre in dl:
        if inputs is None:
            log.warning("study %s → fallback row (%s)", uid, err)
            emit(uid, [fallback] * len(arms), err, t_pre, 0.0)
            continue
        group.append((uid, inputs, t_pre))
        if len(group) == k:
            flush(group)
            group = []
    if group:
        flush(group)
    per_arm = [np.stack(p) if p else np.zeros((0, len(LABELS))) for p in per_arm]
    return uids, per_arm, pd.DataFrame(rows, columns=["StudyInstanceUID", "error", "preprocess_s", "model_s"])


# ---- several GPUs (Kaggle's T4 ×2): one process per GPU, each with the whole model set and its own share of studies.
# Not DataParallel: on the hidden test it ran the host out of RAM (top_solutions_from_notebooks.md §3), and it would
# only split the model work. Here each process also runs its own preprocessing workers.


def shard_studies(study_dirs: list[Path], n: int, cost: dict[str, float] | None = None) -> list[list[int]]:
    """Indices of study_dirs split into n shards of about equal cost (greedy: most expensive study first, onto the
    lightest shard). cost = study uid → relative cost (the series count from test_series.csv is the proxy we have
    before reading any file); missing → 1. Each shard keeps the input order."""
    cost = cost or {}
    order = sorted(range(len(study_dirs)), key=lambda i: (-cost.get(Path(study_dirs[i]).name, 1.0), i))
    shards, load = [[] for _ in range(n)], [0.0] * n
    for i in order:
        j = min(range(n), key=lambda k: (load[k], k))
        shards[j].append(i)
        load[j] += cost.get(Path(study_dirs[i]).name, 1.0)
    return [sorted(sh) for sh in shards]


def _shard_worker(device: str, arm_specs: list[tuple[str, dict]], study_dirs: list[Path], series_info: dict,
                  params: PreprocParams, workers: int, fallback, studies_per_forward: int, out: str) -> None:
    """Child process: load every arm on `device`, predict its studies, pickle (uids, per-arm probs, timing) to `out`.
    Loads the checkpoints itself (a compiled model doesn't cross processes)."""
    import multiprocessing as mp
    import pickle
    if mp.current_process().name != "MainProcess":  # a fresh spawned interpreter has no logging set up
        logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s [{device}] %(message)s", force=True)
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)  # so empty_cache, autocast and Triton all default to this GPU
    t0 = time.time()
    arms = [load_arm(ck, dev, perf) for ck, perf in arm_specs]
    t_load = time.time() - t0
    uids, per_arm, timing = predict_raw(study_dirs, series_info, arms, dev, params, workers, fallback,
                                        studies_per_forward)
    timing["device"] = device
    from knee.model.compiled import save_compile_cache
    save_compile_cache(device)  # no-op without $KNEE_COMPILE_CACHE_OUT or without compiled arms
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"uids": uids, "per_arm": per_arm, "timing": timing, "load_s": t_load}, f)
    os.replace(tmp, out)


def predict_dirs_multi(study_dirs: list[Path], series_info: dict, arm_specs: list[tuple[str, dict]],
                       devices: list[str], work_dir: str | Path, params: PreprocParams = PreprocParams(),
                       workers_per_device: int = 1, weights: list[float] | None = None, how: str = "mean",
                       fallback: np.ndarray | None = None, studies_per_forward: int = 1,
                       cost: dict[str, float] | None = None, deadline: float | None = None
                       ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """predict_dirs over several devices at once → (predictions, timing log with a `device` column, info).

    arm_specs = [(checkpoint path, perf dict)]: each child loads its own copy. The arm blend runs once here over all
    studies, so the output is the same as predict_dirs on one device. If a child dies (CUDA error, host OOM), its
    shard is re-run once in this process on the same device after the others finish; only if that fails too do its
    studies get the fallback row (write_submission fills them). Spawned, not forked: CUDA can't be used in a fork.

    deadline (a time.time() value): a shard still running then is killed and its studies get the fallback row, with
    no re-run, so a hung child can't run the notebook into the 9 h cap and lose the whole submission."""
    import multiprocessing as mp
    import pickle
    fallback = np.full(len(LABELS), 0.5) if fallback is None else np.asarray(fallback, dtype=np.float64)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shards = shard_studies(study_dirs, len(devices), cost)
    ctx = mp.get_context("spawn")
    jobs = []
    for k, (dev, idx) in enumerate(zip(devices, shards)):
        out = work_dir / f"shard{k}.pkl"
        out.unlink(missing_ok=True)
        args = (dev, arm_specs, [study_dirs[i] for i in idx], series_info, params, workers_per_device, fallback,
                studies_per_forward, str(out))
        p = ctx.Process(target=_shard_worker, args=args, name=f"shard{k}")  # not daemonic: it has DataLoader workers
        p.start()
        jobs.append((k, dev, idx, args, out, p))
        log.info("shard %d → %s: %d studies, cost %.0f", k, dev, len(idx),
                 sum((cost or {}).get(Path(study_dirs[i]).name, 1.0) for i in idx))
    info: dict = {"devices": list(devices), "shard_sizes": [len(s) for s in shards], "retried": [], "lost": [],
                  "timed_out": []}
    results = []
    for k, dev, idx, args, out, p in jobs:
        p.join(None if deadline is None else max(0.0, deadline - time.time()))
        if p.is_alive():
            log.error("shard %d on %s still running at the deadline; killing it, its %d studies get the fallback row",
                      k, dev, len(idx))
            p.kill()  # its DataLoader workers exit on their own when their parent dies
            p.join(30)
            info["timed_out"].append(k)
            info["lost"].append(k)
            continue
        if p.exitcode == 0 and out.exists():
            results.append(pickle.loads(out.read_bytes()))
            continue
        if deadline is not None and time.time() > deadline:
            log.error("shard %d on %s exited with %s after the deadline; no re-run, its %d studies get the fallback "
                      "row", k, dev, p.exitcode, len(idx))
            info["lost"].append(k)
            continue
        log.error("shard %d on %s exited with %s; re-running it in the main process", k, dev, p.exitcode)
        info["retried"].append(k)
        try:
            _shard_worker(*args)
            results.append(pickle.loads(out.read_bytes()))
        except Exception as e:  # noqa: BLE001 - the submission must still be written
            log.error("shard %d failed again (%s); its %d studies get the fallback row", k, e, len(idx))
            info["lost"].append(k)
    info["load_s"] = [round(r["load_s"], 2) for r in results]
    uids = [u for r in results for u in r["uids"]]
    per_arm = [np.concatenate([r["per_arm"][a] for r in results]) if results else np.zeros((0, len(LABELS)))
               for a in range(len(arm_specs))]
    timing = pd.concat([r["timing"] for r in results], ignore_index=True) if results else pd.DataFrame(
        columns=["StudyInstanceUID", "error", "preprocess_s", "model_s", "device"])
    return combine_raw(uids, per_arm, weights, how), timing, info


def write_submission(pred: pd.DataFrame, expected_uids: list[str], out: str | Path,
                     fallback: np.ndarray | None = None) -> pd.DataFrame:
    """Exactly one row per expected study, in the expected order, submission column order; missing → fallback."""
    fallback = np.full(len(LABELS), 0.5) if fallback is None else np.asarray(fallback, dtype=np.float64)
    sub = pd.DataFrame({"StudyInstanceUID": list(expected_uids)}).merge(pred, on="StudyInstanceUID", how="left")
    # missing (NaN from the merge) or non-finite (inf/NaN that got past predict_raw) → the whole row is the fallback.
    # Never an assert on the values: that would lose the whole submission after the full run (review 2026-09-30).
    bad = ~np.isfinite(sub[LABELS].to_numpy(dtype=np.float64)).all(axis=1)
    if bad.any():
        log.warning("%d studies without (finite) predictions → fallback", int(bad.sum()))
        sub.loc[bad, LABELS] = fallback
    sub = sub[SUBMISSION_COLUMNS]
    assert list(sub.columns) == SUBMISSION_COLUMNS
    assert sub.StudyInstanceUID.tolist() == list(expected_uids) and sub.StudyInstanceUID.is_unique
    assert np.isfinite(sub[LABELS].to_numpy(dtype=np.float64)).all()  # holds by construction (fallback is finite)
    sub.to_csv(out, index=False)
    return sub
