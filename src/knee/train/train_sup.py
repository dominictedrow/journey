"""Supervised training on report labels, shared by Track A and Track B (S05, S07). The model is chosen by config.

    python -m knee.train.train_sup --config configs/trackB_vits_mil.yaml [--set folds.fold=1 optim.epochs=12]
    (pause)  create <run_dir>/STOP → saves last.pt and exits; re-run the same command to resume.

Inputs (S04 writes them; the format is fixed here):
- labels parquet: `StudyInstanceUID`, the 12 LABELS columns = soft targets in [0,1] (NaN = unknown), optional
  `w_<label>` columns = per-entry confidence weights (missing → 1).
- folds csv: `StudyInstanceUID`, `fold` (0..4). `folds.fold` = the validation fold; -1 = train on everything, no
  validation (then the last `select.top_k` epochs are averaged).
- gold (optional): the competition `train.csv`; rows with all 12 labels present are the 58 image-verified studies.
  Those in the validation fold are scored separately, with a bootstrap CI. **Never used for selection** (master plan §6).

Per epoch: train (K random windows per series/slot; window/series dropout) → eval the validation fold on dense
windows → OOF predictions + per-finding AUC → keep the weights of the top-k epochs (by report-label val macro AUC).
At the end: average the top-k weights (02 §C15, public "top-3 epoch mean") → final eval → `oof.parquet`,
`metrics.json`, `auc_table.md`, `model_avg.pt`.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from knee.data.dataset import CacheReader, StudyBagDataset, StudyWindowsDataset, collate_bag, worker_init
from knee.paths import LABELS
from knee.train import common as C
from knee.train.losses import RankBank, pairwise_rank_loss, soft_bce
from knee.train.optim import build_adamw, param_groups, set_lr, warmup_cosine
from knee.utils.metrics import auc_table, binarize, bootstrap_macro_auc, df_to_markdown, macro_auc

DEFAULTS: dict = {
    "run_name": "sup_default", "seed": 0, "device": "auto", "precision": "auto", "cache_dir": None, "run_dir": None,
    "model": {"kind": "knee_vit", "cfg": {}},
    "init": {"encoder": None},
    "labels": {"file": None, "lo": 0.4, "hi": 0.6},
    "folds": {"file": None, "fold": 0},
    "gold": {"file": None},
    "data": {"format": "bag", "out_px": 256, "train_windows": 8, "eval_windows": 16, "series_dropout": 0.1,
             "max_series": None, "slot_dropout": 0.0, "batch_size": 1, "eval_batch_size": 1, "workers": 4,
             "limit_studies": 0},
    "optim": {"lr": 3e-4, "weight_decay": 0.05, "betas": [0.9, 0.999], "warmup_frac": 0.1, "epochs": 12,
              "min_lr_ratio": 0.01, "clip": 1.0, "accum": 8, "layer_decay": None, "backbone_lr_mult": None,
              # studies per optimizer step. If set, accum = effective_batch / data.batch_size, so raising batch_size
              # (more studies per forward = bigger matmuls, plan §4 item 3) keeps the optimization recipe
              "effective_batch": None},
    "loss": {"rank_w": 0.05, "rank_bank": 256, "pos_thr": 0.6, "neg_thr": 0.4},
    "select": {"top_k": 3},
    "ckpt_minutes": 30, "log_every": 50,
    # docs/plan/03_performance_plan.md. compile (B4): torch.compile the window encoder at fixed chunk shapes (ragged
    # chunks padded up to `buckets`, default powers of two ≥ min_bucket); CUDA/Linux only. channels_last (B3): Track A's
    # backbone, or Track B's conv stem. Both exact (with Track A's default freeze_bn; without it, BatchNorm sees the padded copies in training).
    "perf": {"compile": False, "compile_mode": None, "buckets": None, "min_bucket": 4, "channels_last": False},
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(path: str | Path, sets: list[str]) -> dict:
    """DEFAULTS ← config file ← `--set a.b=value` overrides (values parsed as YAML)."""
    cfg = merge(DEFAULTS, C.load_config(path))
    for kv in sets:
        k, v = kv.split("=", 1)
        d = cfg
        *path_, last = k.split(".")
        for p in path_:
            d = d[p]
        d[last] = yaml.safe_load(v)
    return cfg


def resolve_dirs(cfg: dict) -> dict:
    """Fill cache_dir / run_dir from configs/paths.yaml when the config leaves them null (run dir = per fold)."""
    if cfg["cache_dir"] is None or cfg["run_dir"] is None:
        from knee.paths import load_paths
        p = load_paths()
        cfg["cache_dir"] = cfg["cache_dir"] or str(p.cache_root / "cache_v1_train")
        cfg["run_dir"] = cfg["run_dir"] or str(p.cache_root / "runs" / cfg["run_name"] / f"fold{int(cfg['folds']['fold'])}")
    return cfg


# What makes a run dir "this run": a different split (fold or folds file) would resume or overwrite another fold's
# weights and OOF (leakage); a different run_name is a different experiment. Everything else (optim.epochs, perf.*)
# may change on resume.
IDENTITY_KEYS = (("folds", "fold"), ("folds", "file"), ("run_name",))
WARN_KEYS = (("labels",), ("model",))


def _get(d: dict, keys: tuple):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _same(keys: tuple, a, b) -> bool:
    if keys == ("folds", "fold"):
        return a is not None and b is not None and int(a) == int(b)
    if keys == ("folds", "file") and a and b and str(a) != str(b):
        return _same_file(Path(a), Path(b))
    return a == b


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _same_file(old: Path, new: Path) -> bool:
    """The same folds file under another spelling (relative vs absolute), as a copy (same bytes), or, when the
    recorded path no longer exists (a Kaggle input re-mounted at another /kaggle/input/<slug>), by file name. The
    file-name fallback warns: it can't prove the split is the same."""
    if old.exists() and new.exists():
        try:
            if os.path.samefile(old, new):
                return True
        except OSError:
            pass
        return _sha1(old) == _sha1(new)
    if not old.exists() and old.name == new.name:
        print(f"WARNING: recorded folds file {old} no longer exists; accepting {new} by file name", flush=True)
        return True
    return False


def check_run_identity(run_dir: Path, cfg: dict) -> None:
    """Refuse (SystemExit) to reuse a run dir written by another fold, split or run_name. The recorded config comes
    from run_dir/config.yaml (written before the first epoch, so it also covers a crash that left only epochs/), else
    from last.pt["cfg"] (every checkpoint stores it). Runs once per process start: one small YAML read."""
    rec = None
    if (run_dir / "config.yaml").exists():
        rec = yaml.safe_load((run_dir / "config.yaml").read_text())
    elif (run_dir / "last.pt").exists():
        rec = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False).get("cfg")
    if not rec:
        return
    for keys in IDENTITY_KEYS:
        old, new = _get(rec, keys), _get(cfg, keys)
        if not _same(keys, old, new):
            raise SystemExit(f"run dir {run_dir} belongs to {'.'.join(keys)}={old!r}, but this run has {new!r}. "
                             f"Resuming it would mix two runs (another fold's weights/OOF = leakage). Use a separate "
                             f"run_dir per fold (leave run_dir null to get <runs>/<run_name>/fold<k>).")
    for keys in WARN_KEYS:
        if _get(rec, keys) != _get(cfg, keys):
            print(f"WARNING: {'.'.join(keys)} differs from the config recorded in {run_dir}; resuming anyway",
                  flush=True)


def build_model(mcfg: dict) -> torch.nn.Module:
    if mcfg["kind"] == "knee_vit":
        from knee.model.knee_vit import KneeViT
        return KneeViT(mcfg.get("cfg"))
    if mcfg["kind"] == "trackA":
        from knee.model.trackA import TrackAModel
        return TrackAModel(mcfg.get("cfg"))
    raise ValueError(f"unknown model kind {mcfg['kind']!r}")


def build_param_groups(model, ocfg: dict) -> list[dict]:
    if ocfg.get("layer_decay") and hasattr(model, "layer_id"):
        return param_groups(model, ocfg["lr"], ocfg["weight_decay"], ocfg["layer_decay"], model.layer_id,
                            model.n_layers)
    if ocfg.get("backbone_lr_mult"):
        m = float(ocfg["backbone_lr_mult"])
        return param_groups(model, ocfg["lr"], ocfg["weight_decay"],
                            lr_mult=lambda n: m if n.startswith("backbone.") else 1.0)
    return param_groups(model, ocfg["lr"], ocfg["weight_decay"])


def load_gold(path: str | Path) -> pd.DataFrame:
    """Rows of train.csv with all 12 labels = the image-verified gold studies (58 in S01)."""
    df = pd.read_csv(path)
    df = df[df[LABELS].notna().all(axis=1)]
    return df.set_index("StudyInstanceUID")[LABELS].astype(float)


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def make_dataset(cfg: dict, reader: CacheReader, uids: list[str], labels: pd.DataFrame | None, train: bool):
    d = cfg["data"]
    if d["format"] == "bag":
        return StudyBagDataset(reader, uids, labels, windows_per_series=d["train_windows"],
                               eval_windows_per_series=d["eval_windows"], train=train, out_px=d["out_px"],
                               series_dropout=d["series_dropout"] if train else 0.0, max_series=d["max_series"],
                               seed=cfg["seed"]), collate_bag
    if d["format"] == "slots":
        return StudyWindowsDataset(reader, uids, labels, windows_per_slot=d["train_windows"] if train
                                   else d["eval_windows"], train=train, out_px=d["out_px"],
                                   slot_dropout=d["slot_dropout"] if train else 0.0, seed=cfg["seed"]), None
    raise ValueError(f"unknown data format {d['format']!r}")


def eager_ctx():
    """Run compiled modules eagerly: an fp32 retry through the in-place-compiled encoder would otherwise recompile
    every bucket shape (tens of seconds each, and it eats the recompile limit). set_stance is torch >= 2.6
    (UNVERIFIED on the Kaggle image: check `hasattr(torch.compiler, "set_stance")` there); older torch just recompiles."""
    if hasattr(torch.compiler, "set_stance"):
        return torch.compiler.set_stance("force_eager")
    return contextlib.nullcontext()


def eval_loader(ds, collate, device, workers: int, batch_size: int = 1, persistent: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate,
                      pin_memory=device.type == "cuda", worker_init_fn=worker_init if workers else None,
                      persistent_workers=persistent and workers > 0)


def split_batch(b: dict) -> list[dict]:
    """A collated batch → one batch of 1 per study (tensors and lists sliced along dim 0). Padding stays, masked."""
    n = len(b["study_uid"])
    return [{k: v[i:i + 1] for k, v in b.items()} for i in range(n)]


@torch.no_grad()
def predict(model, ds, collate, device, precision: str, workers: int, batch_size: int = 1,
            loader: DataLoader | None = None) -> tuple[list[str], np.ndarray, dict]:
    """Studies per forward = batch_size (the output doesn't depend on it: padded windows are never encoded, and the
    head masks them; Track A's BatchNorm uses running statistics in eval). `loader` reuses a (persistent) DataLoader
    over `ds` instead of building one.

    A batch whose reduced-precision output isn't finite (an fp16 overflow) or whose forward raises (e.g. CUDA OOM) is
    re-run in fp32, one study at a time. The retry runs after the except block: inside it, the traceback still holds
    the failed forward's frames and their activations, so empty_cache can't free them (review 2026-09-30). The check
    reads the array already copied to the CPU, so the normal path costs nothing.
    Returns (uids, probs, {"n_retried_fp32": batches retried, "n_nonfinite": studies still non-finite})."""
    model.eval()
    dl = loader if loader is not None else eval_loader(ds, collate, device, workers, batch_size)
    uids, out = [], []
    stats = {"n_retried_fp32": 0, "n_nonfinite": 0}
    for b in dl:
        b = to_device(b, device)
        p, err = None, None
        try:
            with C.autocast_ctx(device, precision):
                logits = model(b)
            p = torch.sigmoid(logits.float()).cpu().numpy()
        except RuntimeError as e:
            if precision == "fp32":
                raise
            err = str(e)
        logits = None
        if precision != "fp32" and (p is None or not np.isfinite(p).all()):
            if err is not None:
                print(f"WARNING: {precision} eval forward failed ({err[:200]}); retrying in fp32, one study at a "
                      f"time", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            stats["n_retried_fp32"] += 1
            with C.autocast_ctx(device, "fp32"), eager_ctx():  # autocast off; training never .half()s the weights
                p = np.concatenate([torch.sigmoid(model(bi).float()).cpu().numpy() for bi in split_batch(b)])
        stats["n_nonfinite"] += int((~np.isfinite(p).all(axis=1)).sum())
        out.append(p)
        uids += list(b["study_uid"])
    return uids, (np.concatenate(out) if out else np.zeros((0, len(LABELS)))), stats


def evaluate(uids, probs, labels: pd.DataFrame, gold: pd.DataFrame | None, lo: float, hi: float,
             n_retried: int = 0, n_boot: int = 1000) -> dict:
    """A study still non-finite after the fp32 retry means the weights misbehave on real inputs: that epoch's
    val_macro_auc is NaN, so top_k_epochs ranks it last and it never gets averaged into model_avg.pt (decision
    2026-09-30, pending Dom's confirmation). The AUC over the finite studies is kept as val_macro_auc_finite, for
    diagnosis only. n_boot: the gold CI is never used for selection, so per-epoch evals use fewer replicates
    (~10 s at 1000)."""
    y = binarize(labels.loc[uids, LABELS].to_numpy(), lo, hi)
    n_bad = int((~np.isfinite(probs).all(axis=1)).sum()) if len(probs) else 0
    auc, n = macro_auc(y, probs)
    res = {"val_macro_auc": auc, "val_n_findings": n, "val_n": len(uids), "val_n_retried_fp32": n_retried,
           "val_n_nonfinite": n_bad}
    if n_bad:
        print(f"WARNING: {n_bad} of {len(uids)} validation studies have non-finite predictions after the fp32 "
              f"retry; val_macro_auc = NaN for this eval (finite-subset AUC {auc:.4f})", flush=True)
        res.update(val_macro_auc=float("nan"), val_macro_auc_finite=auc)
    if gold is not None:
        g = [i for i, u in enumerate(uids) if u in gold.index]
        if g:
            yg = gold.loc[[uids[i] for i in g], LABELS].to_numpy()
            bs = bootstrap_macro_auc(yg, probs[g], n_boot=n_boot)
            res.update({"gold_macro_auc": bs["auc"], "gold_lo": bs["lo"], "gold_hi": bs["hi"], "gold_n": len(g)})
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like folds.fold=1 optim.epochs=3")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config, a.set)
    fold = int(cfg["folds"]["fold"])
    resolve_dirs(cfg)
    run_dir = Path(cfg["run_dir"])
    check_run_identity(run_dir, cfg)  # before anything is written to run_dir
    (run_dir / "epochs").mkdir(parents=True, exist_ok=True)
    log = C.JsonlLog(run_dir / "log.jsonl")
    device = C.pick_device(cfg["device"])
    C.seed_everything(cfg["seed"])
    cfg["precision"] = C.resolve_precision(device, cfg["precision"])  # auto → bf16 (5070) / fp16 (T4) / fp32 (CPU)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cuDNN autotunes each conv shape once. The shapes are bounded even without compile: a chunk holds at most
        # window_chunk windows (Track A trains on 3–15 windows per study, so ≤ 5 shapes), so tuning is a one-off cost.
        torch.backends.cudnn.benchmark = True

    # ---- data
    reader = CacheReader(cfg["cache_dir"])
    labels = pd.read_parquet(cfg["labels"]["file"]).set_index("StudyInstanceUID")
    folds = pd.read_csv(cfg["folds"]["file"]).set_index("StudyInstanceUID")
    gold = load_gold(cfg["gold"]["file"]) if cfg["gold"]["file"] else None
    have = sorted(set(reader.study_uids) & set(labels.index) & set(folds.index))
    if cfg["data"]["limit_studies"]:
        have = have[: cfg["data"]["limit_studies"]]
    tr_uids = [u for u in have if folds.loc[u, "fold"] != fold]
    va_uids = [u for u in have if folds.loc[u, "fold"] == fold] if fold >= 0 else []
    wcols = [f"w_{k}" for k in LABELS]
    weights = (labels[wcols].fillna(1.0).to_numpy(np.float32) if set(wcols) <= set(labels.columns)
               else np.ones((len(labels), len(LABELS)), np.float32))
    w_of = dict(zip(labels.index, weights))
    ds_tr, collate = make_dataset(cfg, reader, tr_uids, labels, train=True)
    ds_va, _ = make_dataset(cfg, reader, va_uids, labels, train=False)

    # ---- model + optimizer
    model = build_model(cfg["model"])
    if cfg["init"]["encoder"]:
        ck = torch.load(cfg["init"]["encoder"], map_location="cpu", weights_only=False)
        missing, unexpected = model.load_encoder(ck["encoder"], strict=True)
        print(f"loaded encoder {cfg['init']['encoder']} (epoch {ck.get('epoch')}); missing {missing}, "
              f"unexpected {unexpected}")
    model.to(device)
    pcfg = cfg["perf"]
    if pcfg["channels_last"] and hasattr(model, "use_channels_last"):
        model.use_channels_last()
    if pcfg["compile"] and device.type == "cuda":
        model.compile_encoder(pcfg["compile_mode"], pcfg["buckets"], pcfg["min_bucket"])
    ocfg, dcfg = cfg["optim"], cfg["data"]
    bs, accum = dcfg["batch_size"], ocfg["accum"]
    if ocfg["effective_batch"]:
        if ocfg["effective_batch"] % bs:
            raise SystemExit(f"optim.effective_batch {ocfg['effective_batch']} must be a multiple of data.batch_size {bs}")
        accum = ocfg["accum"] = ocfg["effective_batch"] // bs
    batches_per_epoch = len(ds_tr) // bs
    if batches_per_epoch < accum:
        raise SystemExit(f"{len(ds_tr)} train studies at batch_size {bs} = {batches_per_epoch} batches per epoch: "
                         f"fewer than one optimizer step (accum {accum}); lower optim.effective_batch / accum")
    steps_per_epoch = batches_per_epoch // accum
    total_steps = ocfg["epochs"] * steps_per_epoch
    warmup = int(ocfg["warmup_frac"] * total_steps)
    groups = build_param_groups(model, ocfg)
    opt = build_adamw(groups, ocfg["betas"], fused=True if device.type == "cuda" else None)
    scaler = C.make_scaler(device, cfg["precision"])
    bank = RankBank(cfg["loss"]["rank_bank"])
    lcfg = cfg["loss"]

    state = {"epoch": 0, "batch_in_epoch": 0, "step": 0, "history": []}
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        C.set_rng_state(ck["rng"])
        if ck.get("rank_bank"):
            bank.load_state_dict(ck["rank_bank"], device)
        print(f"resumed: epoch {state['epoch']} batch {state['batch_in_epoch']}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "git_sha": C.git_sha()}, sort_keys=False))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"fold {fold}: {len(tr_uids)} train / {len(va_uids)} val studies; {steps_per_epoch} steps/epoch; "
          f"{n_params / 1e6:.1f}M params; {len(groups)} param groups; device {device}, {cfg['precision']}",
          flush=True)
    meter = C.GpuMeter(device)

    def check_finite(where: str) -> tuple[float, int]:
        """Losses are checked here (log/checkpoint/epoch end), not per micro-batch, so the GPU is never waited on.
        bf16/fp32: any non-finite loss stops the run before a checkpoint can save broken weights. fp16: the scaler
        skips steps with inf/NaN gradients, so isolated bad micro-batches are tolerated; all bad = stop."""
        mean, bad = meter.read()
        if bad and (cfg["precision"] != "fp16" or bad >= meter.n):
            raise SystemExit(f"{bad} non-finite losses before {where} (epoch {state['epoch']} batch "
                             f"{state['batch_in_epoch']}); last good state is {last}")
        return mean, bad

    def checkpoint() -> None:
        check_finite("checkpoint")
        C.save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                       "state": state, "rng": C.rng_state(), "cfg": cfg, "rank_bank": bank.state_dict()}, last)

    # The epoch travels inside the sampler's indices (encode_epoch), so both loaders are built once and keep their
    # workers: on Windows (spawn) a new loader per epoch re-imports torch and re-pickles the cache index into every
    # worker, for training and again for each eval (review 2026-09-30, bug.md #10).
    sampler = C.ResumableRandomSampler(len(ds_tr), seed=cfg["seed"], encode_epoch=True)
    persistent = dcfg["workers"] > 0
    dl = DataLoader(ds_tr, batch_size=bs, sampler=sampler, num_workers=dcfg["workers"], collate_fn=collate,
                    drop_last=True, pin_memory=device.type == "cuda", persistent_workers=persistent,
                    worker_init_fn=worker_init if dcfg["workers"] else None)
    dl_va = eval_loader(ds_va, collate, device, dcfg["workers"], dcfg["eval_batch_size"], persistent) if va_uids \
        else None
    # Stop each epoch at its last full optimizer step: the trailing batches_per_epoch % accum micro-batches would be
    # forward/backward-ed and then dropped by the zero_grad below.
    used_batches = steps_per_epoch * accum
    t_ckpt = time.time()
    while state["epoch"] < ocfg["epochs"]:
        ep = state["epoch"]
        sampler.set_epoch(ep, start=state["batch_in_epoch"] * bs, num_samples=used_batches * bs)
        model.train()
        # Partial gradients never cross an epoch (each epoch ends on a step, and checkpoints only happen at accum
        # boundaries); this is a guard.
        opt.zero_grad(set_to_none=True)
        t0, run_n = time.time(), 0
        meter.reset()
        for b in dl:
            if state["batch_in_epoch"] >= used_batches:
                break
            w = torch.from_numpy(np.stack([w_of[u] for u in b["study_uid"]]))
            b = to_device(b, device)
            w = w.to(device)
            with C.autocast_ctx(device, cfg["precision"]):
                logits = model(b)
            logits = logits.float()
            loss = soft_bce(logits, b["labels"], b["label_mask"], w)
            if lcfg["rank_w"]:
                rl, rt, rm = bank.get()
                loss = loss + lcfg["rank_w"] * pairwise_rank_loss(logits, b["labels"], b["label_mask"], rl, rt, rm,
                                                                  lcfg["pos_thr"], lcfg["neg_thr"])
                bank.push(logits, b["labels"], b["label_mask"])
            meter.add(loss)
            scaler.scale(loss / accum).backward()
            state["batch_in_epoch"] += 1
            run_n += 1
            if state["batch_in_epoch"] % accum:
                continue
            lr = set_lr(opt, ocfg["lr"], warmup_cosine(state["step"], warmup, total_steps, ocfg["min_lr_ratio"]))
            scaler.unscale_(opt)  # clip the true gradients, not the scaled ones (no-op without fp16)
            torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["clip"])
            scaler.step(opt)      # = opt.step(), skipped if an fp16 gradient overflowed
            scaler.update()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            if state["step"] % cfg["log_every"] == 0:
                mean, bad = check_finite("log")
                log.write(step=state["step"], epoch=ep, loss=mean, lr=lr, nonfinite=bad,
                          loss_scale=scaler.get_scale() if scaler.is_enabled() else None,
                          studies_per_s=run_n * bs / max(time.time() - t0, 1e-9), peak_vram_gb=C.cuda_peak_gb())
                print(f"ep {ep} step {state['step']} loss {mean:.4f} lr {lr:.2e}", flush=True)
                t0, run_n = time.time(), 0
                meter.reset()
            if C.stop_requested(run_dir):
                checkpoint()
                print(f"STOP: saved {last}; re-run the same command to resume.")
                return {"stopped": True}
            if time.time() - t_ckpt > cfg["ckpt_minutes"] * 60:
                checkpoint()
                t_ckpt = time.time()

        # ---- end of epoch: eval, keep top-k weights
        check_finite("end of epoch")
        rec = {"epoch": ep}
        if va_uids:
            uids, probs, st = predict(model, ds_va, collate, device, cfg["precision"], dcfg["workers"],
                                      dcfg["eval_batch_size"], loader=dl_va)
            rec.update(evaluate(uids, probs, labels, gold, cfg["labels"]["lo"], cfg["labels"]["hi"],
                                st["n_retried_fp32"], n_boot=200))
            pd.DataFrame(probs, columns=LABELS).assign(StudyInstanceUID=uids).to_parquet(
                run_dir / "epochs" / f"oof_ep{ep:03d}.parquet", index=False)
        C.save_atomic({"model": model.state_dict(), "epoch": ep}, run_dir / "epochs" / f"ep{ep:03d}.pt")
        state["history"].append(rec)
        log.write(**rec)
        print(f"epoch {ep}: {rec}", flush=True)
        state["epoch"] += 1
        state["batch_in_epoch"] = 0
        checkpoint()
        t_ckpt = time.time()
        # Prune only after last.pt records this epoch: pruning first, then a kill before the checkpoint, would resume
        # with a history whose top-k still names a file that's gone (review 2026-09-30).
        keep = top_k_epochs(state["history"], cfg["select"]["top_k"], bool(va_uids))
        for f in (run_dir / "epochs").glob("ep*.pt"):
            if int(f.stem[2:]) not in keep:
                f.unlink()
        # A deadline (multi_fold --hours) that lands during the eval above must stop here, not one epoch later.
        if state["epoch"] < ocfg["epochs"] and C.stop_requested(run_dir):
            print(f"STOP: saved {last} after epoch {ep}; re-run the same command to resume.")
            return {"stopped": True}

    # ---- final: average the top-k epochs, evaluate, write the outputs
    keep = [e for e in top_k_epochs(state["history"], cfg["select"]["top_k"], bool(va_uids))
            if (run_dir / "epochs" / f"ep{e:03d}.pt").exists()]
    if not keep:
        raise SystemExit(f"no top-k epoch weights left in {run_dir / 'epochs'}")
    states = [torch.load(run_dir / "epochs" / f"ep{e:03d}.pt", map_location="cpu", weights_only=False)["model"]
              for e in keep]
    model.load_state_dict(C.average_state_dicts(states))
    # epochs + run_name: multi_fold reads them to tell a finished fold from one whose optim.epochs was raised
    result = {"fold": fold, "run_name": cfg["run_name"], "epochs": ocfg["epochs"], "top_epochs": keep}
    avg_state, used = model.state_dict(), keep
    if va_uids:
        uids, probs, st = predict(model, ds_va, collate, device, cfg["precision"], dcfg["workers"],
                                  dcfg["eval_batch_size"], loader=dl_va)
        res = evaluate(uids, probs, labels, gold, cfg["labels"]["lo"], cfg["labels"]["hi"], st["n_retried_fp32"])
        # The single best epoch is logged next to the average; if the average is worse (e.g. averaged BatchNorm
        # statistics that don't match the averaged weights), the best epoch is used instead. Its OOF and metrics
        # were already computed at its epoch end, so this costs no extra eval.
        best = top_k_epochs(state["history"], 1, True)[0]
        best_rec = next(r for r in state["history"] if r["epoch"] == best)
        b_auc, a_auc = best_rec.get("val_macro_auc", np.nan), res["val_macro_auc"]
        result.update(best_epoch=best, best_epoch_val_macro_auc=b_auc, avg_val_macro_auc=a_auc)
        if len(keep) > 1 and np.isfinite(b_auc) and not (a_auc >= b_auc):
            print(f"WARNING: top-{len(keep)} average (val AUC {a_auc}) is worse than epoch {best} alone ({b_auc}); "
                  f"using epoch {best}", flush=True)
            best_f = run_dir / "epochs" / f"ep{best:03d}.pt"
            avg_state = torch.load(best_f, map_location="cpu", weights_only=False)["model"]
            used = [best]
            oof = pd.read_parquet(run_dir / "epochs" / f"oof_ep{best:03d}.parquet").set_index("StudyInstanceUID")
            uids, probs = list(oof.index), oof[LABELS].to_numpy()
            res = evaluate(uids, probs, labels, gold, cfg["labels"]["lo"], cfg["labels"]["hi"],
                           int(best_rec.get("val_n_retried_fp32", 0)))
        result.update(res)
        result["used_epochs"] = used
        pd.DataFrame(probs, columns=LABELS).assign(StudyInstanceUID=uids).to_parquet(run_dir / "oof.parquet",
                                                                                     index=False)
        y = binarize(labels.loc[uids, LABELS].to_numpy(), cfg["labels"]["lo"], cfg["labels"]["hi"])
        tab = auc_table(y, probs, "val_auc", nonfinite_pred="ignore")  # evaluate() already counted them
        if gold is not None:
            g = [i for i, u in enumerate(uids) if u in gold.index]
            if g:
                tab["gold_auc"] = auc_table(gold.loc[[uids[i] for i in g], LABELS].to_numpy(), probs[g],
                                            nonfinite_pred="ignore")["auc"]
        (run_dir / "auc_table.md").write_text(df_to_markdown(tab))
    C.save_atomic({"model": avg_state, "epochs": used, "cfg": cfg}, run_dir / "model_avg.pt")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float))
    print("FINAL:", result, flush=True)
    return result


def top_k_epochs(history: list[dict], k: int, by_metric: bool) -> list[int]:
    """Best k epochs by report-label validation macro AUC (never gold), or the last k without validation."""
    if not history:
        return []
    if by_metric:
        ranked = sorted(history, key=lambda r: -np.nan_to_num(r.get("val_macro_auc", np.nan), nan=-1.0))
        return sorted(r["epoch"] for r in ranked[:k])
    return [r["epoch"] for r in history[-k:]]


if __name__ == "__main__":
    main()
