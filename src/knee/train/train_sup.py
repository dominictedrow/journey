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
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from knee.data.dataset import CacheReader, StudyBagDataset, StudyWindowsDataset, collate_bag
from knee.paths import LABELS
from knee.train import common as C
from knee.train.losses import RankBank, pairwise_rank_loss, soft_bce
from knee.train.optim import build_adamw, param_groups, set_lr, warmup_cosine
from knee.utils.metrics import auc_table, binarize, bootstrap_macro_auc, df_to_markdown, macro_auc

DEFAULTS: dict = {
    "run_name": "sup_default", "seed": 0, "device": "auto", "precision": "bf16", "cache_dir": None, "run_dir": None,
    "model": {"kind": "knee_vit", "cfg": {}},
    "init": {"encoder": None},
    "labels": {"file": None, "lo": 0.4, "hi": 0.6},
    "folds": {"file": None, "fold": 0},
    "gold": {"file": None},
    "data": {"format": "bag", "out_px": 256, "train_windows": 8, "eval_windows": 16, "series_dropout": 0.1,
             "max_series": None, "slot_dropout": 0.0, "batch_size": 1, "workers": 4, "limit_studies": 0},
    "optim": {"lr": 3e-4, "weight_decay": 0.05, "betas": [0.9, 0.999], "warmup_frac": 0.1, "epochs": 12,
              "min_lr_ratio": 0.01, "clip": 1.0, "accum": 8, "layer_decay": None, "backbone_lr_mult": None},
    "loss": {"rank_w": 0.05, "rank_bank": 256, "pos_thr": 0.6, "neg_thr": 0.4},
    "select": {"top_k": 3},
    "ckpt_minutes": 30, "log_every": 50,
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


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


@torch.no_grad()
def predict(model, ds, collate, device, precision: str, workers: int) -> tuple[list[str], np.ndarray]:
    model.eval()
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers, collate_fn=collate,
                    pin_memory=device.type == "cuda")
    uids, out = [], []
    for b in dl:
        b = to_device(b, device)
        with C.autocast_ctx(device, precision):
            logits = model(b)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
        uids += list(b["study_uid"])
    return uids, np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def evaluate(uids, probs, labels: pd.DataFrame, gold: pd.DataFrame | None, lo: float, hi: float) -> dict:
    y = binarize(labels.loc[uids, LABELS].to_numpy(), lo, hi)
    auc, n = macro_auc(y, probs)
    res = {"val_macro_auc": auc, "val_n_findings": n, "val_n": len(uids)}
    if gold is not None:
        g = [i for i, u in enumerate(uids) if u in gold.index]
        if g:
            yg = gold.loc[[uids[i] for i in g], LABELS].to_numpy()
            bs = bootstrap_macro_auc(yg, probs[g], n_boot=1000)
            res.update({"gold_macro_auc": bs["auc"], "gold_lo": bs["lo"], "gold_hi": bs["hi"], "gold_n": len(g)})
    return res


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], help="overrides like folds.fold=1 optim.epochs=3")
    a = ap.parse_args(argv)
    cfg = merge(DEFAULTS, C.load_config(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for p in path:
            d = d[p]
        d[last] = yaml.safe_load(v)
    fold = int(cfg["folds"]["fold"])
    if cfg["cache_dir"] is None or cfg["run_dir"] is None:
        from knee.paths import load_paths
        p = load_paths()
        cfg["cache_dir"] = cfg["cache_dir"] or str(p.cache_root / "cache_v1_train")
        cfg["run_dir"] = cfg["run_dir"] or str(p.cache_root / "runs" / cfg["run_name"] / f"fold{fold}")
    run_dir = Path(cfg["run_dir"])
    (run_dir / "epochs").mkdir(parents=True, exist_ok=True)
    log = C.JsonlLog(run_dir / "log.jsonl")
    device = C.pick_device(cfg["device"])
    C.seed_everything(cfg["seed"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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
    ocfg, dcfg = cfg["optim"], cfg["data"]
    bs, accum = dcfg["batch_size"], ocfg["accum"]
    batches_per_epoch = len(ds_tr) // bs
    steps_per_epoch = max(1, batches_per_epoch // accum)
    total_steps = ocfg["epochs"] * steps_per_epoch
    warmup = int(ocfg["warmup_frac"] * total_steps)
    groups = build_param_groups(model, ocfg)
    opt = build_adamw(groups, ocfg["betas"], fused=True if device.type == "cuda" else None)
    bank = RankBank(cfg["loss"]["rank_bank"])
    lcfg = cfg["loss"]

    state = {"epoch": 0, "batch_in_epoch": 0, "step": 0, "history": []}
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        C.set_rng_state(ck["rng"])
        print(f"resumed: epoch {state['epoch']} batch {state['batch_in_epoch']}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "git_sha": C.git_sha()}, sort_keys=False))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"fold {fold}: {len(tr_uids)} train / {len(va_uids)} val studies; {steps_per_epoch} steps/epoch; "
          f"{n_params / 1e6:.1f}M params; {len(groups)} param groups; device {device}", flush=True)

    def checkpoint() -> None:
        C.save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "rng": C.rng_state(),
                       "cfg": cfg}, last)

    sampler = C.ResumableRandomSampler(len(ds_tr), seed=cfg["seed"])
    t_ckpt = time.time()
    while state["epoch"] < ocfg["epochs"]:
        ep = state["epoch"]
        ds_tr.set_epoch(ep)
        sampler.set_epoch(ep, start=state["batch_in_epoch"] * bs)
        dl = DataLoader(ds_tr, batch_size=bs, sampler=sampler, num_workers=dcfg["workers"], collate_fn=collate,
                        drop_last=True, pin_memory=device.type == "cuda")
        model.train()
        t0, run_loss, run_n = time.time(), 0.0, 0
        for b in dl:
            if state["batch_in_epoch"] >= batches_per_epoch:
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
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at epoch {ep} batch {state['batch_in_epoch']}")
            (loss / accum).backward()
            state["batch_in_epoch"] += 1
            run_loss, run_n = run_loss + loss.item(), run_n + 1
            if state["batch_in_epoch"] % accum:
                continue
            lr = set_lr(opt, ocfg["lr"], warmup_cosine(state["step"], warmup, total_steps, ocfg["min_lr_ratio"]))
            torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg["clip"])
            opt.step()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            if state["step"] % cfg["log_every"] == 0:
                log.write(step=state["step"], epoch=ep, loss=run_loss / max(run_n, 1), lr=lr,
                          studies_per_s=run_n * bs / max(time.time() - t0, 1e-9), peak_vram_gb=C.cuda_peak_gb())
                print(f"ep {ep} step {state['step']} loss {run_loss / max(run_n, 1):.4f} lr {lr:.2e}", flush=True)
                t0, run_loss, run_n = time.time(), 0.0, 0
            if C.stop_requested(run_dir):
                checkpoint()
                print(f"STOP: saved {last}; re-run the same command to resume.")
                return {"stopped": True}
            if time.time() - t_ckpt > cfg["ckpt_minutes"] * 60:
                checkpoint()
                t_ckpt = time.time()

        # ---- end of epoch: eval, keep top-k weights
        rec = {"epoch": ep}
        if va_uids:
            uids, probs = predict(model, ds_va, collate, device, cfg["precision"], dcfg["workers"])
            rec.update(evaluate(uids, probs, labels, gold, cfg["labels"]["lo"], cfg["labels"]["hi"]))
            pd.DataFrame(probs, columns=LABELS).assign(StudyInstanceUID=uids).to_parquet(
                run_dir / "epochs" / f"oof_ep{ep:03d}.parquet", index=False)
        C.save_atomic({"model": model.state_dict(), "epoch": ep}, run_dir / "epochs" / f"ep{ep:03d}.pt")
        state["history"].append(rec)
        log.write(**rec)
        print(f"epoch {ep}: {rec}", flush=True)
        keep = top_k_epochs(state["history"], cfg["select"]["top_k"], bool(va_uids))
        for f in (run_dir / "epochs").glob("ep*.pt"):
            if int(f.stem[2:]) not in keep and int(f.stem[2:]) != ep:
                f.unlink()
        state["epoch"] += 1
        state["batch_in_epoch"] = 0
        checkpoint()
        t_ckpt = time.time()

    # ---- final: average the top-k epochs, evaluate, write the outputs
    keep = top_k_epochs(state["history"], cfg["select"]["top_k"], bool(va_uids))
    states = [torch.load(run_dir / "epochs" / f"ep{e:03d}.pt", map_location="cpu", weights_only=False)["model"]
              for e in keep]
    model.load_state_dict(C.average_state_dicts(states))
    C.save_atomic({"model": model.state_dict(), "epochs": keep, "cfg": cfg}, run_dir / "model_avg.pt")
    result = {"fold": fold, "top_epochs": keep}
    if va_uids:
        uids, probs = predict(model, ds_va, collate, device, cfg["precision"], dcfg["workers"])
        result.update(evaluate(uids, probs, labels, gold, cfg["labels"]["lo"], cfg["labels"]["hi"]))
        pd.DataFrame(probs, columns=LABELS).assign(StudyInstanceUID=uids).to_parquet(run_dir / "oof.parquet",
                                                                                     index=False)
        y = binarize(labels.loc[uids, LABELS].to_numpy(), cfg["labels"]["lo"], cfg["labels"]["hi"])
        tab = auc_table(y, probs, "val_auc")
        if gold is not None:
            g = [i for i, u in enumerate(uids) if u in gold.index]
            if g:
                tab["gold_auc"] = auc_table(gold.loc[[uids[i] for i in g], LABELS].to_numpy(), probs[g])["auc"]
        (run_dir / "auc_table.md").write_text(df_to_markdown(tab))
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
