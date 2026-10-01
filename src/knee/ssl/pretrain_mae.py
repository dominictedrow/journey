"""MAE pretraining of SliceViT on the S02 train cache (S03). Resumable; pause with a STOP file.

    python -m knee.ssl.pretrain_mae --config configs/mae_vits.yaml                 # start, or resume if last.pt exists
    python -m knee.ssl.pretrain_mae --config configs/mae_vits.yaml --pilot 300     # timed pilot → pilot.json, no resume
    (pause)  create an empty file <run_dir>/STOP → it saves last.pt and exits; run the same command again to resume.

Run dir (default <cache_root>/runs/<run_name>): config.yaml, log.jsonl, last.pt (full state), encoder_ep<NNN>.pt
(SliceViT weights only: what S07 loads), recon/*.png, pilot.json. Never inside the repo (FAT32, 4 GB file limit).

Schedule: AdamW (β2 0.95), lr = blr · effective_batch / 256 (MAE convention), per-step linear warmup + cosine.
Precision: "auto" = bf16 autocast on the 5070 (decision 2026-09-24), fp16 autocast + loss scaling on the T4; fp32
master weights either way.

Loader (docs/plan/03_performance_plan.md, A3): each dataset item is one series read that yields `windows_per_read`
windows, so per-window file access and Python overhead are paid once per read. The DataLoader batches
batch_size / windows_per_read items and concatenates them. data.gpu_crop moves the random resized crop to the GPU
(mae_data, A4): the workers then only read.
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from knee.data.dataset import CacheReader
from knee.model.slice_vit import SliceViT
from knee.ssl.mae import MAE
from knee.ssl.mae_data import MAEWindowDataset, collate_windows, gpu_resized_crop
from knee.train import common as C
from knee.train.optim import build_adamw, param_groups, set_lr, warmup_cosine

DEFAULTS: dict = {
    "run_name": "mae_default", "seed": 0, "cache_dir": None, "run_dir": None, "device": "auto", "precision": "auto",
    "model": {"img_size": 256, "patch": 16, "dim": 384, "depth": 12, "heads": 6, "tokenizer": "conv_stem",
              "stem_norm": "ln", "pos2d": "sincos", "pool": "gap", "qk_norm": True, "layerscale": 0.1,
              "drop_path": 0.0, "attn_impl": "sdpa", "grad_checkpointing": False},
    "mae": {"mask_ratio": 0.75, "dec_dim": 256, "dec_depth": 4, "dec_heads": 8, "norm_pix": True},
    "data": {"out_px": 256, "windows_per_series": 4, "windows_per_read": 1, "scale": [0.6, 1.0], "ratio": [0.85, 1.18],
             "batch_size": 64, "workers": 6, "limit_studies": 0, "gpu_crop": False},
    "optim": {"blr": 1.5e-4, "weight_decay": 0.05, "betas": [0.9, 0.95], "warmup_epochs": 2, "epochs": 40,
              "min_lr_ratio": 0.0, "clip": 1.0, "accum": 1},
    "ckpt_minutes": 30, "snapshot_every_epochs": 5, "log_every": 50, "recon_every_steps": 2000,
    "probe": {"labels": None, "folds": None, "every_epochs": 5, "fold": 0},
    # torch.compile the whole MAE step (docs/plan/03_performance_plan.md B4). Batches are fixed-shape (drop_last), so
    # one graph; CUDA/Linux only. Exact (same maths, fused kernels).
    "perf": {"compile": False, "compile_mode": None},
}


def merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def build(cfg: dict) -> MAE:
    enc = SliceViT(in_ch=3, **cfg["model"])
    return MAE(enc, **cfg["mae"])


def save_recon(mae: MAE, x: torch.Tensor, path: Path) -> float:
    """Rows: input, masked input, reconstruction with visible patches pasted back (centre slice of each window)."""
    was = mae.training
    mae.eval()
    g = torch.Generator(device=x.device).manual_seed(0)  # the same mask every time → comparable images
    r = mae.reconstruct(x, generator=g)
    mae.train(was)
    rows = [np.concatenate(list((r[k][:, 1].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)), axis=1)
            for k in ("input", "masked", "pasted")]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))
    return float(r["loss"])


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--pilot", type=int, default=0, help="run N optimizer steps, write pilot.json, no resume")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like optim.epochs=10 data.batch_size=32")
    a = ap.parse_args(argv)
    cfg = merge(DEFAULTS, C.load_config(a.config))
    for kv in a.set:
        k, v = kv.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for p in path:
            d = d[p]
        d[last] = yaml.safe_load(v)

    if cfg["cache_dir"] is None or cfg["run_dir"] is None:
        from knee.paths import load_paths
        p = load_paths()
        cfg["cache_dir"] = cfg["cache_dir"] or str(p.cache_root / "cache_v1_train")
        cfg["run_dir"] = cfg["run_dir"] or str(p.cache_root / "runs" / cfg["run_name"])
    run_dir = Path(cfg["run_dir"]) / ("pilot" if a.pilot else "")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = C.JsonlLog(run_dir / "log.jsonl")
    device = C.pick_device(cfg["device"])
    C.seed_everything(cfg["seed"])
    cfg["precision"] = C.resolve_precision(device, cfg["precision"])
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    reader = CacheReader(cfg["cache_dir"])
    uids = reader.study_uids
    if cfg["data"]["limit_studies"]:
        uids = uids[: cfg["data"]["limit_studies"]]
    dcfg = cfg["data"]
    kr = int(dcfg["windows_per_read"])
    ds = MAEWindowDataset(reader, uids, dcfg["out_px"], dcfg["windows_per_series"], scale=tuple(dcfg["scale"]),
                          ratio=tuple(dcfg["ratio"]), seed=cfg["seed"], windows_per_read=kr,
                          gpu_crop=bool(dcfg["gpu_crop"]))
    bs, accum = dcfg["batch_size"], cfg["optim"]["accum"]
    if bs % kr:
        raise SystemExit(f"data.batch_size {bs} must be a multiple of data.windows_per_read {kr}")
    ipb = bs // kr  # dataset items (series reads) per batch
    batches_per_epoch = len(ds) // ipb
    if batches_per_epoch < accum:
        raise SystemExit(f"dataset has {ds.n_windows} windows: fewer than one optimizer step at batch {bs}×{accum}")
    steps_per_epoch = batches_per_epoch // accum
    ocfg = cfg["optim"]
    total_steps = ocfg["epochs"] * steps_per_epoch
    warmup = int(ocfg["warmup_epochs"] * steps_per_epoch)
    base_lr = ocfg["blr"] * bs * accum / 256

    mae = build(cfg).to(device)
    if cfg["perf"]["compile"] and device.type == "cuda":
        from knee.model.compiled import compile_module
        compile_module(mae, cfg["perf"]["compile_mode"])
    opt = build_adamw(param_groups(mae, base_lr, ocfg["weight_decay"]), ocfg["betas"],
                      fused=True if device.type == "cuda" else None)
    scaler = C.make_scaler(device, cfg["precision"])
    state = {"epoch": 0, "batch_in_epoch": 0, "step": 0, "elapsed": 0.0}
    last = run_dir / "last.pt"
    if last.exists() and not a.pilot:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        mae.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        C.set_rng_state(ck["rng"])
        print(f"resumed from {last}: epoch {state['epoch']} batch {state['batch_in_epoch']} step {state['step']}")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "git_sha": C.git_sha()}, sort_keys=False))
    n_params = sum(p.numel() for p in mae.encoder.parameters())
    print(f"{len(ds.n)} series, {ds.n_windows} windows/epoch ({kr} per read), {steps_per_epoch} steps/epoch, "
          f"{total_steps} total, lr {base_lr:.2e}, encoder {n_params / 1e6:.1f}M params, device {device}, "
          f"{cfg['precision']}", flush=True)
    meter = C.GpuMeter(device)

    def save(tag: str) -> None:
        C.save_atomic({"model": mae.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                       "state": state, "rng": C.rng_state(), "cfg": cfg}, run_dir / f"{tag}.pt")

    def check_finite(where: str) -> tuple[float, int]:
        """Checked at log/checkpoint time, not per batch (no GPU sync per step). bf16/fp32: any non-finite loss
        stops the run (state → nan_state.pt, last.pt untouched). fp16: tolerated unless every batch was bad."""
        mean, bad = meter.read()
        if bad and (cfg["precision"] != "fp16" or bad >= meter.n // max(bs, 1)):
            save("nan_state")
            raise SystemExit(f"{bad} non-finite losses before {where} (step {state['step']}); state saved to "
                             f"nan_state.pt, last good state is last.pt")
        return mean, bad

    def checkpoint(tag: str = "last") -> None:
        check_finite("checkpoint")
        save(tag)

    def to_model_input(batch) -> torch.Tensor:
        """A loader batch → uint8 windows [N, 3, out_px, out_px] on the device (gpu_crop: crop + resize here)."""
        if isinstance(batch, torch.Tensor):
            return batch.to(device, non_blocking=True)
        x, boxes = batch
        return gpu_resized_crop(x.to(device, non_blocking=True), boxes.to(device, non_blocking=True), dcfg["out_px"])

    recon_batch = to_model_input(collate_windows([ds[i] for i in range(min(8, len(ds)))]))[:8]
    sampler = C.ResumableRandomSampler(len(ds), seed=cfg["seed"])
    t_ckpt, t_start = time.time(), time.time() - state["elapsed"]
    pilot_t0, pilot_windows, pilot_skip = None, 0, 5
    mae.train()
    done = False
    while state["epoch"] < ocfg["epochs"] and not done:
        ep = state["epoch"]
        ds.set_epoch(ep)
        sampler.set_epoch(ep, start=state["batch_in_epoch"] * ipb)
        loader = DataLoader(ds, batch_size=ipb, sampler=sampler, num_workers=dcfg["workers"], drop_last=True,
                            collate_fn=collate_windows, pin_memory=device.type == "cuda", persistent_workers=False,
                            prefetch_factor=4 if dcfg["workers"] else None)
        t_log, n_log = time.time(), 0
        meter.reset()
        for x in loader:
            if state["batch_in_epoch"] >= batches_per_epoch:
                break
            x = to_model_input(x)
            with C.autocast_ctx(device, cfg["precision"]):
                loss = mae(x)["loss"]
            meter.add(loss, len(x))
            scaler.scale(loss / accum).backward()
            state["batch_in_epoch"] += 1
            n_log += len(x)
            if state["batch_in_epoch"] % accum:
                continue
            lr = set_lr(opt, base_lr, warmup_cosine(state["step"], warmup, total_steps, ocfg["min_lr_ratio"]))
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(mae.parameters(), ocfg["clip"])
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            state["step"] += 1
            state["elapsed"] = time.time() - t_start

            if a.pilot:
                if state["step"] == pilot_skip:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    pilot_t0, pilot_windows = time.time(), 0
                elif pilot_t0 is not None:
                    pilot_windows += bs * accum
            if state["step"] % cfg["log_every"] == 0:
                dt = time.time() - t_log
                mean, bad = check_finite("log")
                log.write(step=state["step"], epoch=ep, loss=mean, lr=lr, grad_norm=float(gnorm), nonfinite=bad,
                          loss_scale=scaler.get_scale() if scaler.is_enabled() else None,
                          windows_per_s=n_log / max(dt, 1e-9), peak_vram_gb=C.cuda_peak_gb())
                print(f"ep {ep} step {state['step']}/{total_steps} loss {mean:.4f} "
                      f"lr {lr:.2e} {n_log / max(dt, 1e-9):.0f} win/s", flush=True)
                t_log, n_log = time.time(), 0
                meter.reset()
            if cfg["recon_every_steps"] and state["step"] % cfg["recon_every_steps"] == 0:
                save_recon(mae, recon_batch, run_dir / "recon" / f"step{state['step']:07d}.png")
            if a.pilot and state["step"] >= a.pilot:
                done = True
                break
            if C.stop_requested(run_dir):
                checkpoint()
                print(f"STOP file found: saved {last} at step {state['step']}; re-run the same command to resume.")
                return {"stopped": True, **state}
            if time.time() - t_ckpt > cfg["ckpt_minutes"] * 60:
                checkpoint()
                t_ckpt = time.time()
        if done:
            break
        # end of epoch
        state["epoch"] += 1
        state["batch_in_epoch"] = 0
        checkpoint()
        t_ckpt = time.time()
        e = state["epoch"]
        if e % cfg["snapshot_every_epochs"] == 0 or e == ocfg["epochs"]:
            C.save_atomic({"encoder": mae.encoder.state_dict(), "model_cfg": cfg["model"], "epoch": e,
                           "step": state["step"]}, run_dir / f"encoder_ep{e:03d}.pt")
            save_recon(mae, recon_batch, run_dir / "recon" / f"ep{e:03d}.png")
            pr = cfg["probe"]
            if pr.get("labels") and pr.get("folds") and e % pr["every_epochs"] == 0:
                import pandas as pd
                from knee.ssl.probe import linear_probe
                labels = pd.read_parquet(pr["labels"]).set_index("StudyInstanceUID")
                folds = pd.read_csv(pr["folds"]).set_index("StudyInstanceUID")
                res = linear_probe(mae.encoder, reader, labels, folds, device, dcfg["out_px"], pr["fold"],
                                   autocast=C.autocast_ctx(device, cfg["precision"]))
                mae.train()
                log.write(step=state["step"], epoch=e, **res)
                print(f"probe after epoch {e}: {res}", flush=True)

    result = {**state}
    if a.pilot:
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - pilot_t0 if pilot_t0 else float("nan")
        wps = pilot_windows / dt if pilot_t0 and dt > 0 else float("nan")
        epoch_h = ds.n_windows / wps / 3600 if wps == wps else float("nan")
        result = {"windows_per_s": wps, "peak_vram_gb": C.cuda_peak_gb(), "windows_per_epoch": ds.n_windows,
                  "windows_per_read": kr, "precision": cfg["precision"],
                  "hours_per_epoch": epoch_h, "planned_epochs": ocfg["epochs"],
                  "hours_total": epoch_h * ocfg["epochs"], "batch_size": bs, "accum": accum,
                  "measured_steps": state["step"] - pilot_skip, "device": str(device),
                  "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None}
        (run_dir / "pilot.json").write_text(yaml.safe_dump(result, sort_keys=False))
        print("PILOT:", result, flush=True)
    elif state["epoch"] >= ocfg["epochs"]:
        C.save_atomic({"encoder": mae.encoder.state_dict(), "model_cfg": cfg["model"], "epoch": state["epoch"],
                       "step": state["step"]}, run_dir / "encoder_final.pt")
        print(f"done: {run_dir / 'encoder_final.pt'}", flush=True)
    return result


if __name__ == "__main__":
    main()
