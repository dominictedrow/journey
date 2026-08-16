"""Run resumable one-factor-at-a-time experiments from para.md.

This runner never writes the production checkpoint and never evaluates the
held-out test split.  Every training variant changes exactly one named factor
from a fixed, seeded production-equivalent control.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from dataset import (BODY_SLOT, COORD_DIM, COORD_LANDMARKS, PART_SLOTS, PART_WIDTHS,
                     PRESENCE_SLOT, SEQUENCE_LENGTH, TOTAL_FEATURE_DIM, VELOCITY_SLOT,
                     LazySignVideoDataset, finalize_sequence, load_vocab, mirror_sequence,
                     resample_sequence)
from ingest import list_video_files


BASE_SEED = 20260814


@dataclass(frozen=True)
class Config:
    name: str
    family: str
    value: str
    seed: int = BASE_SEED
    dropout: float = 0.5
    hidden_size: int = 384
    num_layers: int = 3
    lr: float = 5e-4
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    batch_size: int = 16
    sequence_length: int = 60
    jitter_std: float = 0.02
    temporal_min_keep: float = 0.7
    label_smoothing: float = 0.0
    imbalance: str = "none"
    pooling: str = "baseline"
    backbone: str = "lstm"
    features: str = "all"


BASE = Config("seeded_control", "control", "production defaults + fixed seed")


def experiment_matrix() -> list[Config]:
    specs = [
        ("dropout", "dropout", [("0.2", .2), ("0.3", .3), ("0.4", .4)]),
        ("hidden", "hidden_size", [("256", 256), ("512", 512)]),
        ("depth", "num_layers", [("2", 2), ("4", 4)]),
        ("lr", "lr", [("1e-4", 1e-4), ("3e-4", 3e-4), ("1e-3", 1e-3)]),
        ("optimizer", "optimizer", [("adamw", "adamw")]),
        ("weight_decay", "weight_decay", [("1e-6", 1e-6), ("1e-5", 1e-5), ("1e-3", 1e-3)]),
        ("batch_size", "batch_size", [("32", 32), ("64", 64), ("128", 128)]),
        ("sequence_length", "sequence_length", [("40", 40), ("80", 80), ("100", 100)]),
        ("jitter", "jitter_std", [("0", 0.), ("0.01", .01), ("0.04", .04)]),
        ("temporal_keep", "temporal_min_keep", [("0.5", .5), ("0.85", .85), ("1.0", 1.)]),
        ("label_smoothing", "label_smoothing", [("0.05", .05), ("0.1", .1)]),
        ("imbalance", "imbalance", [("class_weighted", "class_weighted"),
                                      ("balanced_sampler", "balanced_sampler")]),
        ("attention", "pooling", [(x, x) for x in
         ("nonlinear", "multihead", "statistics", "presence", "entropy")]),
        ("backbone", "backbone", [("gru", "gru"), ("transformer", "transformer")]),
        ("feature_ablation", "features", [("no_velocity", "no_velocity"),
                                           ("no_presence", "no_presence"),
                                           ("neither", "neither")]),
        ("seed", "seed", [("20260815", 20260815), ("20260816", 20260816)]),
    ]
    configs = [BASE]
    for family, field, values in specs:
        for label, value in values:
            configs.append(replace(BASE, name=f"{family}_{label}", family=family,
                                   value=label, **{field: value}))
    # Names/metadata are bookkeeping; every run must otherwise differ in one field only.
    ignored = {"name", "family", "value"}
    baseline = asdict(BASE)
    for cfg in configs[1:]:
        changed = [k for k, v in asdict(cfg).items() if k not in ignored and v != baseline[k]]
        assert len(changed) == 1, (cfg.name, changed)
    return configs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _augment(coords: np.ndarray, jitter: float, min_keep: float) -> np.ndarray:
    seq_len = coords.shape[0]
    keep_frac = np.random.uniform(min_keep, 1.0)
    n_keep = max(2, round(seq_len * keep_frac))
    keep_idx = np.sort(np.random.choice(seq_len, n_keep, replace=False))
    coords = resample_sequence(list(coords[keep_idx]), seq_len)
    if np.random.random() < .5:
        coords = mirror_sequence(coords)
    points = coords.reshape(seq_len, COORD_LANDMARKS, 3).copy()
    present = np.abs(points).sum(axis=-1, keepdims=True) > 0
    points += np.random.normal(0, jitter, points.shape).astype(np.float32) * present
    return points.reshape(seq_len, COORD_DIM)


class ExperimentDataset(Dataset):
    def __init__(self, files, vocab, cfg: Config, augment: bool):
        self.base = LazySignVideoDataset(files, vocab, cfg.sequence_length, augment=False)
        self.cfg = cfg
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # LazySignVideoDataset safely falls back to source JSON when a 60-frame
        # cache cannot serve a sequence-length experiment.
        x, y = self.base[idx]
        coords = x.numpy()[:, :COORD_DIM]
        if self.augment:
            coords = _augment(coords, self.cfg.jitter_std, self.cfg.temporal_min_keep)
        features = finalize_sequence(coords)
        if self.cfg.features in ("no_velocity", "neither"):
            features[:, VELOCITY_SLOT] = 0
        if self.cfg.features in ("no_presence", "neither"):
            features[:, PRESENCE_SLOT] = 0
        return torch.from_numpy(features), y


class ExperimentModel(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        h = cfg.hidden_size
        if h % 2:
            raise ValueError("hidden size must be even for bidirectional recurrence")
        self.cfg = cfg
        self.hidden_size = h
        self.last_attention = None
        self.input_proj = nn.Sequential(nn.Linear(TOTAL_FEATURE_DIM, h), nn.LayerNorm(h), nn.ReLU())
        self.dropout = nn.Dropout(cfg.dropout)
        self.activation = nn.ReLU()
        if cfg.backbone == "transformer":
            layer = nn.TransformerEncoderLayer(h, nhead=4, dim_feedforward=h * 4,
                                               dropout=cfg.dropout, batch_first=True,
                                               norm_first=False)
            self.encoder = nn.TransformerEncoder(layer, cfg.num_layers)
            self.lstms = None
        else:
            cell = nn.LSTM if cfg.backbone == "lstm" else nn.GRU
            self.lstms = nn.ModuleList(cell(h, h // 2, batch_first=True, bidirectional=True)
                                       for _ in range(cfg.num_layers))
            self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(cfg.num_layers))
        self.attention = nn.Linear(h, 4 if cfg.pooling == "multihead" else 1,
                                   bias=cfg.pooling not in ("multihead", "nonlinear"))
        if cfg.pooling == "nonlinear":
            self.attention = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, 1, bias=False))
        if cfg.pooling == "presence":
            self.presence_attention = nn.Linear(PRESENCE_SLOT.stop - PRESENCE_SLOT.start, 1, bias=False)
        classifier_width = h * (4 if cfg.pooling == "multihead" else 2 if cfg.pooling == "statistics" else 1)
        self.classifier = nn.Linear(classifier_width, vocab_size)

    def _encode(self, x):
        hidden = self.input_proj(x)
        if self.cfg.backbone == "transformer":
            # Fixed sinusoidal positions are required because attention alone is order-agnostic.
            positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(1)
            div = torch.exp(torch.arange(0, self.hidden_size, 2, device=hidden.device) *
                            (-math.log(10000.0) / self.hidden_size))
            positional = torch.zeros_like(hidden[0])
            positional[:, 0::2] = torch.sin(positions * div)
            positional[:, 1::2] = torch.cos(positions * div)
            return self.encoder(hidden + positional.unsqueeze(0))
        for recurrent, norm in zip(self.lstms, self.norms):
            out, _ = recurrent(hidden)
            hidden = self.dropout(self.activation(norm(out) + hidden))
        return hidden

    def forward(self, x):
        hidden = self._encode(x)
        scores = self.attention(hidden)
        if self.cfg.pooling == "presence":
            scores = scores + self.presence_attention(x[:, :, PRESENCE_SLOT])
        weights = torch.softmax(scores, dim=1)
        self.last_attention = weights
        if self.cfg.pooling == "multihead":
            pooled = torch.einsum("bth,btd->bhd", weights, hidden).flatten(1)
        else:
            mean = (hidden * weights).sum(dim=1)
            if self.cfg.pooling == "statistics":
                var = ((hidden - mean.unsqueeze(1)).square() * weights).sum(dim=1)
                pooled = torch.cat((mean, torch.sqrt(var.clamp_min(1e-5))), dim=-1)
            else:
                pooled = mean
        return self.classifier(pooled)


def make_loader(files, vocab, cfg, train, workers):
    dataset = ExperimentDataset(files, vocab, cfg, augment=train)
    generator = torch.Generator().manual_seed(cfg.seed)
    sampler = None
    shuffle = train
    if train and cfg.imbalance == "balanced_sampler":
        counts = Counter(label for _, label in files)
        weights = torch.tensor([1.0 / counts[label] for _, label in files], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler,
                      num_workers=workers, persistent_workers=workers > 0,
                      pin_memory=True, generator=generator)


def class_weights(files, vocab, device):
    counts = Counter(label for _, label in files)
    total = len(files)
    return torch.tensor([total / (len(vocab) * counts.get(label, 1)) for label in vocab],
                        dtype=torch.float32, device=device)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None, collect=False):
    training = optimizer is not None
    model.train(training)
    amp = scaler is not None and scaler.is_enabled()
    total_loss = correct = count = 0
    confidences, correct_flags = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            targets = y.argmax(1).to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = criterion(logits, targets)
                if training and model.cfg.pooling == "entropy":
                    w = model.last_attention.clamp_min(1e-8)
                    loss -= .01 * (-(w * w.log()).sum(1).mean()) / math.log(w.shape[1])
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.)
                scaler.step(optimizer)
                scaler.update()
            preds = logits.detach().argmax(1)
            total_loss += loss.item() * x.shape[0]
            correct += (preds == targets).sum().item()
            count += x.shape[0]
            if collect:
                confidences.extend(torch.softmax(logits.detach(), 1).max(1).values.cpu().tolist())
                correct_flags.extend((preds == targets).cpu().tolist())
    return total_loss / count, correct / count, confidences, correct_flags


def run_one(cfg, args, train_files, val_files, vocab, device):
    out = args.output_dir / cfg.name
    summary_path = out / "summary.json"
    if summary_path.exists() and not args.force:
        return json.loads(summary_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    set_seed(cfg.seed)
    train_loader = make_loader(train_files, vocab, cfg, True, args.num_workers)
    val_loader = make_loader(val_files, vocab, cfg, False, min(args.num_workers, 2))
    model = ExperimentModel(len(vocab), cfg).to(device)
    weights = class_weights(train_files, vocab, device) if cfg.imbalance == "class_weighted" else None
    train_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()
    opt_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=cfg.lr, betas=(.9, .999), weight_decay=cfg.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=.99)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    rows, best = [], {"best_val_acc": -1., "best_val_loss": float("inf"), "best_epoch": 0}
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, train_criterion, optimizer, scaler)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, device, val_criterion, scaler=scaler)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
               "val_loss": val_loss, "val_acc": val_acc, "elapsed_seconds": time.time() - started}
        rows.append(row)
        print(json.dumps({"experiment": cfg.name, **row}), flush=True)
        if val_acc > best["best_val_acc"] or (val_acc == best["best_val_acc"] and val_loss < best["best_val_loss"]):
            best = {"best_val_acc": val_acc, "best_val_loss": val_loss, "best_epoch": epoch}
            torch.save(model.state_dict(), out / "best.pt")
    torch.save(model.state_dict(), out / "latest.pt")
    with (out / "epochs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summary = {"name": cfg.name, "family": cfg.family, "value": cfg.value, **best,
               "final_val_acc": rows[-1]["val_acc"], "final_val_loss": rows[-1]["val_loss"],
               "parameters": sum(p.numel() for p in model.parameters()),
               "elapsed_seconds": time.time() - started}
    summary_path.write_text(json.dumps(summary, indent=2))
    # Persistent workers are useful within one 15-epoch run, but Windows can
    # otherwise retain them after the local DataLoader goes out of scope.  A
    # sweep must release each pair before constructing the next experiment or
    # worker processes and their shared mappings accumulate across variants.
    for loader in (train_loader, val_loader):
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()
    return summary


def write_aggregate(output_dir, summaries):
    control = next(x for x in summaries if x["name"] == BASE.name)
    for row in summaries:
        row["best_delta_pp"] = 100 * (row["best_val_acc"] - control["best_val_acc"])
        row["final_delta_pp"] = 100 * (row["final_val_acc"] - control["final_val_acc"])
        row["rose"] = row["best_val_acc"] > control["best_val_acc"]
    fields = list(summaries[0].keys())
    with (output_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
    lines = ["| Family | Tuning | Best val acc | Delta (pp) | Epoch 15 val acc | Rose? |",
             "|---|---|---:|---:|---:|:---:|"]
    for x in summaries:
        lines.append(f"| {x['family']} | {x['value']} | {x['best_val_acc']:.2%} | "
                     f"{x['best_delta_pp']:+.2f} | {x['final_val_acc']:.2%} | "
                     f"{'yes' if x['rose'] else 'no'} |")
    (output_dir / "results.md").write_text("\n".join(lines) + "\n")


def add_policy_results(args, summaries, val_files, vocab, device):
    """Evaluate the two non-training high-priority policies on the control."""
    control = next((x for x in summaries if x["name"] == BASE.name), None)
    if control is None:
        return summaries
    restore = {"name": "restore_best_checkpoint", "family": "checkpoint_restore",
               "value": "best instead of latest", "best_val_acc": control["best_val_acc"],
               "best_val_loss": control["best_val_loss"], "best_epoch": control["best_epoch"],
               "final_val_acc": control["best_val_acc"], "final_val_loss": control["best_val_loss"],
               "parameters": control["parameters"], "elapsed_seconds": 0.0}
    summaries = [x for x in summaries if x["name"] != restore["name"]] + [restore]

    rejection_path = args.output_dir / "confidence_rejection.csv"
    if not rejection_path.exists() or args.force:
        loader = make_loader(val_files, vocab, BASE, False, min(args.num_workers, 2))
        model = ExperimentModel(len(vocab), BASE).to(device)
        model.load_state_dict(torch.load(args.output_dir / BASE.name / "best.pt", map_location=device,
                                         weights_only=True))
        _, _, confidences, correct = run_epoch(model, loader, device, nn.CrossEntropyLoss(), collect=True)
        rows = []
        for threshold in (0., .25, .4, .5, .6, .7, .8, .9):
            accepted = [ok for conf, ok in zip(confidences, correct) if conf >= threshold]
            rows.append({"threshold": threshold, "coverage": len(accepted) / len(correct),
                         "selective_accuracy": sum(accepted) / len(accepted) if accepted else ""})
        with rejection_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_landmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/high_priority_tuning/seeded_ofat"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--only", nargs="*", help="experiment names to run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    matrix = experiment_matrix()
    if args.list:
        print("\n".join(c.name for c in matrix)); return
    selected = [c for c in matrix if not args.only or c.name in args.only]
    unknown = set(args.only or ()) - {c.name for c in matrix}
    if unknown:
        raise SystemExit(f"unknown experiments: {sorted(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab = load_vocab()
    train_files = list_video_files(args.data_dir, split="train")
    val_files = list_video_files(args.data_dir, split="val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = {"device": str(device), "epochs": args.epochs, "train_examples": len(train_files),
                "val_examples": len(val_files), "test_split_used": False,
                "note": "sequence-length variants parse source JSON; caches are not rebuilt"}
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    summaries = []
    for cfg in selected:
        print(f"starting {cfg.name}", flush=True)
        summaries.append(run_one(cfg, args, train_files, val_files, vocab, device))
        if any(x["name"] == BASE.name for x in summaries):
            write_aggregate(args.output_dir, summaries)
    summaries = add_policy_results(args, summaries, val_files, vocab, device)
    if any(x["name"] == BASE.name for x in summaries):
        write_aggregate(args.output_dir, summaries)


if __name__ == "__main__":
    main()
