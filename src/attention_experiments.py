"""Compare temporal attention-pooling variants on the signer-disjoint val split.

This is an experiment runner, not an inference entry point. It preserves the
production SignLSTM and writes each variant's artifacts under a separate
output directory.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ExponentialLR

from dataset import PRESENCE_SLOT, SEQUENCE_LENGTH, load_vocab, make_lazy_dataloader
from ingest import list_video_files
from model import SignLSTM


VARIANTS = ("baseline", "nonlinear", "multihead", "statistics", "presence", "entropy")


class AttentionExperimentModel(SignLSTM):
    """The production recurrent backbone with a selectable pooling head."""

    def __init__(self, vocab_size: int, variant: str):
        super().__init__(vocab_size=vocab_size)
        self.variant = variant
        self.last_attention = None

        if variant == "nonlinear":
            self.attention = nn.Sequential(
                nn.Linear(self.hidden_size, 128), nn.Tanh(), nn.Linear(128, 1, bias=False))
        elif variant == "multihead":
            self.num_heads = 4
            self.attention = nn.Linear(self.hidden_size, self.num_heads, bias=False)
            self.classifier = nn.Linear(self.hidden_size * self.num_heads, vocab_size)
        elif variant == "statistics":
            self.classifier = nn.Linear(self.hidden_size * 2, vocab_size)
        elif variant == "presence":
            self.presence_attention = nn.Linear(PRESENCE_SLOT.stop - PRESENCE_SLOT.start, 1, bias=False)
        elif variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden_seq = self.input_proj(x)
        for lstm, norm in zip(self.lstms, self.norms):
            out, _ = lstm(hidden_seq)
            hidden_seq = self.dropout(self.relu(norm(out) + hidden_seq))
        return hidden_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_seq = self._encode(x)
        scores = self.attention(hidden_seq)

        if self.variant == "presence":
            scores = scores + self.presence_attention(x[:, :, PRESENCE_SLOT])

        weights = torch.softmax(scores, dim=1)
        self.last_attention = weights

        if self.variant == "multihead":
            pooled = torch.einsum("bth,btd->bhd", weights, hidden_seq).flatten(1)
        else:
            mean = (hidden_seq * weights).sum(dim=1)
            if self.variant == "statistics":
                variance = ((hidden_seq - mean.unsqueeze(1)).square() * weights).sum(dim=1)
                pooled = torch.cat((mean, torch.sqrt(variance.clamp_min(1e-5))), dim=-1)
            else:
                pooled = mean
        return self.classifier(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None, scaler=None, entropy_strength: float = 0.0):
    training = optimizer is not None
    model.train(training)
    ce = nn.CrossEntropyLoss()
    amp_enabled = scaler is not None and scaler.is_enabled()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            targets = y.argmax(dim=1).to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                loss = ce(logits, targets)
                if training and entropy_strength:
                    weights = model.last_attention
                    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1).mean()
                    loss = loss - entropy_strength * entropy / math.log(weights.shape[1])

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * x.size(0)
            correct += (logits.detach().argmax(dim=-1) == targets).sum().item()
            count += x.size(0)
    return total_loss / count, correct / count


def run_variant(args, variant, train_files, val_files, vocab, device):
    set_seed(args.seed)
    train_loader = make_lazy_dataloader(train_files, vocab, SEQUENCE_LENGTH, args.batch_size,
                                        augment=True, num_workers=args.num_workers)
    val_loader = make_lazy_dataloader(val_files, vocab, SEQUENCE_LENGTH, args.batch_size,
                                      shuffle=False, num_workers=min(args.num_workers, 2))
    model = AttentionExperimentModel(len(vocab), variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = ExponentialLR(optimizer, gamma=0.99)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    entropy_strength = 0.01 if variant == "entropy" else 0.0
    rows = []
    best = {"val_acc": -1.0, "val_loss": float("inf"), "epoch": 0}
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, scaler, entropy_strength)
        val_loss, val_acc = run_epoch(model, val_loader, device, scaler=scaler)
        scheduler.step()
        row = {"variant": variant, "epoch": epoch, "train_loss": train_loss,
               "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
               "elapsed_seconds": time.time() - started}
        rows.append(row)
        print(json.dumps(row), flush=True)

        if val_acc > best["val_acc"] or (val_acc == best["val_acc"] and val_loss < best["val_loss"]):
            best = {"val_acc": val_acc, "val_loss": val_loss, "epoch": epoch}
            torch.save(model.state_dict(), args.output_dir / f"{variant}_best.pt")

    torch.save(model.state_dict(), args.output_dir / f"{variant}_latest.pt")
    with (args.output_dir / f"{variant}_epochs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    best.update(variant=variant, final_val_acc=rows[-1]["val_acc"],
                final_val_loss=rows[-1]["val_loss"], elapsed_seconds=time.time() - started)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw_landmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/attention_experiments"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab()
    train_files = list_video_files(args.data_dir, split="train")
    val_files = list_video_files(args.data_dir, split="val")
    metadata = vars(args).copy()
    metadata.update(device=str(device), train_examples=len(train_files), val_examples=len(val_files))
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in metadata.items()}
    (args.output_dir / "config.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)

    summaries = []
    for variant in args.variants:
        print(f"starting {variant}", flush=True)
        summaries.append(run_variant(args, variant, train_files, val_files, vocab, device))
        with (args.output_dir / "summary.json").open("w") as f:
            json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
