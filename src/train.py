"""Train SignLSTM from pre-extracted MediaPipe landmarks in data/raw_landmarks/,
not a live MediaPipe feed. train/val are read from that one directory via
ingest.load_records's split filter, using the train/val/test assignment
split_dataset.py already wrote to data/split.json.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from dataset import SEQUENCE_LENGTH, load_vocab, make_dataloader
from ingest import load_records
from model import SignLSTM, build_model

BATCH_SIZE = 16
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw_landmarks"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "models" / "checkpoint.pt"


def precision_recall_macro(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> tuple[float, float]:
    """Macro precision/recall from class-index predictions and targets.

    No sklearn in this environment -- per-class true/false positive/negative
    counts, averaged unweighted across classes, is the same computation.
    """
    precisions, recalls = [], []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return sum(precisions) / num_classes, sum(recalls) / num_classes


def ctc_greedy_predict(log_probs: torch.Tensor, blank: int) -> torch.Tensor:
    """Collapse each sequence's per-timestep argmax into one predicted word.

    Standard CTC decode drops blanks and repeats; since every sequence here
    has exactly one true word, the most common surviving class is taken as
    the prediction.
    """
    frame_preds = log_probs.argmax(dim=-1)  # (batch, seq_len)
    predictions = []
    for row in frame_preds:
        non_blank = row[row != blank]
        predictions.append(non_blank.mode().values if non_blank.numel() else torch.tensor(blank))
    return torch.stack(predictions)


def run_epoch(model: SignLSTM, loader, vocab_size: int, device: torch.device, optimizer=None,
              clip_norm: float | None = None):
    """One pass over loader. Trains if optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)
    ctc_loss = nn.CTCLoss(blank=vocab_size, zero_infinity=True)

    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            targets = y.argmax(dim=1)

            log_probs = model(x)  # (batch, seq_len, vocab_size + 1)
            batch_size, seq_len, _ = log_probs.shape
            input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
            target_lengths = torch.ones(batch_size, dtype=torch.long)

            loss = ctc_loss(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if clip_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()

            total_loss += loss.item() * batch_size
            all_preds.append(ctc_greedy_predict(log_probs.detach(), blank=vocab_size))
            all_targets.append(targets)

    preds, targets = torch.cat(all_preds), torch.cat(all_targets)
    accuracy = (preds == targets).float().mean().item()
    precision, recall = precision_recall_macro(preds, targets, vocab_size)
    return total_loss / len(targets), accuracy, precision, recall


def train(data_dir: Path, epochs: int, lr: float, lr_scaling: float, momentum: float,
          clip_norm: float, checkpoint_path: Path = DEFAULT_CHECKPOINT) -> SignLSTM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab()
    model = build_model().to(device)

    train_loader = make_dataloader(load_records(data_dir, split="train"), vocab, SEQUENCE_LENGTH, BATCH_SIZE)
    val_loader = make_dataloader(load_records(data_dir, split="val"), vocab, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False)

    # Adam has no literal "momentum" knob -- beta1 plays the same role,
    # controlling how much of the running gradient average carries over
    # between steps.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999))
    scheduler = ExponentialLR(optimizer, gamma=lr_scaling)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_prec, train_rec = run_epoch(
            model, train_loader, len(vocab), device, optimizer, clip_norm=clip_norm)
        val_loss, val_acc, val_prec, val_rec = run_epoch(model, val_loader, len(vocab), device)
        scheduler.step()

        print(f"epoch {epoch}/{epochs} "
              f"train: loss={train_loss:.4f} acc={train_acc:.4f} precision={train_prec:.4f} recall={train_rec:.4f} | "
              f"val: loss={val_loss:.4f} acc={val_acc:.4f} precision={val_prec:.4f} recall={val_rec:.4f}")

        # Saved every epoch, not just at the end -- a crash or interrupt on a
        # long run should still leave the latest weights on disk.
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")

    return model


def _self_check():
    """Synthetic records through one training + one eval step -- no real data folder needed."""
    vocab = ["book", "chair", "hello"]
    records = []
    for label_idx, label in enumerate(vocab):
        for f in range(5):
            landmarks = [{"x": 0.1 * label_idx + 0.01 * i, "y": 0.2, "z": 0.0} for i in range(21)]
            records.append({"video": label, "label": label, "frame": f, "handedness": "Right", "landmarks": landmarks})

    loader = make_dataloader(records, vocab, sequence_length=6, batch_size=3, shuffle=False)
    model = SignLSTM(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))

    train_loss, train_acc, train_prec, train_rec = run_epoch(
        model, loader, len(vocab), torch.device("cpu"), optimizer, clip_norm=1.0)
    val_loss, val_acc, val_prec, val_rec = run_epoch(model, loader, len(vocab), torch.device("cpu"))

    assert train_loss >= 0 and val_loss >= 0
    assert 0.0 <= train_acc <= 1.0 and 0.0 <= val_acc <= 1.0
    assert 0.0 <= train_prec <= 1.0 and 0.0 <= train_rec <= 1.0

    # gradient clipping: a deliberately huge gradient should get capped to clip_norm
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([100.0, 100.0, 100.0])
    pre_clip_norm = nn.utils.clip_grad_norm_([p], max_norm=1.0)
    assert pre_clip_norm > 1.0
    assert torch.allclose(p.grad.norm(), torch.tensor(1.0), atol=1e-4)
    print("self-check passed: one train step + one eval step ran end-to-end with "
          "CTC loss and accuracy/precision/recall all in range, gradient clipping caps norm as expected")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help="folder of *_landmarks.json files, split via data/split.json (default: data/raw_landmarks)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scaling", type=float, default=1.0,
                         help="per-epoch multiplicative LR decay (ExponentialLR gamma); 1.0 = constant LR")
    parser.add_argument("--momentum", type=float, default=0.9, help="Adam's beta1")
    parser.add_argument("--clip-norm", type=float, default=5.0, help="max gradient norm (clips exploding gradients)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                         help="where to save trained weights (default: models/checkpoint.pt)")
    parser.add_argument("--self-test", action="store_true", help="run a synthetic smoke test instead of training")
    args = parser.parse_args()

    if args.self_test:
        _self_check()
        sys.exit(0)

    train(args.data_dir, args.epochs, args.lr, args.lr_scaling, args.momentum,
          args.clip_norm, args.checkpoint)
