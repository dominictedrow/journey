"""Train SignLSTM from pre-extracted MediaPipe landmarks in data/raw_landmarks/,
not a live MediaPipe feed. train/val are read from that one directory via
ingest.load_records's split filter, using the train/val/test assignment
split_dataset.py already wrote to data/split.json.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ExponentialLR

from dataset import SEQUENCE_LENGTH, load_vocab, make_dataloader, make_lazy_dataloader
from ingest import list_video_files
from model import SignLSTM, build_model

BATCH_SIZE = 16
PATIENCE = 10
# Val only runs once per epoch and is far shorter than a train pass, so it
# doesn't need train's worker count -- fewer workers here means fewer
# persistent processes each paying the Windows spawn/pickle cost up front.
VAL_NUM_WORKERS = 2
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw_landmarks"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "models" / "checkpoint.pt"
LOG_DIR = Path(__file__).parent.parent / "logs"


class _Tee:
    """Mirrors writes to multiple streams -- lets print() reach both the
    console and a log file without touching every call site."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


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


def run_epoch(model: SignLSTM, loader, vocab_size: int, device: torch.device, optimizer=None,
              clip_norm: float | None = None, scaler: GradScaler | None = None):
    """One pass over loader. Trains if optimizer is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)
    ce_loss = nn.CrossEntropyLoss()
    amp_enabled = scaler is not None and scaler.is_enabled()

    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            targets = y.argmax(dim=1)

            with autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)  # (batch, vocab_size)
                loss = ce_loss(logits, targets)

            if is_train:
                optimizer.zero_grad()
                # GradScaler multiplies the loss up before backward so small
                # fp16 gradients don't underflow to zero, then unscales them
                # back down before clipping/stepping so clip_norm and the
                # optimizer see real gradient magnitudes.
                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.detach().argmax(dim=-1))
            all_targets.append(targets)

    preds, targets = torch.cat(all_preds), torch.cat(all_targets)
    accuracy = (preds == targets).float().mean().item()
    precision, recall = precision_recall_macro(preds, targets, vocab_size)
    return total_loss / len(targets), accuracy, precision, recall


def _is_improvement(val_loss: float, val_acc: float, best_val_loss: float, best_val_acc: float) -> bool:
    """True if this epoch beat the best val loss or val accuracy seen so far --
    either one counts as real progress and resets the early-stop patience counter.
    """
    return val_loss < best_val_loss or val_acc > best_val_acc


def train(data_dir: Path, epochs: int, lr: float, lr_scaling: float, momentum: float,
          clip_norm: float, weight_decay: float, batch_size: int = BATCH_SIZE, patience: int = PATIENCE,
          checkpoint_path: Path = DEFAULT_CHECKPOINT, num_workers: int = 8) -> SignLSTM:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab()
    model = build_model().to(device)

    train_loader = make_lazy_dataloader(list_video_files(data_dir, split="train"), vocab, SEQUENCE_LENGTH, batch_size,
                                         augment=True, num_workers=num_workers)
    val_loader = make_lazy_dataloader(list_video_files(data_dir, split="val"), vocab, SEQUENCE_LENGTH, batch_size,
                                       shuffle=False, num_workers=min(num_workers, VAL_NUM_WORKERS))

    # Adam has no literal "momentum" knob -- beta1 plays the same role,
    # controlling how much of the running gradient average carries over
    # between steps.
    # weight_decay is L2 regularization: it shrinks every weight toward zero
    # a little each step, penalizing the large weights memorization tends to
    # produce, so the model favors smaller weights that still fit the data.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=lr_scaling)
    # fp16 autocast + gradient scaling only pay off on CUDA's tensor cores --
    # enabled=False makes GradScaler a no-op on CPU.
    scaler = GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_loss, best_val_acc, stall_epochs = float("inf"), 0.0, 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_prec, train_rec = run_epoch(
            model, train_loader, len(vocab), device, optimizer, clip_norm=clip_norm, scaler=scaler)
        val_loss, val_acc, val_prec, val_rec = run_epoch(model, val_loader, len(vocab), device, scaler=scaler)
        scheduler.step()

        print(f"epoch {epoch}/{epochs} "
              f"train: loss={train_loss:.4f} acc={train_acc:.2%} precision={train_prec:.2%} recall={train_rec:.2%} | "
              f"val: loss={val_loss:.4f} acc={val_acc:.2%} precision={val_prec:.2%} recall={val_rec:.2%}")

        # Saved every epoch, not just at the end -- a crash or interrupt on a
        # long run should still leave the latest weights on disk.
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")

        # Early stopping: stop once val loss AND val accuracy have both sat
        # still for `patience` epochs in a row -- either one moving is real
        # progress and resets the counter.
        improved = _is_improvement(val_loss, val_acc, best_val_loss, best_val_acc)
        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)
        stall_epochs = 0 if improved else stall_epochs + 1
        if stall_epochs >= patience:
            print(f"stopping early: val loss/accuracy hasn't improved in {patience} epochs")
            break

    return model


def _self_check():
    """Synthetic records through one eval step -- no real data folder needed."""
    vocab = ["book", "chair", "hello"]
    records = []
    for label_idx, label in enumerate(vocab):
        for f in range(5):
            landmarks = [{"x": 0.1 * label_idx + 0.01 * i, "y": 0.2, "z": 0.0} for i in range(21)]
            records.append({"video": label, "label": label, "frame": f, "handedness": "Right", "landmarks": landmarks})

    loader = make_dataloader(records, vocab, sequence_length=6, batch_size=3, shuffle=False, num_workers=0)
    model = SignLSTM(vocab_size=len(vocab))

    val_loss, val_acc, val_prec, val_rec = run_epoch(model, loader, len(vocab), torch.device("cpu"))

    assert val_loss >= 0
    assert 0.0 <= val_acc <= 1.0
    assert 0.0 <= val_prec <= 1.0 and 0.0 <= val_rec <= 1.0

    # gradient clipping: a deliberately huge gradient should get capped to clip_norm
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = torch.tensor([100.0, 100.0, 100.0])
    pre_clip_norm = nn.utils.clip_grad_norm_([p], max_norm=1.0)
    assert pre_clip_norm > 1.0
    assert torch.allclose(p.grad.norm(), torch.tensor(1.0), atol=1e-4)

    # early-stop trigger: loss improving, or accuracy improving, both count as
    # progress; only "neither moved" should signal a stall.
    assert _is_improvement(val_loss=0.5, val_acc=0.10, best_val_loss=1.0, best_val_acc=0.20)  # loss dropped
    assert _is_improvement(val_loss=1.5, val_acc=0.30, best_val_loss=1.0, best_val_acc=0.20)  # acc rose
    assert not _is_improvement(val_loss=1.0, val_acc=0.20, best_val_loss=1.0, best_val_acc=0.20)  # neither moved

    print("self-check passed: one eval step ran end-to-end with "
          "cross-entropy loss and accuracy/precision/recall all in range, gradient clipping caps norm as expected, "
          "and the early-stop improvement check reacts to loss/accuracy correctly")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                         help="folder of *_landmarks.json files, split via data/split.json (default: data/raw_landmarks)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr-scaling", type=float, default=0.99,
                         help="per-epoch multiplicative LR decay (ExponentialLR gamma); 1.0 = constant LR")
    parser.add_argument("--momentum", type=float, default=0.9, help="Adam's beta1")
    parser.add_argument("--clip-norm", type=float, default=5.0, help="max gradient norm (clips exploding gradients)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                         help="Adam's L2 penalty on weights (regularization against overfitting)")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                         help="stop early if val loss/accuracy hasn't improved in this many epochs")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                         help="where to save trained weights (default: models/checkpoint.pt)")
    parser.add_argument("--num-workers", type=int, default=8,
                         help="DataLoader subprocesses preparing batches (raise on more CPU cores to keep the GPU fed)")
    parser.add_argument("--self-test", action="store_true", help="run a synthetic smoke test instead of training")
    args = parser.parse_args()

    if args.self_test:
        _self_check()
        sys.exit(0)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    real_stdout = sys.stdout
    with open(log_path, "w") as log_file:
        # Restored in finally so nothing printed after this block (e.g. a
        # DataLoader worker's teardown output during interpreter shutdown)
        # writes to log_file after it's closed.
        sys.stdout = _Tee(real_stdout, log_file)
        try:
            print(f"logging to {log_path}")
            train(args.data_dir, args.epochs, args.lr, args.lr_scaling, args.momentum,
                  args.clip_norm, args.weight_decay, args.batch_size, args.patience, args.checkpoint,
                  args.num_workers)
        finally:
            sys.stdout = real_stdout
