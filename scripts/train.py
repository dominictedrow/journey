"""
Training loop for MnistCNN.

Each epoch: sweep the full training set once in shuffled batches, and for
every batch have the model guess, measure how wrong it was (loss), and
nudge its weights to be less wrong (the optimizer, using the learning
rate). After each epoch, check accuracy on the validation set (data the
model never trains on) to see if it's actually learning or just
memorizing.

Checkpointing saves the model's weights to disk during training so a run
isn't lost if it's interrupted, and so the best-performing version (by
validation accuracy) is kept even if later epochs get worse.

Usage:
  python scripts/train.py --epochs 10 --lr 0.001 --batch-size 64
  python scripts/train.py --epochs 5 --resume checkpoints/last.pt
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from model import MnistCNN
from preprocess import load_preprocessed, iterate_batches, iterate_batches_augmented

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_tensor_images(images):
    # (N, 28, 28, 1) -> (N, 1, 28, 28): PyTorch conv layers expect channels first.
    # clone(contiguous_format) instead of .contiguous(): since the channel dim has
    # size 1, .contiguous() treats the permuted view as already contiguous and
    # skips the copy, leaving non-canonical strides that MPS's backward pass for
    # BatchNorm/Flatten can't handle.
    return torch.from_numpy(images).permute(0, 3, 1, 2).clone(memory_format=torch.contiguous_format)


def to_tensor_labels(one_hot_labels):
    # CrossEntropyLoss wants class indices (0-9), not one-hot vectors.
    return torch.from_numpy(one_hot_labels.argmax(axis=1)).long()


def evaluate(model, images, labels, device, batch_size=256):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch_images = images[start:start + batch_size].to(device)
            batch_labels = labels[start:start + batch_size].to(device)
            logits = model(batch_images)
            total_loss += loss_fn(logits, batch_labels).item()
            correct += (logits.argmax(dim=1) == batch_labels).sum().item()
    n = images.shape[0]
    return total_loss / n, correct / n


def save_checkpoint(path, model, optimizer, epoch, val_accuracy):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train MnistCNN")
    parser.add_argument("--epochs", type=int, default=10,
                         help="Number of full passes over the training set (default: 10)")
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="Learning rate: how large each weight update step is (default: 0.001)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                         help="L2 regularization strength: penalizes large weights to reduce "
                              "overfitting (default: 0.0001, set to 0 to disable)")
    parser.add_argument("--batch-size", type=int, default=64,
                         help="Number of images per training step (default: 64)")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR,
                         help="Where to save model checkpoints (default: checkpoints/)")
    parser.add_argument("--resume", type=Path, default=None,
                         help="Path to a checkpoint file to resume training from")
    parser.add_argument("--seed", type=int, default=0,
                         help="Seed for batch shuffling, for reproducible runs")
    parser.add_argument("--patience", type=int, default=5,
                         help="Stop early if validation accuracy doesn't improve for this many "
                              "epochs in a row (default: 5, set to 0 to disable)")
    parser.add_argument("--no-augment", action="store_true",
                         help="Disable data augmentation (random small rotation/shift) on "
                              "training images. On by default.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    data = load_preprocessed()
    train_images = to_tensor_images(data["train_images"])
    train_labels = to_tensor_labels(data["train_labels"])
    val_images = to_tensor_images(data["val_images"])
    val_labels = to_tensor_labels(data["val_labels"])

    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_accuracy = 0.0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_accuracy = checkpoint["val_accuracy"]
        print(f"Resumed from {args.resume} (epoch {checkpoint['epoch']}, "
              f"val accuracy {best_val_accuracy:.4f})")

    last_path = args.checkpoint_dir / "last.pt"
    best_path = args.checkpoint_dir / "best.pt"
    history_path = args.checkpoint_dir / "history.json"

    history = []
    if args.resume is not None and history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

    epochs_without_improvement = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        n_seen = 0

        if args.no_augment:
            batches = iterate_batches(
                data["train_images"], data["train_labels"],
                batch_size=args.batch_size, seed=args.seed + epoch,
            )
        else:
            batches = iterate_batches_augmented(
                data["train_images_raw"], data["train_labels"], data["mean"], data["std"],
                batch_size=args.batch_size, seed=args.seed + epoch,
            )
        for batch_images_np, batch_labels_np in batches:
            batch_images = to_tensor_images(batch_images_np).to(device)
            batch_labels = to_tensor_labels(batch_labels_np).to(device)

            optimizer.zero_grad()
            logits = model(batch_images)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_n = batch_images.shape[0]
            running_loss += loss.item() * batch_n
            running_correct += (logits.argmax(dim=1) == batch_labels).sum().item()
            n_seen += batch_n

        train_loss = running_loss / n_seen
        train_accuracy = running_correct / n_seen
        val_loss, val_accuracy = evaluate(model, val_images, val_labels, device)

        print(f"Epoch {epoch + 1}: "
              f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(last_path, model, optimizer, epoch, val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, optimizer, epoch, val_accuracy)
            print(f"  -> new best (val_acc={val_accuracy:.4f}), saved to {best_path}")
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\nStopping early: no improvement in validation accuracy for "
                  f"{args.patience} epochs in a row.")
            break

    print(f"\nDone. Best validation accuracy: {best_val_accuracy:.4f} (saved to {best_path})")


if __name__ == "__main__":
    main()
