"""
Standalone evaluation: run a saved, already-trained checkpoint against
whichever split you choose (train, validation, or test), and report loss,
accuracy, per-class precision/recall, and (if training history is
available) a loss curve plot.

This is deliberately separate from train.py. Training only ever learns
from the train split, and only ever picks its "best" checkpoint using the
validation split -- this script never feeds back into either of those
decisions. It just loads frozen weights and reports numbers, so pointing
it at the test split here can't leak test information into the model or
into which epoch got saved as best.

Precision and recall are computed per digit (0-9) from a confusion matrix
built by comparing every prediction to its true label:
  precision for digit c: of everything the model called "c", what fraction
    actually was c?
  recall for digit c:    of everything that actually was "c", what
    fraction did the model catch?

Usage:
  python scripts/evaluate.py --split validation
  python scripts/evaluate.py --split test --checkpoint checkpoints/best.pt
  python scripts/evaluate.py --split train --no-loss-graph
"""

import argparse
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn

from model import MnistCNN
from preprocess import load_preprocessed
from train import get_device, to_tensor_images, to_tensor_labels

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
NUM_CLASSES = 10


def predict_all(model, images, labels, device, batch_size=256):
    """Runs inference over the whole split and returns (loss, accuracy,
    true_labels, predicted_labels) as numpy arrays of class indices."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    all_preds = []
    with torch.no_grad():
        for start in range(0, images.shape[0], batch_size):
            batch_images = images[start:start + batch_size].to(device)
            batch_labels = labels[start:start + batch_size].to(device)
            logits = model(batch_images)
            total_loss += loss_fn(logits, batch_labels).item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            all_preds.append(preds.cpu().numpy())
    n = images.shape[0]
    pred_labels = np.concatenate(all_preds)
    true_labels = labels.numpy()
    return total_loss / n, correct / n, true_labels, pred_labels


def confusion_matrix(true_labels, pred_labels, num_classes=NUM_CLASSES):
    """Rows = true digit, columns = predicted digit. cm[t, p] counts how
    often a true-digit-t image got predicted as p."""
    flat_index = true_labels * num_classes + pred_labels
    counts = np.bincount(flat_index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def precision_recall_per_class(cm):
    true_positives = np.diag(cm).astype(np.float64)
    predicted_totals = cm.sum(axis=0)   # column sums: everything predicted as each class
    actual_totals = cm.sum(axis=1)      # row sums: everything actually each class

    precision = np.divide(true_positives, predicted_totals,
                           out=np.zeros_like(true_positives), where=predicted_totals > 0)
    recall = np.divide(true_positives, actual_totals,
                        out=np.zeros_like(true_positives), where=actual_totals > 0)

    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom,
                    out=np.zeros_like(true_positives), where=denom > 0)
    return precision, recall, f1


def plot_loss_curve(checkpoint_dir, output_path):
    history_path = checkpoint_dir / "history.json"
    if not history_path.exists():
        print(f"\n(No training history found at {history_path} -- skipping loss graph. "
              "This file is written by train.py as it trains.)")
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] + 1 for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, marker="o", label="train loss")
    plt.plot(epochs, val_loss, marker="o", label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on one dataset split")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation",
                         help="Which dataset split to evaluate on (default: validation)")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt",
                         help="Path to a checkpoint saved by train.py (default: checkpoints/best.pt)")
    parser.add_argument("--no-loss-graph", action="store_true",
                         help="Skip generating the training/validation loss curve plot")
    args = parser.parse_args()

    if args.split == "test":
        print("Note: the test split is meant as a final, one-time check. Using it repeatedly "
              "to guide decisions (e.g. re-training or tuning based on the score you see here) "
              "leaks test information into the model, the same way training on it directly would.\n")

    split_prefix = "val" if args.split == "validation" else args.split

    device = get_device()
    data = load_preprocessed()
    images = to_tensor_images(data[f"{split_prefix}_images"])
    labels = to_tensor_labels(data[f"{split_prefix}_labels"])

    model = MnistCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    loss, accuracy, true_labels, pred_labels = predict_all(model, images, labels, device)

    print(f"Checkpoint: {args.checkpoint} (from epoch {checkpoint['epoch'] + 1})")
    print(f"Split: {args.split} ({images.shape[0]} images)")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(true_labels, pred_labels)
    precision, recall, f1 = precision_recall_per_class(cm)

    print(f"\n{'digit':>5} {'precision':>10} {'recall':>10} {'f1':>10}")
    for digit in range(NUM_CLASSES):
        print(f"{digit:>5} {precision[digit]:>10.4f} {recall[digit]:>10.4f} {f1[digit]:>10.4f}")
    print(f"{'macro':>5} {precision.mean():>10.4f} {recall.mean():>10.4f} {f1.mean():>10.4f}")

    if not args.no_loss_graph:
        output_path = args.checkpoint.parent / "loss_curve.png"
        saved_path = plot_loss_curve(args.checkpoint.parent, output_path)
        if saved_path is not None:
            print(f"\nLoss curve saved to {saved_path}")


if __name__ == "__main__":
    main()
