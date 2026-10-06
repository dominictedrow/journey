import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline import CatDogDataset, DEFAULT_BATCH_SIZE, get_or_compute_pipeline_stats, worker_init_fn
from inference import load_model
from model import get_default_device


def evaluate_split(model, split, mean, std, device, batch_size, num_workers=4):
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    num_samples = 0
    tp = fp = fn = tn = 0

    dataset = CatDogDataset(split, mean, std, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, worker_init_fn=worker_init_fn)

    with torch.no_grad():
        for inputs, targets in loader:  # 1.0 = dog, 0.0 = cat; dog is the positive class
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)

            batch_n = inputs.size(0)
            running_loss += loss.item() * batch_n
            num_samples += batch_n

            predictions = (torch.sigmoid(outputs) > 0.5).float()
            tp += ((predictions == 1) & (targets == 1)).sum().item()
            fp += ((predictions == 1) & (targets == 0)).sum().item()
            fn += ((predictions == 0) & (targets == 1)).sum().item()
            tn += ((predictions == 0) & (targets == 0)).sum().item()

    return {
        "loss": running_loss / num_samples,
        "accuracy": (tp + tn) / num_samples,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "num_samples": num_samples,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def plot_loss_curves(runs, output_path):
    """runs: list of (label, history_path) tuples, each plotted as its own train/val pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, history_path) in enumerate(runs):
        with open(history_path) as f:
            history = json.load(f)
        epochs = range(1, len(history["train_loss"]) + 1)
        color = color_cycle[i % len(color_cycle)]
        plt.plot(epochs, history["train_loss"], label=f"{label} train", color=color, linestyle="-")
        plt.plot(epochs, history["val_loss"], label=f"{label} val", color=color, linestyle="--")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def plot_loss_curve(history_path, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the cat/dog CNN")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--history-path", type=str, default="checkpoints/history.json")
    parser.add_argument("--loss-plot-path", type=str, default="checkpoints/loss_curve.png")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=4,
                         help="parallel data-loading worker processes (0 = load in the main process)")
    parser.add_argument("--device", type=str, default=get_default_device())
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    _, mean, std = get_or_compute_pipeline_stats()
    model = load_model(args.checkpoint_path, device)

    metrics = evaluate_split(model, args.split, mean, std, device, args.batch_size, args.num_workers)

    print(f"split: {args.split} (n={metrics['num_samples']})")
    print(f"  loss:      {metrics['loss']:.4f}")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  tp={metrics['tp']} fp={metrics['fp']} fn={metrics['fn']} tn={metrics['tn']}")

    if os.path.exists(args.history_path):
        plot_loss_curve(args.history_path, args.loss_plot_path)
        print(f"saved loss curve -> {args.loss_plot_path}")
    else:
        print(f"no history file found at {args.history_path}, skipping loss curve")


if __name__ == "__main__":
    main()
