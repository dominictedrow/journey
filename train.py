import argparse
import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_pipeline import CatDogDataset, get_or_compute_pipeline_stats, worker_init_fn, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS
from model import CatDogCNN, get_default_device


def parse_args():
    parser = argparse.ArgumentParser(description="Train the cat/dog CNN")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr-schedule", action="store_true",
                         help="decay lr across epochs via cosine annealing instead of holding it fixed")
    parser.add_argument("--num-workers", type=int, default=4,
                         help="parallel data-loading worker processes (0 = load in the main process)")
    parser.add_argument("--device", type=str, default=get_default_device())
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_model.pt")
    return parser.parse_args()


def run_epoch(model, batches, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    running_correct = 0
    num_samples = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in batches:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_train:
                optimizer.zero_grad()

            outputs = model(inputs).squeeze(1)  # (batch, 1) -> (batch,)
            loss = criterion(outputs, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            running_correct += (predictions == targets).sum().item()
            num_samples += batch_size

    return running_loss / num_samples, running_correct / num_samples


def train(args):
    torch.set_num_threads(os.cpu_count())  # torch defaults to fewer intra-op threads than cores available

    device = torch.device(args.device)
    model = CatDogCNN(dropout=args.dropout).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.lr_schedule else None
    criterion = nn.BCEWithLogitsLoss()

    _, mean, std = get_or_compute_pipeline_stats()

    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    history_path = os.path.join(checkpoint_dir, "history.json") if checkpoint_dir else "history.json"

    train_dataset = CatDogDataset("train", mean, std, augment=True)
    val_dataset = CatDogDataset("val", mean, std, augment=False)
    loader_kwargs = dict(num_workers=args.num_workers, worker_init_fn=worker_init_fn,
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    best_val_acc = -1.0
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)

        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch + 1}/{args.epochs} - "
              f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - lr: {current_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.checkpoint_path)
            print(f"  saved new best checkpoint (val_acc={val_acc:.4f}) -> {args.checkpoint_path}")

        with open(history_path, "w") as f:
            json.dump(history, f)

        if scheduler is not None:
            scheduler.step()

    return model


if __name__ == "__main__":
    train(parse_args())
