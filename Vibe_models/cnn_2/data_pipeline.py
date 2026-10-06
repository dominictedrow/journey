import os
import json
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROCESSED_DIR = "processed"
CLASSES = ["cat", "dog"]  # index order fixes the label encoding

DEFAULT_SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
DEFAULT_BATCH_SIZE = 48
DEFAULT_EPOCHS = 10

STATS_PATH = os.path.join(PROCESSED_DIR, "pipeline_stats.json")


def list_split_files(split):
    split_dir = os.path.join(PROCESSED_DIR, split)
    return [os.path.join(split_dir, f) for f in os.listdir(split_dir)]


def resplit(split_ratios=DEFAULT_SPLIT_RATIOS, seed=42):
    assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, "split ratios must sum to 1.0"

    all_files = []
    for split in ("train", "val", "test"):
        all_files.extend(list_split_files(split))

    random.seed(seed)
    random.shuffle(all_files)

    n = len(all_files)
    train_end = int(n * split_ratios["train"])
    val_end = train_end + int(n * split_ratios["val"])
    new_splits = {
        "train": all_files[:train_end],
        "val": all_files[train_end:val_end],
        "test": all_files[val_end:],
    }

    for split, files in new_splits.items():
        split_dir = os.path.join(PROCESSED_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        for f in files:
            dest = os.path.join(split_dir, os.path.basename(f))
            if os.path.abspath(f) != os.path.abspath(dest):
                os.rename(f, dest)

    if os.path.exists(STATS_PATH):
        os.remove(STATS_PATH)  # target size / normalization stats depend on the split

    return {split: len(files) for split, files in new_splits.items()}


def resize_array(pixels, target_size):
    target_h, target_w = target_size
    img = Image.fromarray(pixels)
    img = img.resize((target_w, target_h), Image.BILINEAR)  # PIL takes (W, H)
    return np.array(img)


def augment_image(pixels):
    if random.random() < 0.5:
        pixels = pixels[:, ::-1, :]  # horizontal flip

    brightness_delta = random.uniform(-20.0, 20.0)
    pixels = pixels + brightness_delta

    contrast_scale = random.uniform(0.8, 1.2)
    channel_mean = pixels.mean(axis=(0, 1), keepdims=True)
    pixels = (pixels - channel_mean) * contrast_scale + channel_mean

    return np.clip(pixels, 0, 255)


def compute_target_size(files):
    heights, widths = [], []
    for path in files:
        with np.load(path) as npz:
            h, w, _ = npz["pixels"].shape
        heights.append(h)
        widths.append(w)
    return int(round(np.mean(heights))), int(round(np.mean(widths)))


def compute_normalization_stats(files):
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for path in files:
        with np.load(path) as npz:
            pixels = npz["pixels"].astype(np.float64)  # already resized to target_size on disk
        channel_sum += pixels.sum(axis=(0, 1))
        channel_sq_sum += (pixels ** 2).sum(axis=(0, 1))
        pixel_count += pixels.shape[0] * pixels.shape[1]

    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sq_sum / pixel_count - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)  # accumulate in float64, store as float32


def get_or_compute_pipeline_stats(force=False):
    if not force and os.path.exists(STATS_PATH):
        with open(STATS_PATH) as f:
            stats = json.load(f)
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        return tuple(stats["target_size"]), mean, std

    train_files = list_split_files("train")
    target_size = compute_target_size(train_files)
    mean, std = compute_normalization_stats(train_files)

    with open(STATS_PATH, "w") as f:
        json.dump({
            "target_size": list(target_size),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }, f)

    return target_size, mean, std


class CatDogDataset(Dataset):
    def __init__(self, split, mean, std, augment=False):
        self.files = list_split_files(split)
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with np.load(self.files[idx]) as npz:
            pixels = npz["pixels"].astype(np.float32)  # already resized to target_size on disk
            label = str(npz["label"])

        if self.augment:
            pixels = augment_image(pixels)

        pixels = (pixels - self.mean) / self.std  # z-score normalization, per channel
        pixels = pixels.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
        target = 1.0 if label == "dog" else 0.0  # 1.0 = dog, 0.0 = cat
        return torch.from_numpy(pixels), torch.tensor(target, dtype=torch.float32)


def worker_init_fn(worker_id):
    worker_seed = torch.utils.data.get_worker_info().seed % (2 ** 32)
    random.seed(worker_seed)  # each DataLoader worker process needs its own seed for augment_image
