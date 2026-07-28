"""
Splits the raw 60,000-image MNIST training set into a training set and a
validation set using stratified random sampling with a fixed seed.

Result (out of the full 70,000 images across train+test):
  train:      42,000  (60%)
  validation: 18,000  (~25.7%)
  test:       10,000  (~14.3%, untouched)

The original raw files in dataset/ are never modified. The split is written
to dataset/train/ and dataset/validation/ as new, independent IDX files.
Every training-set index goes to exactly one of train or validation, so
there is zero overlap between them -- the same guarantee that already
exists between train and test.
"""

import struct
import numpy as np
from pathlib import Path

SEED = 42
TRAIN_FRACTION = 0.7  # 42,000 / 60,000

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
RAW_IMAGES = DATASET_DIR / "train-images.idx3-ubyte"
RAW_LABELS = DATASET_DIR / "train-labels.idx1-ubyte"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"


def read_idx_images(path):
    with open(path, "rb") as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"unexpected magic number {magic} in {path}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(count, rows, cols)


def read_idx_labels(path):
    with open(path, "rb") as f:
        magic, count = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"unexpected magic number {magic} in {path}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def write_idx_images(path, images):
    count, rows, cols = images.shape
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 2051, count, rows, cols))
        f.write(images.tobytes())


def write_idx_labels(path, labels):
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 2049, labels.shape[0]))
        f.write(labels.tobytes())


def stratified_split_indices(labels, train_fraction, seed):
    """Returns (train_indices, val_indices), partitioned per class so both
    splits keep the same digit balance as the original set, with an exact
    global split size (largest-remainder method for the rounding)."""
    rng = np.random.default_rng(seed)

    per_class_train_idx = []
    per_class_val_idx = []
    remainders = []  # (fractional leftover, class) to hand out extra train slots

    total_train_target = round(len(labels) * train_fraction)

    class_indices = {}
    for digit in range(10):
        idx = np.where(labels == digit)[0]
        rng.shuffle(idx)
        class_indices[digit] = idx

    exact_train_counts = {}
    floor_sum = 0
    for digit, idx in class_indices.items():
        exact = len(idx) * train_fraction
        floor_count = int(np.floor(exact))
        exact_train_counts[digit] = floor_count
        floor_sum += floor_count
        remainders.append((exact - floor_count, digit))

    leftover_slots = total_train_target - floor_sum
    remainders.sort(reverse=True)  # largest fractional remainder first
    for _, digit in remainders[:leftover_slots]:
        exact_train_counts[digit] += 1

    for digit, idx in class_indices.items():
        n_train = exact_train_counts[digit]
        per_class_train_idx.append(idx[:n_train])
        per_class_val_idx.append(idx[n_train:])

    train_indices = np.concatenate(per_class_train_idx)
    val_indices = np.concatenate(per_class_val_idx)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def main():
    images = read_idx_images(RAW_IMAGES)
    labels = read_idx_labels(RAW_LABELS)
    assert images.shape[0] == labels.shape[0]

    train_idx, val_idx = stratified_split_indices(labels, TRAIN_FRACTION, SEED)

    overlap = np.intersect1d(train_idx, val_idx)
    assert overlap.size == 0, "train/validation indices overlap!"
    assert len(train_idx) + len(val_idx) == len(labels)

    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)

    write_idx_images(TRAIN_DIR / "images.idx3-ubyte", images[train_idx])
    write_idx_labels(TRAIN_DIR / "labels.idx1-ubyte", labels[train_idx])
    write_idx_images(VAL_DIR / "images.idx3-ubyte", images[val_idx])
    write_idx_labels(VAL_DIR / "labels.idx1-ubyte", labels[val_idx])

    print(f"Total original training images: {len(labels)}")
    print(f"  -> train:      {len(train_idx)}")
    print(f"  -> validation: {len(val_idx)}")
    print("\nPer-digit counts:")
    print(f"{'digit':>5} {'train':>8} {'val':>8}")
    for digit in range(10):
        n_train = int(np.sum(labels[train_idx] == digit))
        n_val = int(np.sum(labels[val_idx] == digit))
        print(f"{digit:>5} {n_train:>8} {n_val:>8}")


if __name__ == "__main__":
    main()
