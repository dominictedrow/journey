"""
Preprocessing data model for the MNIST splits produced by split_validation.py.

Three things happen here:
  1. Pixel values are z-score scaled: (x - mean) / std, using the mean/std of
     the TRAINING set only. Validation and test are scaled with those same
     numbers, never their own -- the model should never "see" statistics from
     data it wasn't trained on.
  2. Labels (single digits 0-9) are one-hot encoded into 10-length vectors,
     since the network will output a probability per digit.
  3. Optionally, training images can be randomly augmented (small rotation +
     shift) each time a batch is drawn, via iterate_batches_augmented(). This
     only ever applies to training data -- validation and test images are
     always the untouched originals, since evaluation has to reflect what the
     model will actually see, not an artificially easier or harder version of
     it.

This module doesn't write files -- it loads raw IDX data and exposes ready-to-
train arrays in memory via load_preprocessed().
"""

import struct
import numpy as np
from pathlib import Path
from PIL import Image

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

TRAIN_IMAGES = DATASET_DIR / "train" / "images.idx3-ubyte"
TRAIN_LABELS = DATASET_DIR / "train" / "labels.idx1-ubyte"
VAL_IMAGES = DATASET_DIR / "validation" / "images.idx3-ubyte"
VAL_LABELS = DATASET_DIR / "validation" / "labels.idx1-ubyte"
TEST_IMAGES = DATASET_DIR / "t10k-images.idx3-ubyte"
TEST_LABELS = DATASET_DIR / "t10k-labels.idx1-ubyte"


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


def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def compute_zscore_stats(images):
    """Mean and std over every pixel in the given image set (single scalars,
    not per-pixel), computed on float64 for numerical stability."""
    pixels = images.astype(np.float64)
    return pixels.mean(), pixels.std()


def apply_zscore(images, mean, std):
    return ((images.astype(np.float32) - mean) / std).astype(np.float32)


def add_channel_dim(images):
    """Adds a trailing channel axis: (N, 28, 28) -> (N, 28, 28, 1)."""
    return images[..., np.newaxis]


def iterate_batches(images, labels, batch_size, seed=None):
    """Yields (batch_images, batch_labels) pairs covering the full dataset once,
    in a random order. Call again for a fresh shuffle (e.g. once per epoch)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(images.shape[0])
    for start in range(0, images.shape[0], batch_size):
        batch_idx = indices[start:start + batch_size]
        yield images[batch_idx], labels[batch_idx]


def augment_image(image, rng, max_rotation_degrees=12, max_shift_pixels=2):
    """Applies a random small rotation and random small x/y shift to a single
    raw uint8 image (28, 28). Empty space introduced by either transform is
    filled with 0 (black), matching MNIST's background."""
    img = Image.fromarray(image)

    angle = rng.uniform(-max_rotation_degrees, max_rotation_degrees)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)

    dx = int(rng.integers(-max_shift_pixels, max_shift_pixels + 1))
    dy = int(rng.integers(-max_shift_pixels, max_shift_pixels + 1))
    img = img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), fillcolor=0)

    return np.array(img)


def iterate_batches_augmented(images_raw, labels, mean, std, batch_size, seed=None):
    """Like iterate_batches, but for training only: draws from the RAW (not
    yet normalized) images, applies a fresh random rotation + shift to each
    image, and only then normalizes with the fixed train-set mean/std. Because
    the augmentation is random and applied on every draw, the model almost
    never sees the exact same pixels for a given image twice, even across
    epochs -- that's what discourages it from memorizing specific images
    instead of learning the general shape of each digit."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(images_raw.shape[0])
    for start in range(0, images_raw.shape[0], batch_size):
        batch_idx = indices[start:start + batch_size]
        raw_batch = images_raw[batch_idx]
        augmented = np.stack([augment_image(img, rng) for img in raw_batch])
        normalized = apply_zscore(augmented, mean, std)
        yield add_channel_dim(normalized), labels[batch_idx]


def load_preprocessed():
    """Loads train/validation/test, z-score scales all three using
    train-set statistics, and one-hot encodes all labels.

    Returns a dict of numpy arrays:
      train_images, train_labels, val_images, val_labels, test_images, test_labels
    Image arrays have shape (N, 28, 28, 1) and dtype float32.
    Label arrays have shape (N, 10) and dtype float32.

    Also includes train_images_raw: the un-normalized (N, 28, 28) uint8 training
    images, needed by iterate_batches_augmented() since augmentation happens
    before normalization.
    """
    train_images_raw = read_idx_images(TRAIN_IMAGES)
    train_labels_raw = read_idx_labels(TRAIN_LABELS)
    val_images_raw = read_idx_images(VAL_IMAGES)
    val_labels_raw = read_idx_labels(VAL_LABELS)
    test_images_raw = read_idx_images(TEST_IMAGES)
    test_labels_raw = read_idx_labels(TEST_LABELS)

    mean, std = compute_zscore_stats(train_images_raw)

    return {
        "train_images": add_channel_dim(apply_zscore(train_images_raw, mean, std)),
        "train_images_raw": train_images_raw,
        "train_labels": one_hot_encode(train_labels_raw),
        "val_images": add_channel_dim(apply_zscore(val_images_raw, mean, std)),
        "val_labels": one_hot_encode(val_labels_raw),
        "test_images": add_channel_dim(apply_zscore(test_images_raw, mean, std)),
        "test_labels": one_hot_encode(test_labels_raw),
        "mean": mean,
        "std": std,
    }


if __name__ == "__main__":
    data = load_preprocessed()
    print(f"Train-set stats used for scaling: mean={data['mean']:.4f}, std={data['std']:.4f}")
    for split in ("train", "val", "test"):
        imgs = data[f"{split}_images"]
        labels = data[f"{split}_labels"]
        print(f"{split:>5}: images {imgs.shape} (mean={imgs.mean():.4f}, std={imgs.std():.4f}), "
              f"labels {labels.shape}")
    print("\nExample one-hot label:", data["train_labels"][0])

    print("\nBatching demo (batch_size=8, first 3 batches of train set):")
    for i, (batch_images, batch_labels) in enumerate(
        iterate_batches(data["train_images"], data["train_labels"], batch_size=8, seed=0)
    ):
        digits = batch_labels.argmax(axis=1)
        print(f"  batch {i}: images {batch_images.shape}, digits {digits}")
        if i == 2:
            break
