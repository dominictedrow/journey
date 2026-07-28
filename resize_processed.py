import math
import os

import numpy as np

from data_pipeline import STATS_PATH, compute_target_size, list_split_files, resize_array

SPLITS = ("train", "val", "test")
DOWNSCALE_FACTOR = 0.65  # shrinks current on-disk size further to cut conv compute

# model.py's CatDogCNN has 3 MaxPool2d(2,2) stages (halves each dim) feeding a 6x6
# adaptive pool; MPS's adaptive_avg_pool2d requires the input to divide evenly by
# the output size, so each on-disk dimension must be a multiple of 2**3 * 6 = 48.
POOL_DIVISOR = 48


def round_up_to_multiple(value, multiple):
    return math.ceil(value / multiple) * multiple


def resize_all():
    train_files = list_split_files("train")
    current_size = compute_target_size(train_files)  # size of files already on disk
    raw_target = (current_size[0] * DOWNSCALE_FACTOR, current_size[1] * DOWNSCALE_FACTOR)
    # round up (never down) so we don't give up more resolution than the requested scale
    target_size = tuple(round_up_to_multiple(round(v), POOL_DIVISOR) for v in raw_target)
    print(f"current size: {current_size} -> target size: {target_size} (rounded up to multiple of {POOL_DIVISOR})")

    for split in SPLITS:
        files = list_split_files(split)
        for path in files:
            with np.load(path) as npz:
                pixels = npz["pixels"]
                label = str(npz["label"])
            resized = resize_array(pixels, target_size)
            np.savez_compressed(path, pixels=resized, label=label)
        print(f"{split}: resized {len(files)} files")

    if os.path.exists(STATS_PATH):
        os.remove(STATS_PATH)  # target size changed; force recompute of stats on next run


if __name__ == "__main__":
    resize_all()
