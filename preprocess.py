import os
import numpy as np
from PIL import Image

DATASET_DIR = "dataset"
CLASSES = {"Cat": "cat", "Dog": "dog"}
OUTPUT_DIR = "processed"

SKIP_FILES = {"Thumbs.db", ".DS_Store"}


def process():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    kept = 0
    skipped_non_rgb = 0
    skipped_unreadable = 0

    for folder_name, label in CLASSES.items():
        folder_path = os.path.join(DATASET_DIR, folder_name)
        out_label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(out_label_dir, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename in SKIP_FILES:
                continue
            total += 1
            file_path = os.path.join(folder_path, filename)
            try:
                with Image.open(file_path) as img:
                    if img.mode != "RGB":
                        skipped_non_rgb += 1
                        continue
                    pixels = np.array(img)  # shape (H, W, 3): R, G, B channel values
            except Exception as e:
                print(f"Skipping unreadable file: {file_path} ({e})")
                skipped_unreadable += 1
                continue

            out_name = os.path.splitext(filename)[0] + ".npz"
            out_path = os.path.join(out_label_dir, out_name)
            np.savez_compressed(out_path, pixels=pixels, label=label)
            kept += 1

    print(f"Total files seen: {total}")
    print(f"Kept (RGB, saved as .npz): {kept}")
    print(f"Skipped (non-RGB): {skipped_non_rgb}")
    print(f"Skipped (unreadable): {skipped_unreadable}")


if __name__ == "__main__":
    process()
