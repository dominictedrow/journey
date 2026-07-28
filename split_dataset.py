import os
import random
import shutil

PROCESSED_DIR = "processed"
SOURCE_LABELS = ["cat", "dog"]
SPLITS = {"train": 0.6, "val": 0.2, "test": 0.2}
SEED = 42


def main():
    random.seed(SEED)

    entries = []  # (source_path, label, filename)
    for label in SOURCE_LABELS:
        label_dir = os.path.join(PROCESSED_DIR, label)
        for filename in os.listdir(label_dir):
            entries.append((os.path.join(label_dir, filename), label, filename))

    random.shuffle(entries)

    n = len(entries)
    train_end = int(n * SPLITS["train"])
    val_end = train_end + int(n * SPLITS["val"])

    split_slices = {
        "train": entries[:train_end],
        "val": entries[train_end:val_end],
        "test": entries[val_end:],
    }

    for split_name in SPLITS:
        os.makedirs(os.path.join(PROCESSED_DIR, split_name), exist_ok=True)

    counts = {}
    for split_name, items in split_slices.items():
        cat_count = 0
        dog_count = 0
        for source_path, label, filename in items:
            new_filename = f"{label}_{filename}"
            dest_path = os.path.join(PROCESSED_DIR, split_name, new_filename)
            shutil.move(source_path, dest_path)
            if label == "cat":
                cat_count += 1
            else:
                dog_count += 1
        counts[split_name] = (cat_count, dog_count)

    for label in SOURCE_LABELS:
        label_dir = os.path.join(PROCESSED_DIR, label)
        if os.path.isdir(label_dir) and not os.listdir(label_dir):
            os.rmdir(label_dir)

    print(f"Total images: {n}")
    for split_name in SPLITS:
        cat_count, dog_count = counts[split_name]
        total = cat_count + dog_count
        print(f"{split_name}: {total} images ({total/n*100:.1f}%) - cat: {cat_count}, dog: {dog_count}")


if __name__ == "__main__":
    main()
