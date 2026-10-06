import argparse

import numpy as np
import torch
from PIL import Image

from data_pipeline import CLASSES, get_or_compute_pipeline_stats, resize_array
from model import CatDogCNN, get_default_device


def load_model(checkpoint_path, device):
    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def preprocess_image(path, target_size, mean, std):
    with Image.open(path) as img:
        pixels = np.array(img.convert("RGB"))
    pixels = resize_array(pixels, target_size).astype(np.float32)
    pixels = (pixels - mean) / std
    return pixels.astype(np.float32)


@torch.no_grad()
def predict(model, image_paths, target_size, mean, std, device):
    batch = np.stack([preprocess_image(p, target_size, mean, std) for p in image_paths], axis=0)
    inputs = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)  # NHWC -> NCHW
    logits = model(inputs).squeeze(1)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels = [CLASSES[int(p > 0.5)] for p in probs]
    return list(zip(image_paths, labels, probs))


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained cat/dog CNN")
    parser.add_argument("images", nargs="+", help="paths to image files")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--device", type=str, default=get_default_device())
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    target_size, mean, std = get_or_compute_pipeline_stats()
    model = load_model(args.checkpoint_path, device)

    results = predict(model, args.images, target_size, mean, std, device)
    for path, label, prob in results:
        print(f"{path}: {label} (p_dog={prob:.4f})")


if __name__ == "__main__":
    main()
