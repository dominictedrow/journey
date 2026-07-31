"""Run a trained SignLSTM checkpoint over a test folder of MediaPipe landmark
files (the same *_landmarks.json format ingest.load_records reads) and report
per-clip predictions plus accuracy/precision/recall -- the same metrics
train.py computes on val, just against held-out test data instead.
"""
import argparse
from pathlib import Path

import torch

from dataset import load_vocab, make_dataloader
from ingest import load_records
from model import build_model
from train import BATCH_SIZE, SEQUENCE_LENGTH, ctc_greedy_predict, precision_recall_macro


def run_inference(test_dir: Path, checkpoint: Path, vocab_path: Path = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = load_vocab(vocab_path) if vocab_path else load_vocab()

    model = build_model(vocab_path).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    loader = make_dataloader(load_records(test_dir), vocab, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            targets = y.argmax(dim=1)
            log_probs = model(x)
            preds = ctc_greedy_predict(log_probs, blank=len(vocab))

            for pred, target in zip(preds.tolist(), targets.tolist()):
                predicted_word = vocab[pred] if pred < len(vocab) else "<blank>"
                print(f"predicted={predicted_word} actual={vocab[target]}")

            all_preds.append(preds)
            all_targets.append(targets)

    preds, targets = torch.cat(all_preds), torch.cat(all_targets)
    accuracy = (preds == targets).float().mean().item()
    precision, recall = precision_recall_macro(preds, targets, len(vocab))
    print(f"\ntest set: accuracy={accuracy:.2%} precision={precision:.2%} recall={recall:.2%}")


def _self_check():
    """Synthetic test folder + a freshly-saved checkpoint -- no real trained model needed."""
    import json
    import tempfile
    from collections import defaultdict

    vocab_words = ["book", "chair", "hello"]
    records = []
    for label_idx, label in enumerate(vocab_words):
        for f in range(5):
            landmarks = [{"x": 0.1 * label_idx + 0.01 * i, "y": 0.2, "z": 0.0} for i in range(21)]
            records.append({"video": label, "label": label, "frame": f, "handedness": "Right", "landmarks": landmarks})

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("\n".join(vocab_words))

        test_dir = tmp_path / "test_data"
        test_dir.mkdir()
        by_label = defaultdict(list)
        for r in records:
            by_label[r["label"]].append(r)
        for label, recs in by_label.items():
            (test_dir / f"{label}_landmarks.json").write_text(json.dumps(recs))

        model = build_model(vocab_path)
        checkpoint_path = tmp_path / "checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)

        run_inference(test_dir, checkpoint_path, vocab_path)

    print("self-check passed: checkpoint loaded and test folder scored end-to-end")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, default=None, help="folder of *_landmarks.json files to score")
    parser.add_argument("--checkpoint", type=Path, default=None, help="path to a checkpoint saved by train.py")
    parser.add_argument("--self-test", action="store_true", help="run a synthetic smoke test instead of scoring real data")
    args = parser.parse_args()

    if args.self_test or not (args.test_dir and args.checkpoint):
        _self_check()
    else:
        run_inference(args.test_dir, args.checkpoint)
