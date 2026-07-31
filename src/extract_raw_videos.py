"""Extract MediaPipe hand landmarks for one clip in data/raw_videos/, using
label_lookup to know what word(s) it shows and where to trim.

Reuses process_video.process_video() for the actual per-frame MediaPipe run
-- decoded once per raw file even when a file yields several labeled
instances (some WLASL YouTube source videos carry dozens of glosses at
different frame ranges), so those instances never re-run MediaPipe over
frames they already share.

Output goes to data/raw_landmarks/, one JSON per instance, each record
already tagged with its own "video" and "label" -- unlike ingest.py's
*_landmarks.json files, nothing here is inferred from the filename, since a
raw_videos filename doesn't encode a label the way a personal recording does.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from label_lookup import lookup
from process_video import process_video

HERE = Path(__file__).parent.parent
RAW_DIR = HERE / "data" / "raw_videos"
OUT_DIR = HERE / "data" / "raw_landmarks"
COMPLETED_LOG = "_completed.txt"


def instance_records(video_path: Path, instances: list[dict]) -> list[tuple[str, list[dict]]]:
    """One decoded video + its label_lookup instances -> [(video_id, records), ...],
    one entry per instance, sliced to that instance's frame range and tagged
    with its own video id + label. video_id stays unique even when several
    instances share the source file (multi-label WLASL videos) or share a
    label (many different signers of the same word).
    """
    frames = process_video(str(video_path))
    results = []
    for i, inst in enumerate(instances):
        start, end = inst["frame_start"], inst["frame_end"]
        if end == -1:
            window = [r for r in frames if r["frame"] >= start]
        else:
            window = [r for r in frames if start <= r["frame"] <= end]
        if not window:
            continue
        video_id = video_path.stem if len(instances) == 1 else "{}__{}".format(video_path.stem, i)
        tagged = [{"video": video_id, "label": inst["label"], **r} for r in window]
        results.append((video_id, tagged))
    return results


def process_and_save(filename: str, raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> list[str]:
    """One data/raw_videos/ filename -> one *_landmarks.json per instance in
    out_dir. Returns the written filenames (empty if the file has no
    vocab.txt-recognized instance, e.g. a partial-stream artifact).
    """
    instances = lookup(filename)
    if not instances:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for video_id, records in instance_records(raw_dir / filename, instances):
        label = records[0]["label"]
        out_path = out_dir / "{}__{}_landmarks.json".format(label, video_id)
        out_path.write_text(json.dumps(records))
        written.append(out_path.name)
    return written


def _load_completed(out_dir: Path) -> set[str]:
    log_path = out_dir / COMPLETED_LOG
    if not log_path.exists():
        return set()
    return set(log_path.read_text().splitlines())


def _mark_completed(out_dir: Path, filename: str) -> None:
    # Only the main process (single-threaded, driving as_completed below)
    # ever appends here -- worker processes just return results -- so
    # concurrent writers/interleaved lines aren't a concern.
    with open(out_dir / COMPLETED_LOG, "a") as f:
        f.write(filename + "\n")


def pending_files(filenames: list[str], out_dir: Path = OUT_DIR) -> list[str]:
    """Filenames not already recorded as successfully processed in out_dir's
    completed-log -- this is what makes re-running process_batch after an
    interruption resume instead of redoing the whole (multi-hour) batch.
    Failed files aren't logged, so they're retried on every run; that's fine,
    a bad file fails fast (cv2 can't even open it).
    """
    completed = _load_completed(out_dir)
    return [f for f in filenames if f not in completed]


def process_batch(filenames: list[str], workers: int, raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> None:
    """Fan the pending subset of `filenames` out across `workers` processes,
    each running its own MediaPipe instance (mirrors process_video.process_dir).
    One bad file (corrupt download, unreadable codec) fails its own future and
    gets logged -- it doesn't take down the batch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = pending_files(filenames, out_dir)
    skipped = len(filenames) - len(pending)
    if skipped:
        print(f"skipping {skipped} already-completed file(s)")
    if not pending:
        print("nothing pending")
        return

    t0 = time.time()
    ok, failed, total_instances = 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_and_save, f, raw_dir, out_dir): f for f in pending}
        for i, future in enumerate(as_completed(futures), 1):
            f = futures[future]
            try:
                written = future.result()
                total_instances += len(written)
                ok += 1
                _mark_completed(out_dir, f)
                print(f"[{i}/{len(pending)}] {f} -> {len(written)} instance(s)")
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(pending)}] FAILED {f}: {e}")
    elapsed = time.time() - t0
    print(f"\n{ok} ok, {failed} failed, {total_instances} instances written, "
          f"{elapsed:.1f}s for {len(pending)} files ({elapsed / len(pending):.2f}s/file avg)")


def _self_check():
    import tempfile

    import cv2
    import numpy as np

    # instance_records on a synthetic 6-frame clip cut into two 3-frame
    # instances with different labels -- proves the frame-range slicing and
    # per-instance video-id tagging without needing a real (possibly long)
    # multi-label source video.
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / "clip.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
        for _ in range(6):
            writer.write(blank)
        writer.release()

        fake_instances = [
            {"label": "a", "frame_start": 0, "frame_end": 2},
            {"label": "b", "frame_start": 3, "frame_end": 5},
        ]
        results = instance_records(video_path, fake_instances)
        assert len(results) == 2
        (id_a, records_a), (id_b, records_b) = results
        assert id_a != id_b  # distinct video ids even though they share a source file
        assert [r["frame"] for r in records_a] == [0, 1, 2]
        assert [r["frame"] for r in records_b] == [3, 4, 5]
        assert all(r["label"] == "a" for r in records_a)
        assert all(r["label"] == "b" for r in records_b)
        assert all(r["video"] == id_a for r in records_a)

    # process_and_save against a real raw_videos clip end to end: real label
    # lookup, real MediaPipe run, real file written.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        written = process_and_save("69241.mp4", out_dir=out_dir)
        assert len(written) == 1
        assert written[0].startswith("book__")

        saved = json.loads((out_dir / written[0]).read_text())
        assert len(saved) > 0
        assert all(r["label"] == "book" for r in saved)
        assert all(r["video"] == saved[0]["video"] for r in saved)
        assert [r["frame"] for r in saved] == sorted(r["frame"] for r in saved)

        # a file with no vocab.txt-recognized instance (or a partial-stream
        # artifact) writes nothing instead of erroring.
        assert process_and_save("not_a_real_file_12345.mp4", out_dir=out_dir) == []

    # resumability: a file process_batch already completed must be skipped on
    # a second pass -- this is what lets an interrupted multi-hour run resume
    # instead of redoing everything.
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        assert pending_files(["69241.mp4", "05298.mp4"], out_dir) == ["69241.mp4", "05298.mp4"]

        process_batch(["69241.mp4"], workers=1, out_dir=out_dir)
        assert pending_files(["69241.mp4", "05298.mp4"], out_dir) == ["05298.mp4"]

        # a second process_batch call over the same list only has "05298.mp4"
        # left pending -- "69241.mp4" doesn't get reprocessed.
        process_batch(["69241.mp4", "05298.mp4"], workers=1, out_dir=out_dir)
        assert pending_files(["69241.mp4", "05298.mp4"], out_dir) == []

    print("self-check passed: multi-instance frame slicing stays separate per label, "
          "a real raw_videos clip extracts + saves end to end with the right label, "
          "and a completed file is skipped on resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                         help="parallel MediaPipe worker processes")
    parser.add_argument("--self-test", action="store_true",
                         help="run wiring smoke test instead of processing data/raw_videos")
    args = parser.parse_args()

    if args.self_test:
        _self_check()
    else:
        filenames = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".mp4"))
        print(f"{len(filenames)} raw_videos files found, {args.workers} workers")
        process_batch(filenames, workers=args.workers)
