"""Convert Kaggle's asl-signs parquet landmark files into ingest.py-compatible
JSON records. Replaces extract_raw_videos.py's role for this corpus: Kaggle's
landmarks are already extracted (MediaPipe Holistic), so no MediaPipe run
happens here, just reshaping.

Kaggle's Holistic parquet mixes face/pose/hand rows per frame; all four are
kept now (previously only left_hand/right_hand). landmark_index 0-20 for a
hand matches MediaPipe HandLandmarker's own numbering (wrist=0, middle-finger
MCP=9), landmark_index 0-32 for pose matches BlazePose's, and 0-467 for face
matches FaceMesh's, so dataset.py's normalization and its LIP_INDICES/POSE_KEEP
subsets need no changes here -- this file only reshapes, and stores all 468 face
points even though only the 40 lip ones survive fusion. A part not
detected in a frame comes through as all-NaN rows rather than being absent,
so a (frame, part) group is only kept when every one of its landmarks is
present; a part that's NaN a given frame simply produces no record for that
part that frame (dropped, not zero-filled -- dataset.fuse_landmarks is what
zero-fills a missing part, same as it already does for a missing hand).

Output goes to data/raw_landmarks/, one JSON per sequence, named
"{sign}__{participant_id}-{sequence_id}_landmarks.json" -- participant_id is
baked into the filename so split_dataset.py can group by signer directly,
with no separate lookup step needed (see split_dataset.py's _signer_key).
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.parent
KAGGLE_DIR = HERE / "data" / "kaggle_raw"
TRAIN_CSV = KAGGLE_DIR / "train.csv"
OUT_DIR = HERE / "data" / "raw_landmarks"
COMPLETED_LOG = "_completed.txt"

# Kaggle's own "type" column values double as dataset.fuse_landmarks' "part"
# field directly -- no renaming/mapping needed. Each maps to how many
# landmark_index rows that type must have every frame it's kept.
PART_LANDMARK_COUNTS = {"right_hand": 21, "left_hand": 21, "pose": 33, "face": 468}


def sequence_records(parquet_path: Path, video_id: str, label: str) -> list[dict]:
    """One parquet file (one Kaggle sequence) -> ingest.py-shaped records, one per
    (frame, part) that was fully tracked."""
    df = pd.read_parquet(parquet_path, columns=["frame", "type", "landmark_index", "x", "y", "z"])
    df = df[df["type"].isin(PART_LANDMARK_COUNTS)]

    records = []
    for (frame, part), group in df.groupby(["frame", "type"], sort=True):
        group = group.sort_values("landmark_index")
        if len(group) != PART_LANDMARK_COUNTS[part] or group[["x", "y", "z"]].isna().any().any():
            continue  # part not detected this frame -- Kaggle marks that as all-NaN, not an absent row
        records.append({
            "video": video_id,
            "label": label,
            "frame": int(frame),
            "part": part,
            "landmarks": [{"x": r.x, "y": r.y, "z": r.z} for r in group.itertuples()],
        })
    return records


def process_row(parquet_path: Path, video_id: str, label: str, out_dir: Path) -> str | None:
    """One train.csv row -> one *_landmarks.json, or None if nothing was ever tracked."""
    records = sequence_records(parquet_path, video_id, label)
    if not records:
        return None
    out_path = out_dir / "{}__{}_landmarks.json".format(label, video_id)
    out_path.write_text(json.dumps(records))
    return out_path.name


def _load_completed(out_dir: Path) -> set[str]:
    log_path = out_dir / COMPLETED_LOG
    if not log_path.exists():
        return set()
    return set(log_path.read_text().splitlines())


def _mark_completed(out_dir: Path, video_id: str) -> None:
    # Only the main process (single-threaded, driving as_completed below) ever
    # appends here -- worker processes just return results -- so concurrent
    # writers/interleaved lines aren't a concern.
    with open(out_dir / COMPLETED_LOG, "a") as f:
        f.write(video_id + "\n")


def process_batch(rows: list[dict], workers: int, kaggle_dir: Path = KAGGLE_DIR,
                   out_dir: Path = OUT_DIR) -> None:
    """Fan train.csv's rows out across `workers` processes. Resumable: a row already
    recorded in the completed-log is skipped, so an interrupted run just re-runs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_completed(out_dir)

    def video_id_of(row):
        return "{}-{}".format(row["participant_id"], row["sequence_id"])

    pending = [r for r in rows if video_id_of(r) not in completed]
    skipped = len(rows) - len(pending)
    if skipped:
        print(f"skipping {skipped} already-completed row(s)")
    if not pending:
        print("nothing pending")
        return

    t0 = time.time()
    ok, empty, failed = 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for r in pending:
            vid = video_id_of(r)
            futures[pool.submit(process_row, kaggle_dir / r["path"], vid, r["sign"], out_dir)] = vid

        for i, future in enumerate(as_completed(futures), 1):
            vid = futures[future]
            try:
                written = future.result()
                _mark_completed(out_dir, vid)
                if written:
                    ok += 1
                else:
                    empty += 1
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(pending)}] FAILED {vid}: {e}")
            if i % 2000 == 0 or i == len(pending):
                print(f"[{i}/{len(pending)}] {ok} written, {empty} empty, {failed} failed")

    elapsed = time.time() - t0
    print(f"\n{ok} ok, {empty} empty (nothing ever tracked), {failed} failed, "
          f"{elapsed:.1f}s for {len(pending)} rows ({elapsed / len(pending):.3f}s/row avg)")


def _self_check():
    import tempfile

    def landmark_row(frame, row_type, idx, x=None, y=None, z=None):
        return {"frame": frame, "row_id": f"{frame}-{row_type}-{idx}", "type": row_type,
                "landmark_index": idx, "x": x, "y": y, "z": z}

    rows = []
    # frame 0: right hand fully tracked, left hand fully NaN (untracked that frame)
    for i in range(21):
        rows.append(landmark_row(0, "right_hand", i, x=0.1 * i, y=0.2, z=0.0))
        rows.append(landmark_row(0, "left_hand", i))
    # frame 1: nothing tracked -- must vanish entirely, not appear as a zero row
    for i in range(21):
        rows.append(landmark_row(1, "right_hand", i))
        rows.append(landmark_row(1, "left_hand", i))
    # frame 2: left hand, pose, and face all tracked
    for i in range(21):
        rows.append(landmark_row(2, "left_hand", i, x=0.3 * i, y=0.4, z=0.0))
    for i in range(33):
        rows.append(landmark_row(2, "pose", i, x=0.5, y=0.5, z=0.0))
    for i in range(468):
        rows.append(landmark_row(2, "face", i, x=0.6, y=0.6, z=0.0))
    # frame 2's pose is only partially tracked (one landmark short) -> dropped,
    # same as an incomplete hand would be
    rows = [r for r in rows if not (r["frame"] == 2 and r["type"] == "pose" and r["landmark_index"] == 32)]

    df = pd.DataFrame(rows)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parquet_path = tmp_path / "sample.parquet"
        df.to_parquet(parquet_path)

        records = sequence_records(parquet_path, video_id="p1-s1", label="book")
        assert sorted(r["frame"] for r in records) == [0, 2, 2]  # frame 1 (nothing tracked) dropped entirely

        right0 = next(r for r in records if r["frame"] == 0 and r["part"] == "right_hand")
        assert len(right0["landmarks"]) == 21
        assert right0["landmarks"][1]["x"] == 0.1

        assert [r for r in records if r["frame"] == 0 and r["part"] == "left_hand"] == []

        left2 = next(r for r in records if r["frame"] == 2 and r["part"] == "left_hand")
        assert len(left2["landmarks"]) == 21
        assert all(r["video"] == "p1-s1" and r["label"] == "book" for r in records)

        # face fully tracked at frame 2 -> kept; pose short one landmark -> dropped entirely
        assert any(r["frame"] == 2 and r["part"] == "face" and len(r["landmarks"]) == 468 for r in records)
        assert [r for r in records if r["frame"] == 2 and r["part"] == "pose"] == []

        # process_row end to end: writes one JSON named by sign + participant-sequence id
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        written = process_row(parquet_path, "p1-s1", "book", out_dir)
        assert written == "book__p1-s1_landmarks.json"
        saved = json.loads((out_dir / written).read_text())
        assert len(saved) == len(records)

        # an all-NaN parquet (tracking totally failed) -> no file written
        empty_df = pd.DataFrame([landmark_row(0, "right_hand", i) for i in range(21)])
        empty_path = tmp_path / "empty.parquet"
        empty_df.to_parquet(empty_path)
        assert process_row(empty_path, "p2-s2", "book", out_dir) is None

        # resumability: process_batch skips a video_id already in the completed log
        out_dir2 = tmp_path / "out2"
        rows_csv = [{"participant_id": "p1", "sequence_id": "s1", "sign": "book", "path": "sample.parquet"}]
        process_batch(rows_csv, workers=1, kaggle_dir=tmp_path, out_dir=out_dir2)
        assert (out_dir2 / "book__p1-s1_landmarks.json").exists()
        process_batch(rows_csv, workers=1, kaggle_dir=tmp_path, out_dir=out_dir2)  # second call: nothing pending

    print("self-check passed: extraction keeps every fully-tracked (frame, part) -- hands, "
          "pose, and face alike -- drops incomplete ones the same way regardless of part, "
          "writes participant-tagged JSON, and resumes correctly via the completed log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--self-test", action="store_true",
                         help="run wiring smoke test instead of processing train.csv")
    args = parser.parse_args()

    if args.self_test:
        _self_check()
    else:
        train_rows = pd.read_csv(TRAIN_CSV).to_dict("records")
        print(f"{len(train_rows)} rows in train.csv, {args.workers} workers")
        process_batch(train_rows, workers=args.workers)
