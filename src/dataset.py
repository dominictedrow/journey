"""Turn ingested MediaPipe records (from ingest.load_records) into batched
tensors ready for an LSTM. Each sequence is one whole video (one sign
performance), not an arbitrary chunk of frames.

Four body parts feed the model, in a fixed slot order: right hand, left
hand, face (468 FaceMesh points), pose (33 BlazePose points). Coordinate
normalization: MediaPipe's `landmarks` are already image-relative (0-1), but
that's relative to the camera frame, not the body part -- the same sign
performed in a different spot in frame, closer/further from the camera, or
tilted at a different angle, produces different raw numbers. Every part gets
the exact same three-step fix (translate, scale, rotate) -- only the
anatomical reference points change, since "wrist" and "MCP" only mean
something for a hand:
  1. Subtract a part-specific origin (hand: wrist; face: nose tip; pose:
     shoulder midpoint) from every point -> position in the frame no longer
     matters (translation invariance).
  2. Divide by a part-specific reference distance (hand: wrist-to-middle-
     finger-MCP; face: eye-to-eye; pose: shoulder-to-shoulder) -> part size /
     distance from camera no longer matters, and the numbers land in a
     small, consistent range (this *is* the scaling + normalization).
  3. Rotate so that same reference vector always points the same direction
     (hand: up; face/pose: horizontal) -> in-image-plane tilt no longer
     matters.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ingest import cache_path_for, load_video

VOCAB_PATH = Path(__file__).parent / "vocab.txt"
WRIST = 0
MIDDLE_FINGER_MCP = 9
RIGHT, LEFT = "Right", "Left"

# MediaPipe Pose Landmarker's 33-point topology -- only the shoulders are
# used, as a stable reference frame for the whole upper body.
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12

# MediaPipe FaceMesh's 468-point topology -- nose tip + outer eye corners,
# the standard reference points for face alignment.
FACE_NOSE_TIP = 1
FACE_RIGHT_EYE_OUTER = 33
FACE_LEFT_EYE_OUTER = 263

HAND_LANDMARKS = 21
FACE_LANDMARKS = 468
POSE_LANDMARKS = 33
HAND_FEATURE_DIM = HAND_LANDMARKS * 3
FACE_FEATURE_DIM = FACE_LANDMARKS * 3
POSE_FEATURE_DIM = POSE_LANDMARKS * 3

# Fixed slot layout for one frame's fused feature vector -- hands keep the
# same placement they had before face/pose existed, face and pose are
# appended after.
RIGHT_HAND_SLOT = slice(0, HAND_FEATURE_DIM)
LEFT_HAND_SLOT = slice(HAND_FEATURE_DIM, 2 * HAND_FEATURE_DIM)
FACE_SLOT = slice(LEFT_HAND_SLOT.stop, LEFT_HAND_SLOT.stop + FACE_FEATURE_DIM)
POSE_SLOT = slice(FACE_SLOT.stop, FACE_SLOT.stop + POSE_FEATURE_DIM)
TOTAL_FEATURE_DIM = POSE_SLOT.stop  # 1629 = (21+21+468+33) landmarks * 3 coords
TOTAL_LANDMARKS = TOTAL_FEATURE_DIM // 3  # 543

# Clips run 14-384 frames (median ~80); 60 is the fixed step count every
# clip gets resampled to in _build_sequences, short or long.
SEQUENCE_LENGTH = 60

# Train-only augmentation: small random rotation/scale/jitter around the
# already-normalized coordinates, so the model sees many plausible variants
# of each real clip instead of memorizing one fixed point-cloud per example.
AUGMENT_ROTATION_DEG = 15.0
AUGMENT_SCALE_RANGE = 0.1
AUGMENT_JITTER_STD = 0.02
AUGMENT_TEMPORAL_MIN_KEEP = 0.7  # smallest fraction of timesteps kept before restretching back to seq_len


def _points_array(landmarks: list[dict]) -> np.ndarray:
    """N {x,y,z} dicts -> (N, 3) float32 array."""
    return np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)


def _normalize(points: np.ndarray, origin: np.ndarray, axis_vector: np.ndarray, target_angle: float) -> np.ndarray:
    """Shared translate/scale/rotate step used by every body part: recenter
    on `origin`, scale so `axis_vector` has length 1, then rotate so
    `axis_vector` points at `target_angle`. Only the anatomical reference
    points differ between parts -- this geometry is identical for all of them.

    ponytail: only corrects in-plane (x/y) rotation, not full 3D orientation.
    A second reference axis would be needed to also remove roll around the
    origin->axis_vector axis, and MediaPipe's z is a rougher estimate than
    x/y anyway, so a full 3D fix wouldn't be very trustworthy here.
    """
    points = points - origin
    scale = max(float(np.linalg.norm(axis_vector)), 1e-6)
    points = points / scale

    angle = np.arctan2(axis_vector[1], axis_vector[0])
    theta = target_angle - angle
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rotation.T
    return points


def normalize_landmarks(landmarks: list[dict]) -> np.ndarray:
    """21 {x,y,z} dicts (one hand) -> (21, 3): wrist-relative, scaled by the
    wrist-to-middle-finger-MCP distance, rotated so that vector points up."""
    points = _points_array(landmarks)
    return _normalize(points, points[WRIST], points[MIDDLE_FINGER_MCP] - points[WRIST], target_angle=-np.pi / 2)


def normalize_pose(landmarks: list[dict]) -> np.ndarray:
    """33 {x,y,z} dicts (BlazePose) -> (33, 3): shoulder-midpoint-relative,
    scaled by shoulder width, rotated so the shoulder line is horizontal."""
    points = _points_array(landmarks)
    origin = (points[POSE_LEFT_SHOULDER] + points[POSE_RIGHT_SHOULDER]) / 2
    axis_vector = points[POSE_RIGHT_SHOULDER] - points[POSE_LEFT_SHOULDER]
    return _normalize(points, origin, axis_vector, target_angle=0.0)


def normalize_face(landmarks: list[dict]) -> np.ndarray:
    """468 {x,y,z} dicts (FaceMesh) -> (468, 3): nose-tip-relative, scaled by
    inter-eye distance, rotated so the eye line is horizontal."""
    points = _points_array(landmarks)
    origin = points[FACE_NOSE_TIP]
    axis_vector = points[FACE_LEFT_EYE_OUTER] - points[FACE_RIGHT_EYE_OUTER]
    return _normalize(points, origin, axis_vector, target_angle=0.0)


# extract_kaggle_landmarks.py tags every record with "part" directly
# ("right_hand"/"left_hand"/"face"/"pose", straight from Kaggle's own parquet
# column). process_video.py's live hand-only recordings predate that field
# and use "handedness" ("Right"/"Left"/None) instead -- _PART_FROM_HANDEDNESS
# maps those onto the same part names so fuse_landmarks can treat both record
# shapes identically.
_PART_FROM_HANDEDNESS = {RIGHT: "right_hand", LEFT: "left_hand"}
_NORMALIZE_BY_PART = {
    "right_hand": normalize_landmarks,
    "left_hand": normalize_landmarks,
    "face": normalize_face,
    "pose": normalize_pose,
}
_SLOT_BY_PART = {
    "right_hand": RIGHT_HAND_SLOT,
    "left_hand": LEFT_HAND_SLOT,
    "face": FACE_SLOT,
    "pose": POSE_SLOT,
}


def fuse_landmarks(frame_records: list[dict]) -> np.ndarray:
    """Combine every body part detected in one frame into one fixed
    TOTAL_FEATURE_DIM-vector.

    Each part is normalized independently (its own origin, its own scale,
    its own rotation -- one part's geometry has nothing to do with
    another's). Slot order is fixed by part, not detection order: a right
    hand's numbers always go in RIGHT_HAND_SLOT, a face's in FACE_SLOT, etc.
    A missing part's slot stays zero -- fixed slots mean "which part" is
    already fully encoded by position, no separate part flag is needed on
    the model's input side.

    A record naming no known part (e.g. process_video.py's placeholder for a
    frame where nothing was detected, handedness=None) falls through
    untouched, leaving its slot -- or the whole vector, if it's the only
    record for that frame -- zero.
    """
    fused = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
    for r in frame_records:
        part = r.get("part") or _PART_FROM_HANDEDNESS.get(r.get("handedness"))
        if part in _SLOT_BY_PART:
            fused[_SLOT_BY_PART[part]] = _NORMALIZE_BY_PART[part](r["landmarks"]).reshape(-1)
    return fused


def augment_sequence(coords: np.ndarray) -> np.ndarray:
    """Randomly perturb one (seq_len, TOTAL_FEATURE_DIM) fused sequence: one
    shared random rotation + scale for the whole clip (mimics a different
    camera angle/distance), independent per-landmark noise per frame (mimics
    MediaPipe's own detection jitter), and a random temporal resample (mimics
    a signer performing faster, or MediaPipe dropping frames).

    A part's slots are all-zero when fuse_landmarks found no detection for it
    that frame -- that zero means "absent", not "at the origin". Rotation/
    scale are linear so they leave zero at zero, but additive noise would
    turn an absent part into a fake noisy one, so noise is masked out
    wherever a landmark is absent.
    """
    seq_len = coords.shape[0]
    # Temporal: drop a random fraction of timesteps, then restretch back to
    # seq_len via the same linspace resampling every sequence is already
    # built with (resample_sequence). Fewer real frames stretched over the
    # same length plays back like a faster performance, so the model sees
    # speed variation instead of one fixed-speed resample per clip.
    keep_frac = np.random.uniform(AUGMENT_TEMPORAL_MIN_KEEP, 1.0)
    n_keep = max(2, round(seq_len * keep_frac))
    keep_idx = np.sort(np.random.choice(seq_len, n_keep, replace=False))
    coords = resample_sequence(list(coords[keep_idx]), seq_len)

    coords = coords.reshape(coords.shape[0], TOTAL_LANDMARKS, 3).copy()  # (seq, landmark, xyz)
    present = np.abs(coords).sum(axis=-1, keepdims=True) > 0  # (seq, landmark, 1)

    angle = np.radians(np.random.uniform(-AUGMENT_ROTATION_DEG, AUGMENT_ROTATION_DEG))
    cos_t, sin_t = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    coords[..., :2] = coords[..., :2] @ rotation.T

    scale = np.random.uniform(1 - AUGMENT_SCALE_RANGE, 1 + AUGMENT_SCALE_RANGE)
    coords *= scale

    noise = np.random.normal(0, AUGMENT_JITTER_STD, coords.shape).astype(np.float32)
    coords += noise * present

    return coords.reshape(coords.shape[0], TOTAL_FEATURE_DIM)


def load_vocab(path: Path = VOCAB_PATH) -> list[str]:
    """Fixed label order, one word per line -> index i always means the same word."""
    return path.read_text().splitlines()


def one_hot(label: str, vocab: list[str]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    vec[vocab.index(label)] = 1.0
    return vec


def resample_sequence(coords: list[np.ndarray], sequence_length: int) -> np.ndarray:
    """Uniformly resample a variable-length list of fused per-frame vectors down/up to
    exactly sequence_length steps, spanning the whole list start to finish. Shared by
    _build_sequences (training clips) and live inference (mac.py's segmented gestures)
    so both go through the identical normalization."""
    indices = np.linspace(0, len(coords) - 1, sequence_length).round().astype(int)
    return np.stack([coords[i] for i in indices])


def _video_to_sequence(video_records: list[dict], sequence_length: int) -> np.ndarray | None:
    """One video's records -> one (sequence_length, TOTAL_FEATURE_DIM) array:
    group by frame, fuse each frame's detections, resample to sequence_length
    steps. None if the video had no frames at all."""
    frames = defaultdict(list)
    for r in video_records:
        frames[r["frame"]].append(r)

    coords = [fuse_landmarks(frames[frame_idx]) for frame_idx in sorted(frames)]
    if not coords:
        return None

    return resample_sequence(coords, sequence_length)


def _build_sequences(records: list[dict], sequence_length: int) -> list[tuple[np.ndarray, str]]:
    """One sequence per video, not per label.

    Grouping has to start from "video", not "label": two different videos of
    the same word both have a frame 0, a frame 1, etc, and those are
    unrelated moments from different performances. Grouping by label first
    would pool every video sharing a word into one frame-indexed bucket,
    silently collapsing all of them into a single garbled sequence. Keying by
    video first keeps every performance -- however many share a word -- as
    its own separate example.

    Videos vary in length depending on how long the sign took to perform, but
    an LSTM batch needs a fixed number of steps. Instead of chopping a video
    into arbitrary sequence_length-sized pieces (which would slice a single
    sign performance into several pieces, each wrongly stamped with the full
    sign's label), every video's frames are uniformly resampled down/up to
    exactly sequence_length steps -- so the whole performance is always
    covered, whether the source video was shorter or longer than that.
    """
    by_video = defaultdict(list)
    for r in records:
        by_video[r["video"]].append(r)

    sequences = []
    for video_records in by_video.values():
        label = video_records[0]["label"]
        coords = _video_to_sequence(video_records, sequence_length)
        if coords is None:
            continue
        sequences.append((coords, label))

    return sequences


class SignSequenceDataset(Dataset):
    """One item = one video, resampled to (sequence_length, TOTAL_FEATURE_DIM) + one-hot label."""

    def __init__(self, records: list[dict], vocab: list[str], sequence_length: int = SEQUENCE_LENGTH,
                 augment: bool = False):
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.augment = augment
        self.sequences = _build_sequences(records, sequence_length)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        coords, label = self.sequences[idx]
        if self.augment:
            coords = augment_sequence(coords)
        x = torch.from_numpy(coords)
        y = torch.from_numpy(one_hot(label, self.vocab))
        return x, y


def make_dataloader(records: list[dict], vocab: list[str], sequence_length: int = SEQUENCE_LENGTH,
                     batch_size: int = 16, shuffle: bool = True, num_workers: int = 8,
                     augment: bool = False) -> DataLoader:
    # num_workers spawns subprocesses that prepare upcoming batches (landmark
    # normalization/resampling/augmentation) while the GPU is busy on the
    # current one -- without this, that CPU work happens in-line on the main
    # process and the GPU sits idle between batches instead of overlapping with it.
    # persistent_workers keeps those subprocesses alive across epochs instead
    # of respawning them every epoch (Windows' spawn start method makes that
    # respawn expensive). pin_memory lets the host->GPU copy for the next
    # batch happen asynchronously instead of blocking the training step.
    dataset = SignSequenceDataset(records, vocab, sequence_length, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                       persistent_workers=num_workers > 0, pin_memory=True)


class LazySignVideoDataset(Dataset):
    """Same per-video (sequence_length, TOTAL_FEATURE_DIM) + one-hot label
    contract as SignSequenceDataset, but reads and fuses each video from disk
    only when __getitem__ actually asks for it, instead of building
    self.sequences for the whole corpus up front at construction time. Trades
    memory (flat, regardless of corpus size) for repeated per-epoch disk I/O.

    files: (path, label) pairs, e.g. from ingest.list_video_files. A file
    that already has a build_landmark_cache.py .npy cache next to it (see
    ingest.cache_path_for) is read from that instead of the source JSON --
    np.load on a flat binary array is far cheaper than json.loads on the
    same video's text, and this is why lazy loading doesn't have to mean
    "re-parse everything every epoch". Falls back to the JSON + fuse/resample
    path when no cache exists yet, or when a cache's sequence_length doesn't
    match this dataset's (stale cache from a different SEQUENCE_LENGTH).
    """

    def __init__(self, files: list[tuple[Path, str]], vocab: list[str], sequence_length: int = SEQUENCE_LENGTH,
                 augment: bool = False):
        self.files = files
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path, label = self.files[idx]
        coords = None
        cache_path = cache_path_for(path)
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape[0] == self.sequence_length:
                coords = cached
        if coords is None:
            coords = _video_to_sequence(load_video(path, label), self.sequence_length)
            if coords is None:
                raise ValueError(f"{path} has no frames -- every written landmarks file should have at least one")
        if self.augment:
            coords = augment_sequence(coords)
        x = torch.from_numpy(coords)
        y = torch.from_numpy(one_hot(label, self.vocab))
        return x, y


def make_lazy_dataloader(files: list[tuple[Path, str]], vocab: list[str], sequence_length: int = SEQUENCE_LENGTH,
                          batch_size: int = 16, shuffle: bool = True, num_workers: int = 8,
                          augment: bool = False) -> DataLoader:
    # Mirrors make_dataloader exactly, except each of the num_workers spawned
    # Windows worker processes now pickles this small (Path, str) list instead
    # of a multi-GB pre-fused sequence array -- the same laziness that keeps
    # load_video from ever materializing the whole corpus in one process also
    # fixes the earlier per-worker duplication problem make_dataloader hit.
    dataset = LazySignVideoDataset(files, vocab, sequence_length, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                       persistent_workers=num_workers > 0, pin_memory=True)


def _self_check():
    INDEX_FINGER_MCP = 5

    def hand(video, label, handedness, frame, index_x_delta=0.0):
        # wrist=landmark 0 (origin), MCP=landmark 9 fixes scale+rotation for
        # every hand identically -- vary a third landmark (index MCP) so
        # right vs. left hands actually end up geometrically distinguishable
        # after normalization, instead of testing two hands with the same shape.
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
        landmarks[MIDDLE_FINGER_MCP] = {"x": 0.5, "y": 0.6, "z": 0.0}
        landmarks[INDEX_FINGER_MCP] = {"x": 0.5 + index_x_delta, "y": 0.55, "z": 0.0}
        return {"video": video, "label": label, "handedness": handedness, "frame": frame, "landmarks": landmarks}

    def pose(video, label, frame, x_shift=0.0):
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(33)]
        landmarks[POSE_LEFT_SHOULDER] = {"x": 0.4 + x_shift, "y": 0.3, "z": 0.0}
        landmarks[POSE_RIGHT_SHOULDER] = {"x": 0.6 + x_shift, "y": 0.3, "z": 0.0}
        return {"video": video, "label": label, "part": "pose", "frame": frame, "landmarks": landmarks}

    def face(video, label, frame, x_shift=0.0):
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(468)]
        landmarks[FACE_NOSE_TIP] = {"x": 0.5, "y": 0.5, "z": 0.0}
        landmarks[FACE_RIGHT_EYE_OUTER] = {"x": 0.45 + x_shift, "y": 0.4, "z": 0.0}
        landmarks[FACE_LEFT_EYE_OUTER] = {"x": 0.55 + x_shift, "y": 0.4, "z": 0.0}
        return {"video": video, "label": label, "part": "face", "frame": frame, "landmarks": landmarks}

    # two separate "book" videos (different signers/takes) -- must stay two
    # separate sequences even though they share a label.
    records = [hand("book_a", "book", RIGHT, f) for f in range(5)]
    records += [hand("book_b", "book", RIGHT, f, index_x_delta=0.05) for f in range(4)]
    for f in range(8):  # two-handed sign, each hand a different shape
        records.append(hand("go_a", "go", RIGHT, f, index_x_delta=0.02))
        records.append(hand("go_a", "go", LEFT, f, index_x_delta=-0.02))

    # a third "book" video with a completely handless middle frame (process_video's
    # placeholder shape: handedness=None, no landmarks) -- must zero-fill that
    # frame, not drop it and shrink the sequence.
    records.append(hand("book_c", "book", RIGHT, 0))
    records.append({"video": "book_c", "label": "book", "handedness": None, "frame": 1, "landmarks": []})
    records.append(hand("book_c", "book", RIGHT, 2))

    vocab = load_vocab()
    assert len(vocab) == 250
    assert vocab[28] == "book" and vocab[97] == "go"  # fixed Kaggle asl-signs indices, not alphabetical
    one_hot_go = one_hot("go", vocab)
    assert one_hot_go.sum() == 1.0 and one_hot_go[97] == 1.0

    norm = normalize_landmarks(records[0]["landmarks"])
    assert norm[WRIST].tolist() == [0.0, 0.0, 0.0]

    # rotation check: MCP starting out pointing right instead of down should
    # still land on the same canonical position after normalization.
    mirrored = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    mirrored[MIDDLE_FINGER_MCP] = {"x": 0.6, "y": 0.5, "z": 0.0}
    norm_mirrored = normalize_landmarks(mirrored)
    assert np.allclose(norm[MIDDLE_FINGER_MCP], norm_mirrored[MIDDLE_FINGER_MCP], atol=1e-5)

    # pose/face use the same translate/scale/rotate technique, just anchored
    # on different landmarks -- verify each anchor lands where it should.
    norm_pose = normalize_pose(pose("p", "book", 0)["landmarks"])
    assert np.allclose((norm_pose[POSE_LEFT_SHOULDER] + norm_pose[POSE_RIGHT_SHOULDER]) / 2, 0.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(norm_pose[POSE_RIGHT_SHOULDER] - norm_pose[POSE_LEFT_SHOULDER]), 1.0)

    norm_face = normalize_face(face("f", "book", 0)["landmarks"])
    assert np.allclose(norm_face[FACE_NOSE_TIP], 0.0, atol=1e-5)
    assert np.isclose(np.linalg.norm(norm_face[FACE_LEFT_EYE_OUTER] - norm_face[FACE_RIGHT_EYE_OUTER]), 1.0)

    # sequence check: one sequence per video (not per frame-chunk, and not
    # merged with other videos sharing a label), fused to a fixed
    # TOTAL_FEATURE_DIM-wide vector (hands first, unchanged placement).
    dataset = SignSequenceDataset(records, vocab, sequence_length=3)
    assert len(dataset) == 4  # book_a, book_b, book_c, go_a -- all three book videos stay separate

    seqs_by_label = defaultdict(list)
    for coords, label in dataset.sequences:
        seqs_by_label[label].append(coords)

    book_seqs = seqs_by_label["book"]
    assert len(book_seqs) == 3  # none of the three book videos collapsed into another
    assert not np.allclose(book_seqs[0], book_seqs[1])  # different takes, different coords
    for book_seq in book_seqs:
        assert book_seq.shape == (3, TOTAL_FEATURE_DIM)
        assert not np.allclose(book_seq[0, RIGHT_HAND_SLOT], 0.0)   # right hand present
        assert np.allclose(book_seq[0, LEFT_HAND_SLOT], 0.0)        # left hand absent -> zero-filled
        assert np.allclose(book_seq[0, FACE_SLOT], 0.0)             # face never provided -> zero-filled
        assert np.allclose(book_seq[0, POSE_SLOT], 0.0)             # pose never provided -> zero-filled

    # book_c's middle frame had no hand at all (process_video's None-handedness
    # placeholder) -- it must show up as a zero row, not vanish and pull frame 2
    # into frame 1's slot.
    book_c_seq = book_seqs[2]
    assert np.allclose(book_c_seq[1], 0.0)                          # handless frame -> all zero
    assert not np.allclose(book_c_seq[0, RIGHT_HAND_SLOT], 0.0)     # frame 0's hand still there
    assert not np.allclose(book_c_seq[2, RIGHT_HAND_SLOT], 0.0)     # frame 2's hand still there

    go_seq = seqs_by_label["go"][0]
    assert not np.allclose(go_seq[0, RIGHT_HAND_SLOT], 0.0)   # right hand present
    assert not np.allclose(go_seq[0, LEFT_HAND_SLOT], 0.0)    # left hand present too
    assert not np.allclose(go_seq[0, RIGHT_HAND_SLOT], go_seq[0, LEFT_HAND_SLOT])  # not the same values

    # a video with all four parts tagged via Kaggle's "part" field (not
    # "handedness") -- every part must land in its own slot in one fused row.
    all_parts_records = [
        hand("all_a", "chair", RIGHT, 0),
        {"video": "all_a", "label": "chair", "frame": 0, "part": "left_hand",
         "landmarks": hand("x", "x", LEFT, 0, index_x_delta=-0.02)["landmarks"]},
        pose("all_a", "chair", 0),
        face("all_a", "chair", 0),
    ]
    all_parts_seq = SignSequenceDataset(all_parts_records, vocab, sequence_length=1).sequences[0][0]
    assert not np.allclose(all_parts_seq[0, RIGHT_HAND_SLOT], 0.0)
    assert not np.allclose(all_parts_seq[0, LEFT_HAND_SLOT], 0.0)
    assert not np.allclose(all_parts_seq[0, FACE_SLOT], 0.0)
    assert not np.allclose(all_parts_seq[0, POSE_SLOT], 0.0)

    loader = make_dataloader(records, vocab, sequence_length=3, batch_size=2, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == (2, 3, TOTAL_FEATURE_DIM)
    assert y.shape[1] == len(vocab)

    # augmentation: a present part should move (rotation/scale/noise all
    # apply), an absent part must stay exactly zero -- not get hallucinated
    # noise -- and the un-augmented dataset must be untouched (val/test path).
    np.random.seed(0)
    augmented = augment_sequence(book_seqs[0])
    assert not np.allclose(augmented[0, RIGHT_HAND_SLOT], book_seqs[0][0, RIGHT_HAND_SLOT])  # right hand: perturbed
    assert np.allclose(augmented[0, LEFT_HAND_SLOT], 0.0)   # left hand: still absent -> still zero
    assert np.allclose(augmented[0, FACE_SLOT], 0.0)        # face: still absent -> still zero
    assert np.allclose(augmented[0, POSE_SLOT], 0.0)        # pose: still absent -> still zero

    plain_dataset = SignSequenceDataset(records, vocab, sequence_length=3, augment=False)
    aug_dataset = SignSequenceDataset(records, vocab, sequence_length=3, augment=True)
    plain_x, _ = plain_dataset[0]
    aug_x, _ = aug_dataset[0]
    assert not torch.allclose(plain_x, aug_x)  # augment=True actually perturbs __getitem__ output

    # Lazy path (list_video_files -> LazySignVideoDataset) must match the eager
    # path (load_records -> SignSequenceDataset) exactly, for both record
    # shapes: pre-tagged (Kaggle-style filename) and untagged (personal-
    # recording filename, label inferred from the name).
    import json
    import tempfile
    from ingest import list_video_files

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        book_c_records = [
            hand("book_c", "book", RIGHT, 0),
            {"video": "book_c", "label": "book", "handedness": None, "frame": 1, "landmarks": []},
            hand("book_c", "book", RIGHT, 2),
        ]
        (tmp_path / "book__1-1_landmarks.json").write_text(json.dumps(book_c_records))

        hello_records = [{
            "frame": 0, "handedness": "Right",
            "landmarks": [{"x": 0.1, "y": 0.2, "z": 0.0}] * 21,
        }]
        (tmp_path / "hello_landmarks.json").write_text(json.dumps(hello_records))

        files = list_video_files(tmp_path)
        lazy_dataset = LazySignVideoDataset(files, vocab, sequence_length=3)
        assert not hasattr(lazy_dataset, "sequences")

        for i, (path, label) in enumerate(files):
            eager_records = load_video(path, label)
            eager_x, eager_y = SignSequenceDataset(eager_records, vocab, sequence_length=3)[0]
            lazy_x, lazy_y = lazy_dataset[i]
            assert torch.allclose(lazy_x, eager_x)
            assert torch.allclose(lazy_y, eager_y)
            if path.name == "book__1-1_landmarks.json":
                assert torch.allclose(lazy_x[1], torch.zeros(TOTAL_FEATURE_DIM))  # handless frame -> zero-filled

        # Cache hit: once a per-video .npy cache exists (as build_landmark_cache.py
        # writes), LazySignVideoDataset must read that directly instead of
        # re-parsing JSON -- deleting the source file after caching and still
        # getting the right tensor is the proof, not just "the cache looks right".
        book_path, book_label = files[0]
        assert book_path.name == "book__1-1_landmarks.json"
        book_coords = _video_to_sequence(load_video(book_path, book_label), sequence_length=3)
        np.save(cache_path_for(book_path), book_coords)
        book_path.unlink()  # source JSON gone -- only the cache remains

        cached_dataset = LazySignVideoDataset(files, vocab, sequence_length=3)
        cached_x, _ = cached_dataset[0]
        assert torch.allclose(cached_x, torch.from_numpy(book_coords))

        # A cache built at a different sequence_length must be ignored, not
        # served with the wrong shape -- falls back to recomputing from JSON.
        hello_path, _ = files[1]
        assert hello_path.name == "hello_landmarks.json"
        np.save(cache_path_for(hello_path), np.zeros((99, TOTAL_FEATURE_DIM), dtype=np.float32))
        guarded_x, _ = cached_dataset[1]
        assert guarded_x.shape == (3, TOTAL_FEATURE_DIM)  # recomputed, not served the (99, ...) cache

        lazy_loader = make_lazy_dataloader(files, vocab, sequence_length=3, batch_size=2, shuffle=False,
                                            num_workers=0)
        lazy_batch_x, lazy_batch_y = next(iter(lazy_loader))
        assert lazy_batch_x.shape[1:] == (3, TOTAL_FEATURE_DIM)
        assert lazy_batch_y.shape[1] == len(vocab)

    print("self-check passed: per-part normalization (hand/face/pose sharing the same "
          "translate/scale/rotate technique on different anchors), fixed 4-part slot fusion, "
          "per-video sequences, train-only augmentation (perturbs present parts, leaves "
          "absent parts zero), the lazy per-video path (list_video_files -> "
          "LazySignVideoDataset) matching the eager path exactly, and its .npy cache "
          "(hit + stale-shape fallback) all line up")


if __name__ == "__main__":
    _self_check()
