"""Turn ingested MediaPipe records (from ingest.load_records) into batched
tensors ready for an LSTM. Each sequence is one whole video (one sign
performance), not an arbitrary chunk of frames.

A sign is defined by five parameters: handshape, location, orientation,
movement, and non-manual markers. The feature layout below exists to put all
five in front of the model, and to keep signer identity out of it:

  A  body frame (282)   both hands, 40 lip points, 12 upper-body pose points,
                        all normalized into ONE shared reference frame (the
                        shoulder line) -- so location and orientation survive
  B  handshape (126)    both hands additionally re-normalized into their own
                        wrist frame -- scale/rotation-invariant handshape
  C  velocity (282)     frame-to-frame delta of block A -- movement, handed
                        over explicitly instead of left for the LSTM to find
  D  presence (4)       right hand / left hand / lips / pose detected? -- so
                        "absent" stops being indistinguishable from "at the
                        reference origin", which zero-filling makes it

Blocks A+B are what gets cached to .npy (COORD_DIM); C and D are derived from
them at load time by finalize_sequence, so they cost no disk and cannot go
stale.

Normalization is one shared translate/scale/rotate step (_normalize):
translate so the reference origin is 0, scale so the reference vector has
length 1, rotate so it points at a fixed angle. Block A runs every part
through it with the *same* origin/axis (shoulder midpoint + shoulder width,
falling back to nose tip + inter-eye distance when pose is missing), which
removes only genuine nuisance -- where the signer sits in frame, distance from
the camera, camera tilt -- while keeping where the hands are relative to the
body, which way they face, and how they travel. Block B runs each hand through
the same step with its own wrist/MCP anchors, which throws location and
orientation away on purpose to leave a clean handshape signal.

The full 468-point face mesh is deliberately NOT here: it is 86% of the input
dims and encodes the signer, not the sign, so on a signer-disjoint split it is
a shortcut feature the model will happily memorize instead of learning ASL.
Only the 40 lip points survive, because mouth morphemes are real ASL grammar.
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

# MediaPipe Pose Landmarker's 33-point topology. Shoulders define the shared
# body reference frame; POSE_KEEP is the upper body (shoulders, elbows,
# wrists, pinkies, index fingers, thumbs) -- legs and facial pose points are
# dropped, they say nothing about a sign. Indices 11-22 alternate
# left, right, left, right..., which is what makes the mirror swap below a
# plain adjacent-pair swap.
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_KEEP = list(range(11, 23))

# MediaPipe FaceMesh's 468-point topology. Nose tip + outer eye corners are
# only used as the fallback reference frame when pose is missing.
FACE_NOSE_TIP = 1
FACE_RIGHT_EYE_OUTER = 33
FACE_LEFT_EYE_OUTER = 263

# The standard FaceMesh lip contours, listed as (left, right) mirror pairs so
# the same list doubles as the mirror map for mirror_sequence. LIP_MIDLINE
# sits on the face's centerline and has no partner.
LIP_PAIRS = [(37, 267), (39, 269), (40, 270), (185, 409), (61, 291),   # outer upper
             (84, 314), (181, 405), (91, 321), (146, 375),             # outer lower
             (82, 312), (81, 311), (80, 310), (191, 415), (78, 308),   # inner upper
             (87, 317), (178, 402), (88, 318), (95, 324)]              # inner lower
LIP_MIDLINE = [0, 17, 13, 14]
LIP_INDICES = [i for pair in LIP_PAIRS for i in pair] + LIP_MIDLINE  # 40 points, paired ones first

HAND_LANDMARKS = 21
LIP_LANDMARKS = len(LIP_INDICES)
POSE_LANDMARKS = len(POSE_KEEP)
HAND_FEATURE_DIM = HAND_LANDMARKS * 3
LIP_FEATURE_DIM = LIP_LANDMARKS * 3
POSE_FEATURE_DIM = POSE_LANDMARKS * 3

# Block A -- everything in the shared body frame.
A_RIGHT_HAND_SLOT = slice(0, HAND_FEATURE_DIM)
A_LEFT_HAND_SLOT = slice(A_RIGHT_HAND_SLOT.stop, A_RIGHT_HAND_SLOT.stop + HAND_FEATURE_DIM)
A_LIPS_SLOT = slice(A_LEFT_HAND_SLOT.stop, A_LEFT_HAND_SLOT.stop + LIP_FEATURE_DIM)
A_POSE_SLOT = slice(A_LIPS_SLOT.stop, A_LIPS_SLOT.stop + POSE_FEATURE_DIM)
BODY_SLOT = slice(0, A_POSE_SLOT.stop)  # 282 = (21+21+40+12) landmarks * 3

# Block B -- the same two hands, re-normalized into their own wrist frame.
B_RIGHT_HAND_SLOT = slice(BODY_SLOT.stop, BODY_SLOT.stop + HAND_FEATURE_DIM)
B_LEFT_HAND_SLOT = slice(B_RIGHT_HAND_SLOT.stop, B_RIGHT_HAND_SLOT.stop + HAND_FEATURE_DIM)
HANDSHAPE_SLOT = slice(BODY_SLOT.stop, B_LEFT_HAND_SLOT.stop)

COORD_DIM = B_LEFT_HAND_SLOT.stop  # 408 -- what fuse_landmarks returns and what .npy caches hold
COORD_LANDMARKS = COORD_DIM // 3   # 136

# The four block-A parts, in slot order. Presence flags (block D) and
# velocity masking (block C) both iterate this, so they can never disagree
# about which columns belong to which part.
PART_SLOTS = (A_RIGHT_HAND_SLOT, A_LEFT_HAND_SLOT, A_LIPS_SLOT, A_POSE_SLOT)
PART_WIDTHS = [s.stop - s.start for s in PART_SLOTS]

# Blocks C and D -- derived by finalize_sequence, never cached.
VELOCITY_SLOT = slice(COORD_DIM, COORD_DIM + BODY_SLOT.stop)
PRESENCE_SLOT = slice(VELOCITY_SLOT.stop, VELOCITY_SLOT.stop + len(PART_SLOTS))
TOTAL_FEATURE_DIM = PRESENCE_SLOT.stop  # 694 = 408 coords + 282 velocity + 4 presence

# Clips run 14-384 frames (median ~80); 60 is the fixed step count every
# clip gets resampled to in _build_sequences, short or long.
SEQUENCE_LENGTH = 60

# Train-only augmentation. Mirroring is the big one: ASL is handedness-
# agnostic, so a left-dominant signer's version of a sign is the mirror image
# of a right-dominant signer's and carries the same label -- asserting that
# symmetry effectively doubles the corpus. Rotation and scale augmentation
# used to live here and were deleted: body-frame normalization pins shoulder
# width to 1 and the shoulder line to horizontal, so a rotated or rescaled
# clip is an input the pipeline will never produce at inference time, and the
# model was spending capacity learning invariance to an impossible variation.
AUGMENT_MIRROR_PROB = 0.5
AUGMENT_JITTER_STD = 0.02
AUGMENT_TEMPORAL_MIN_KEEP = 0.7  # smallest fraction of timesteps kept before restretching back to seq_len


def _points_array(landmarks: list[dict]) -> np.ndarray:
    """N {x,y,z} dicts -> (N, 3) float32 array."""
    return np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)


def _normalize(points: np.ndarray, origin: np.ndarray, axis_vector: np.ndarray, target_angle: float) -> np.ndarray:
    """Shared translate/scale/rotate step: recenter on `origin`, scale so
    `axis_vector` has length 1, then rotate so `axis_vector` points at
    `target_angle`. Block A passes the whole frame's shared reference here;
    block B passes each hand's own wrist reference. Same geometry either way,
    only the anchors differ.

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


def normalize_landmarks(landmarks: list[dict] | np.ndarray) -> np.ndarray:
    """One hand (21 {x,y,z} dicts, or an already-built (21, 3) array) -> (21, 3):
    wrist-relative, scaled by the wrist-to-middle-finger-MCP distance, rotated
    so that vector points up. This is block B: handshape with location and
    orientation deliberately discarded."""
    points = landmarks if isinstance(landmarks, np.ndarray) else _points_array(landmarks)
    return _normalize(points, points[WRIST], points[MIDDLE_FINGER_MCP] - points[WRIST], target_angle=-np.pi / 2)


# extract_kaggle_landmarks.py tags every record with "part" directly
# ("right_hand"/"left_hand"/"face"/"pose", straight from Kaggle's own parquet
# column). process_video.py's live hand-only recordings predate that field
# and use "handedness" ("Right"/"Left"/None) instead -- _PART_FROM_HANDEDNESS
# maps those onto the same part names so fuse_landmarks can treat both record
# shapes identically.
_PART_FROM_HANDEDNESS = {RIGHT: "right_hand", LEFT: "left_hand"}
# Which rows of a part's raw landmark array end up in block A. Hands keep all
# 21 points; face keeps only the lips; pose keeps only the upper body.
_KEEP_ROWS = {"face": LIP_INDICES, "pose": POSE_KEEP}
_BODY_SLOT_BY_PART = {
    "right_hand": A_RIGHT_HAND_SLOT,
    "left_hand": A_LEFT_HAND_SLOT,
    "face": A_LIPS_SLOT,
    "pose": A_POSE_SLOT,
}
_HANDSHAPE_SLOT_BY_PART = {"right_hand": B_RIGHT_HAND_SLOT, "left_hand": B_LEFT_HAND_SLOT}


def body_frame(points_by_part: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """One frame's parts -> the single (origin, axis_vector) every part in
    block A gets normalized by, or None if the frame has no usable reference.

    Shoulders first: the shoulder midpoint and shoulder width are the most
    stable body-scale reference MediaPipe gives, and they are what "location"
    in ASL is actually relative to. Falls back to the face's nose tip +
    inter-eye distance when pose wasn't detected, so a frame with a face and
    hands but no pose still produces real coordinates instead of zeros.

    Takes the already-parsed per-part arrays rather than the raw records so
    fuse_landmarks parses each frame's landmarks exactly once -- at 94k videos
    of ~80 frames, parsing 468 face dicts twice per frame is not free.
    """
    pose = points_by_part.get("pose")
    if pose is not None:
        origin = (pose[POSE_LEFT_SHOULDER] + pose[POSE_RIGHT_SHOULDER]) / 2
        return origin, pose[POSE_RIGHT_SHOULDER] - pose[POSE_LEFT_SHOULDER]

    face = points_by_part.get("face")
    if face is not None:
        return face[FACE_NOSE_TIP], face[FACE_LEFT_EYE_OUTER] - face[FACE_RIGHT_EYE_OUTER]

    return None


def fuse_landmarks(frame_records: list[dict]) -> np.ndarray:
    """Combine every body part detected in one frame into one fixed
    COORD_DIM-vector: block A (shared body frame) + block B (per-hand
    canonical handshape).

    Two passes, and the two passes are the whole point. The first resolves one
    reference frame for the entire frame; the second pushes every part through
    that same reference. Normalizing each part into its own private coordinate
    system -- which is what this used to do -- makes "hand near the chin" and
    "hand near the forehead" identical, and those are different signs.

    Block B is filled whenever a hand is present, body frame or not: it only
    needs the hand's own wrist, so a hands-only recording (process_video.py)
    still produces handshape features even though its block A stays zero.

    Slot order is fixed by part, not detection order. A missing part's slot
    stays zero -- and because zero is a legal coordinate after normalization,
    finalize_sequence adds explicit presence flags rather than leaving the
    model to guess which zeros mean "absent".
    """
    fused = np.zeros(COORD_DIM, dtype=np.float32)

    points_by_part = {}
    for r in frame_records:
        part = r.get("part") or _PART_FROM_HANDEDNESS.get(r.get("handedness"))
        if part in _BODY_SLOT_BY_PART and r["landmarks"]:
            points_by_part[part] = _points_array(r["landmarks"])

    for part, points in points_by_part.items():
        if part in _HANDSHAPE_SLOT_BY_PART:
            fused[_HANDSHAPE_SLOT_BY_PART[part]] = normalize_landmarks(points).reshape(-1)

    frame = body_frame(points_by_part)
    if frame is None:
        return fused
    origin, axis_vector = frame

    for part, points in points_by_part.items():
        keep = _KEEP_ROWS.get(part)
        kept = points[keep] if keep else points
        fused[_BODY_SLOT_BY_PART[part]] = _normalize(kept, origin, axis_vector, target_angle=0.0).reshape(-1)

    return fused


def _swap_landmark_pairs(block: np.ndarray, n_paired: int) -> np.ndarray:
    """(seq, k*3) block of landmarks -> same block with landmarks swapped
    within each consecutive (0,1), (2,3), ... pair. Only the first n_paired
    landmarks are paired; any remainder (the lip midline) stays put."""
    points = block.reshape(len(block), -1, 3).copy()
    paired = points[:, :n_paired].reshape(len(block), -1, 2, 3)
    points[:, :n_paired] = paired[:, :, ::-1].reshape(len(block), n_paired, 3)
    return points.reshape(len(block), -1)


def mirror_sequence(coords: np.ndarray) -> np.ndarray:
    """Reflect a whole (seq_len, COORD_DIM) clip left-right: negate x on every
    point, swap the right and left hand blocks, and swap each left/right
    landmark pair within lips and pose.

    Augmentation is not "add noise for regularization" -- it is asserting that
    a transform does not change the answer. ASL is handedness-agnostic, so the
    mirror image of a sign is the same word, which makes this label-preserving
    and roughly doubles the effective corpus. Negating x without also swapping
    the paired landmarks would produce an anatomically impossible body (a left
    hand where a right hand's index still points inward), so both halves of
    this function are load-bearing.
    """
    out = coords.copy()
    for right, left in ((A_RIGHT_HAND_SLOT, A_LEFT_HAND_SLOT), (B_RIGHT_HAND_SLOT, B_LEFT_HAND_SLOT)):
        out[:, right], out[:, left] = coords[:, left], coords[:, right]
    out[:, A_LIPS_SLOT] = _swap_landmark_pairs(out[:, A_LIPS_SLOT], n_paired=2 * len(LIP_PAIRS))
    out[:, A_POSE_SLOT] = _swap_landmark_pairs(out[:, A_POSE_SLOT], n_paired=len(POSE_KEEP))

    points = out.reshape(len(out), COORD_LANDMARKS, 3)
    points[:, :, 0] *= -1
    return points.reshape(len(out), COORD_DIM)


def finalize_sequence(coords: np.ndarray) -> np.ndarray:
    """(seq_len, COORD_DIM) cached coordinates -> (seq_len, TOTAL_FEATURE_DIM)
    model input: append block C (velocity) and block D (presence flags).

    Derived here rather than cached so blocks C and D can change without a
    cache rebuild, and called by both the training datasets and mac.py's live
    path so the two cannot drift apart.

    Velocity: an LSTM *can* learn a first-order difference, but it has to
    spend capacity and data discovering that x_t - x_{t-1} matters, out of a
    noisy input where the hands are only a few percent of the variance.
    Handing it over costs 282 dims and zero learned parameters, and movement
    is a phonemic parameter of ASL -- it deserves to be a first-class input.
    Deltas are zeroed wherever either frame's part was absent, so a hand
    appearing mid-clip produces no spurious 0 -> real jump.

    Presence: zero-filling a missing part is ambiguous, because zero is a
    legal normalized coordinate meaning "at the reference origin". With hands
    absent in roughly 65% of frames that is the common case, not an edge case,
    and when the hands leave frame is itself informative about gesture
    boundaries.
    """
    presence = np.stack([np.abs(coords[:, s]).sum(axis=-1) > 0 for s in PART_SLOTS], axis=-1).astype(np.float32)

    body = coords[:, BODY_SLOT]
    velocity = np.diff(body, axis=0, prepend=body[:1])  # first row pairs with itself -> zero
    both_present = presence * np.vstack([presence[:1], presence[:-1]])
    velocity = velocity * np.repeat(both_present, PART_WIDTHS, axis=1)

    return np.concatenate([coords, velocity, presence], axis=1).astype(np.float32)


def augment_sequence(coords: np.ndarray) -> np.ndarray:
    """Randomly perturb one (seq_len, COORD_DIM) fused sequence: a random
    temporal resample (mimics a signer performing faster, or MediaPipe
    dropping frames), a coin-flip left-right mirror (a left-dominant signer),
    and independent per-landmark noise per frame (MediaPipe's own detection
    jitter).

    Runs on the coordinate blocks only, before finalize_sequence, so velocity
    and presence are derived from the augmented clip instead of describing the
    original one.

    A part's slots are all-zero when fuse_landmarks found no detection for it
    that frame -- that zero means "absent", not "at the origin". Mirroring is
    linear so it leaves zero at zero, but additive noise would turn an absent
    part into a fake noisy one, so noise is masked out wherever a landmark is
    absent.
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

    if np.random.random() < AUGMENT_MIRROR_PROB:
        coords = mirror_sequence(coords)

    coords = coords.reshape(seq_len, COORD_LANDMARKS, 3).copy()  # (seq, landmark, xyz)
    present = np.abs(coords).sum(axis=-1, keepdims=True) > 0  # (seq, landmark, 1)
    noise = np.random.normal(0, AUGMENT_JITTER_STD, coords.shape).astype(np.float32)
    coords += noise * present

    return coords.reshape(seq_len, COORD_DIM)


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
    """One video's records -> one (sequence_length, COORD_DIM) array: group by
    frame, fuse each frame's detections, resample to sequence_length steps.
    None if the video had no frames at all. This is what gets cached to .npy --
    blocks A+B only; finalize_sequence adds C+D at load time."""
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
        x = torch.from_numpy(finalize_sequence(coords))
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
    path whenever a cache's shape doesn't match what this dataset expects --
    BOTH axes are checked, because a stale cache from an older feature layout
    is the wrong width, and a width-blind guard would serve it silently as if
    it were correct. A wrong-but-plausible tensor produces a wrong-but-
    plausible accuracy number, which is a far worse failure than a loud one.
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
            if cached.shape == (self.sequence_length, COORD_DIM):
                coords = cached
        if coords is None:
            coords = _video_to_sequence(load_video(path, label), self.sequence_length)
            if coords is None:
                raise ValueError(f"{path} has no frames -- every written landmarks file should have at least one")
        if self.augment:
            coords = augment_sequence(coords)
        x = torch.from_numpy(finalize_sequence(coords))
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


def _check_lip_pairs_against_corpus(sample_files: int = 300) -> None:
    """Validate LIP_PAIRS empirically instead of trusting a hand-typed list.

    Averaged over many signers and frames the mouth is bilaterally symmetric,
    so mirroring the mean lip shape must land back on the mean lip shape --
    but only if each point is swapped with its true partner. Comparing against
    the same mirror WITHOUT the pair swap is what makes this discriminating: a
    wrong pairing scores no better than not pairing at all.

    Mirrored about the lips' own centroid, not about x=0, and that distinction
    is a measured fact rather than a convenience: the face midline is NOT the
    shoulder midline. Averaged over the corpus the head sits ~0.043 shoulder-
    widths toward the right shoulder. Reflecting about x=0 therefore moves the
    whole mouth by twice that, which swamps the pairing signal this function
    exists to test. (mirror_sequence still reflects about x=0 and is right to:
    the shoulders are at exactly -/+0.5 by construction so they map onto each
    other exactly, and turning a head that leans right into one that leans
    left is what a mirrored signer actually looks like.)

    Skipped when no rebuilt cache is on disk yet (fresh clone, or caches still
    at the old layout), since there is nothing to average.

    ponytail: catches a scrambled or misaligned pair list, not a near-miss
    (swapping 267 for its neighbour 269 moves a point ~1mm). Upgrade path if
    that ever matters: check against MediaPipe's canonical_face_model.obj.
    """
    from ingest import RAW_LANDMARKS_DIR

    caches = []
    for path in sorted(RAW_LANDMARKS_DIR.glob("*_landmarks.npy"))[:sample_files]:
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[1] == COORD_DIM:
            caches.append(arr)
    if not caches:
        print("  lip-pair check SKIPPED: no COORD_DIM caches in data/raw_landmarks "
              "(run build_landmark_cache.py, then re-run this self-check)")
        return

    frames = np.concatenate(caches)[:, A_LIPS_SLOT]
    frames = frames[np.abs(frames).sum(axis=-1) > 0]  # lips actually detected
    assert len(frames) > 0, "no frame in the sample had lips detected"
    mean_lips = frames.mean(axis=0).reshape(LIP_LANDMARKS, 3)
    mean_lips[:, 0] -= mean_lips[:, 0].mean()  # reflect about the face's own midline, not the shoulders'

    # Sharpest single signal, and free of any centering assumption: true
    # mirror partners sit at the same height, whatever x offset the head has.
    paired = mean_lips[:2 * len(LIP_PAIRS)].reshape(len(LIP_PAIRS), 2, 3)
    height_gap = np.abs(paired[:, 0, 1] - paired[:, 1, 1]).max()
    assert height_gap < 0.01, (
        f"LIP_PAIRS look wrong: some pair's two points sit {height_gap:.4f} apart in y, "
        f"so they are not reflections of each other")

    flipped = mean_lips.copy()
    flipped[:, 0] *= -1
    flat = flipped.reshape(1, LIP_FEATURE_DIM)
    swapped_err = np.abs(_swap_landmark_pairs(flat, 2 * len(LIP_PAIRS)).reshape(LIP_LANDMARKS, 3) - mean_lips).mean()
    unswapped_err = np.abs(flipped - mean_lips).mean()
    assert swapped_err < unswapped_err / 3, (
        f"LIP_PAIRS look wrong: mirroring the mean lip shape with the pair swap is off by "
        f"{swapped_err:.4f}, barely better than not swapping at all ({unswapped_err:.4f})")
    print(f"  lip-pair check passed on {len(frames)} real frames: mirror error {swapped_err:.4f} "
          f"with the pair swap vs {unswapped_err:.4f} without, worst pair height gap {height_gap:.4f}")


def _self_check():
    INDEX_FINGER_MCP = 5

    def hand(video, label, handedness, frame, index_x_delta=0.0, x=0.5, y=0.5):
        # wrist=landmark 0 (origin), MCP=landmark 9 fixes scale+rotation for
        # every hand identically -- vary a third landmark (index MCP) so
        # right vs. left hands actually end up geometrically distinguishable
        # after normalization, instead of testing two hands with the same shape.
        landmarks = [{"x": x, "y": y, "z": 0.0} for _ in range(21)]
        landmarks[MIDDLE_FINGER_MCP] = {"x": x, "y": y + 0.1, "z": 0.0}
        landmarks[INDEX_FINGER_MCP] = {"x": x + index_x_delta, "y": y + 0.05, "z": 0.0}
        return {"video": video, "label": label, "handedness": handedness, "frame": frame, "landmarks": landmarks}

    def pose(video, label, frame, x_shift=0.0):
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(33)]
        landmarks[POSE_LEFT_SHOULDER] = {"x": 0.4 + x_shift, "y": 0.3, "z": 0.0}
        landmarks[POSE_RIGHT_SHOULDER] = {"x": 0.6 + x_shift, "y": 0.3, "z": 0.0}
        for i in POSE_KEEP[2:]:  # elbows/wrists/fingers, spread out so they aren't all identical
            landmarks[i] = {"x": 0.35 + 0.02 * i + x_shift, "y": 0.4 + 0.01 * i, "z": 0.0}
        return {"video": video, "label": label, "part": "pose", "frame": frame, "landmarks": landmarks}

    def face(video, label, frame, x_shift=0.0):
        landmarks = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(468)]
        landmarks[FACE_NOSE_TIP] = {"x": 0.5, "y": 0.5, "z": 0.0}
        landmarks[FACE_RIGHT_EYE_OUTER] = {"x": 0.45 + x_shift, "y": 0.4, "z": 0.0}
        landmarks[FACE_LEFT_EYE_OUTER] = {"x": 0.55 + x_shift, "y": 0.4, "z": 0.0}
        for j, i in enumerate(LIP_INDICES):  # lips, spread out so the block isn't uniform
            landmarks[i] = {"x": 0.48 + 0.001 * j + x_shift, "y": 0.56 + 0.0005 * j, "z": 0.0}
        return {"video": video, "label": label, "part": "face", "frame": frame, "landmarks": landmarks}

    def clip(video, label, n_frames, **hand_kw):
        """A full-body clip: right hand + pose + face on every frame."""
        records = []
        for f in range(n_frames):
            records.append(hand(video, label, RIGHT, f, **hand_kw))
            records.append(pose(video, label, f))
            records.append(face(video, label, f))
        return records

    # two separate "book" videos (different signers/takes) -- must stay two
    # separate sequences even though they share a label.
    records = clip("book_a", "book", 5)
    records += clip("book_b", "book", 4, index_x_delta=0.05)
    for f in range(8):  # two-handed sign, each hand a different shape and place
        records.append(hand("go_a", "go", RIGHT, f, index_x_delta=0.02, x=0.6))
        records.append(hand("go_a", "go", LEFT, f, index_x_delta=-0.02, x=0.3))
        records.append(pose("go_a", "go", f))
        records.append(face("go_a", "go", f))

    # a third "book" video with a completely empty middle frame (process_video's
    # placeholder shape: handedness=None, no landmarks) -- must zero-fill that
    # frame, not drop it and shrink the sequence.
    records += clip("book_c", "book", 1)
    records.append({"video": "book_c", "label": "book", "handedness": None, "frame": 1, "landmarks": []})
    records += [r | {"frame": 2} for r in clip("book_c", "book", 1)]

    vocab = load_vocab()
    assert len(vocab) == 250
    assert vocab[28] == "book" and vocab[97] == "go"  # fixed Kaggle asl-signs indices, not alphabetical
    one_hot_go = one_hot("go", vocab)
    assert one_hot_go.sum() == 1.0 and one_hot_go[97] == 1.0

    # layout arithmetic: the documented block sizes have to actually add up.
    assert (HAND_FEATURE_DIM, LIP_FEATURE_DIM, POSE_FEATURE_DIM) == (63, 120, 36)
    assert BODY_SLOT.stop == 282 and COORD_DIM == 408 and TOTAL_FEATURE_DIM == 694
    assert len(LIP_INDICES) == 40 and len(set(LIP_INDICES)) == 40  # no duplicate lip point
    assert sum(PART_WIDTHS) == BODY_SLOT.stop

    # block B (per-hand canonical handshape) is unchanged from the old
    # per-part normalization: wrist at the origin, MCP pointing up.
    norm = normalize_landmarks(records[0]["landmarks"])
    assert norm[WRIST].tolist() == [0.0, 0.0, 0.0]
    mirrored = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    mirrored[MIDDLE_FINGER_MCP] = {"x": 0.6, "y": 0.5, "z": 0.0}
    assert np.allclose(norm[MIDDLE_FINGER_MCP], normalize_landmarks(mirrored)[MIDDLE_FINGER_MCP], atol=1e-5)

    # body_frame: pose wins when present, face is the fallback, neither -> None.
    pose_points = _points_array(pose("p", "book", 0)["landmarks"])
    face_points = _points_array(face("f", "book", 0)["landmarks"])
    origin, axis_vector = body_frame({"pose": pose_points, "face": face_points})
    assert np.allclose(origin, (pose_points[POSE_LEFT_SHOULDER] + pose_points[POSE_RIGHT_SHOULDER]) / 2)
    face_origin, _ = body_frame({"face": face_points})
    assert np.allclose(face_origin, face_points[FACE_NOSE_TIP])  # fell back to the face
    assert body_frame({"right_hand": np.zeros((21, 3), dtype=np.float32)}) is None

    # the shared frame really is shared: after fusing, the two shoulders sit at
    # +/-0.5 on x (shoulder width scaled to 1, shoulder line horizontal) and
    # the hand is expressed relative to that same origin -- not to its own wrist.
    fused = fuse_landmarks([hand("v", "book", RIGHT, 0), pose("v", "book", 0), face("v", "book", 0)])
    assert fused.shape == (COORD_DIM,)
    fused_pose = fused[A_POSE_SLOT].reshape(POSE_LANDMARKS, 3)
    assert np.isclose(np.linalg.norm(fused_pose[1] - fused_pose[0]), 1.0, atol=1e-5)  # shoulders 1 apart
    assert np.allclose((fused_pose[0] + fused_pose[1])[:2] / 2, 0.0, atol=1e-5)       # midpoint at the origin
    assert not np.allclose(fused[A_RIGHT_HAND_SLOT].reshape(21, 3)[WRIST], 0.0)       # hand NOT re-anchored to itself
    assert np.allclose(fused[B_RIGHT_HAND_SLOT].reshape(21, 3)[WRIST], 0.0)           # block B still is

    # location survives: the same handshape at two different heights must now
    # produce different block-A coordinates (this is exactly what per-part
    # normalization used to destroy) while block B stays identical.
    low = fuse_landmarks([hand("v", "book", RIGHT, 0, y=0.5), pose("v", "book", 0), face("v", "book", 0)])
    high = fuse_landmarks([hand("v", "book", RIGHT, 0, y=0.2), pose("v", "book", 0), face("v", "book", 0)])
    assert not np.allclose(low[A_RIGHT_HAND_SLOT], high[A_RIGHT_HAND_SLOT])  # "chin" vs "forehead" distinguishable
    assert np.allclose(low[B_RIGHT_HAND_SLOT], high[B_RIGHT_HAND_SLOT], atol=1e-5)  # same handshape either way

    # camera nuisance is still removed: shifting the whole body sideways must
    # not change anything.
    shifted = fuse_landmarks([hand("v", "book", RIGHT, 0, x=0.6), pose("v", "book", 0, x_shift=0.1),
                               face("v", "book", 0, x_shift=0.1)])
    unshifted = fuse_landmarks([hand("v", "book", RIGHT, 0, x=0.5), pose("v", "book", 0), face("v", "book", 0)])
    assert np.allclose(shifted, unshifted, atol=1e-4)

    # hands-only records (process_video.py's shape) have no body frame, so
    # block A stays zero -- but block B is still filled, since a hand's own
    # wrist is all that needs.
    hands_only = fuse_landmarks([hand("v", "book", RIGHT, 0)])
    assert np.allclose(hands_only[BODY_SLOT], 0.0)
    assert not np.allclose(hands_only[B_RIGHT_HAND_SLOT], 0.0)

    # sequence check: one sequence per video (not per frame-chunk, and not
    # merged with other videos sharing a label).
    dataset = SignSequenceDataset(records, vocab, sequence_length=3)
    assert len(dataset) == 4  # book_a, book_b, book_c, go_a -- all three book videos stay separate

    seqs_by_label = defaultdict(list)
    for coords, label in dataset.sequences:
        seqs_by_label[label].append(coords)

    book_seqs = seqs_by_label["book"]
    assert len(book_seqs) == 3  # none of the three book videos collapsed into another
    assert not np.allclose(book_seqs[0], book_seqs[1])  # different takes, different coords
    for book_seq in book_seqs:
        assert book_seq.shape == (3, COORD_DIM)
        assert not np.allclose(book_seq[0, A_RIGHT_HAND_SLOT], 0.0)  # right hand present
        assert np.allclose(book_seq[0, A_LEFT_HAND_SLOT], 0.0)       # left hand absent -> zero-filled
        assert not np.allclose(book_seq[0, A_LIPS_SLOT], 0.0)        # lips present
        assert not np.allclose(book_seq[0, A_POSE_SLOT], 0.0)        # pose present

    # book_c's middle frame had nothing at all -- it must show up as a zero
    # row, not vanish and pull frame 2 into frame 1's slot.
    book_c_seq = book_seqs[2]
    assert np.allclose(book_c_seq[1], 0.0)
    assert not np.allclose(book_c_seq[0, A_RIGHT_HAND_SLOT], 0.0)
    assert not np.allclose(book_c_seq[2, A_RIGHT_HAND_SLOT], 0.0)

    go_seq = seqs_by_label["go"][0]
    assert not np.allclose(go_seq[0, A_RIGHT_HAND_SLOT], 0.0)
    assert not np.allclose(go_seq[0, A_LEFT_HAND_SLOT], 0.0)
    assert not np.allclose(go_seq[0, A_RIGHT_HAND_SLOT], go_seq[0, A_LEFT_HAND_SLOT])

    # finalize_sequence: blocks C and D land in their documented slots, the
    # first velocity row is zero, an absent part's velocity is zeroed rather
    # than spiking, and presence flags match which slots are actually filled.
    final = finalize_sequence(book_c_seq)
    assert final.shape == (3, TOTAL_FEATURE_DIM)
    assert np.allclose(final[:, :COORD_DIM], book_c_seq)          # coords passed through untouched
    assert np.allclose(final[0, VELOCITY_SLOT], 0.0)              # nothing to difference against
    assert np.allclose(final[1, VELOCITY_SLOT], 0.0)              # part vanished -> no real -> 0 spike
    assert np.allclose(final[2, VELOCITY_SLOT], 0.0)              # part reappeared -> no 0 -> real spike
    assert final[0, PRESENCE_SLOT].tolist() == [1.0, 0.0, 1.0, 1.0]  # right hand, no left hand, lips, pose
    assert final[1, PRESENCE_SLOT].tolist() == [0.0, 0.0, 0.0, 0.0]  # empty frame

    moving = finalize_sequence(np.stack([low, high]))
    assert np.allclose(moving[0, VELOCITY_SLOT], 0.0)
    assert not np.allclose(moving[1, VELOCITY_SLOT], 0.0)  # both frames present -> real delta
    assert np.allclose(moving[1, VELOCITY_SLOT], high[BODY_SLOT] - low[BODY_SLOT], atol=1e-6)

    # mirroring: a mirrored right-handed clip is a left-handed clip with the
    # same handshape, and mirroring twice is the identity.
    right_only = np.stack([fuse_landmarks([hand("v", "book", RIGHT, 0, x=0.65),
                                            pose("v", "book", 0), face("v", "book", 0)])])
    flipped = mirror_sequence(right_only)
    assert np.allclose(flipped[0, A_RIGHT_HAND_SLOT], 0.0)         # right hand became the left one
    assert not np.allclose(flipped[0, A_LEFT_HAND_SLOT], 0.0)
    assert np.allclose(mirror_sequence(flipped), right_only, atol=1e-6)  # involution
    flipped_pose = flipped[0, A_POSE_SLOT].reshape(POSE_LANDMARKS, 3)
    orig_pose = right_only[0, A_POSE_SLOT].reshape(POSE_LANDMARKS, 3)
    assert np.allclose(flipped_pose[0, 0], -orig_pose[1, 0], atol=1e-6)  # shoulders swapped AND x-negated
    assert np.allclose(flipped[0, A_LIPS_SLOT].reshape(LIP_LANDMARKS, 3)[36:, 1],
                        right_only[0, A_LIPS_SLOT].reshape(LIP_LANDMARKS, 3)[36:, 1], atol=1e-6)  # midline stays

    _check_lip_pairs_against_corpus()

    loader = make_dataloader(records, vocab, sequence_length=3, batch_size=2, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == (2, 3, TOTAL_FEATURE_DIM)
    assert y.shape[1] == len(vocab)

    # augmentation: a present part should move, an absent part must stay
    # exactly zero -- not get hallucinated noise -- and the un-augmented
    # dataset must be untouched (val/test path).
    np.random.seed(0)
    augmented = augment_sequence(book_seqs[0])
    assert augmented.shape == book_seqs[0].shape
    assert not np.allclose(augmented[0, A_RIGHT_HAND_SLOT], book_seqs[0][0, A_RIGHT_HAND_SLOT])
    assert np.allclose(augmented[0, A_LEFT_HAND_SLOT], 0.0) or np.allclose(augmented[0, A_RIGHT_HAND_SLOT], 0.0)
    # exactly one hand was present, so exactly one of the two hand slots is
    # zero after augmentation whether or not the mirror coin came up heads.

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
        book_c_records = clip("book_c", "book", 1)
        book_c_records.append({"video": "book_c", "label": "book", "handedness": None, "frame": 1, "landmarks": []})
        book_c_records += [r | {"frame": 2} for r in clip("book_c", "book", 1)]
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
                assert torch.allclose(lazy_x[1], torch.zeros(TOTAL_FEATURE_DIM))  # empty frame -> zero-filled

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
        assert torch.allclose(cached_x, torch.from_numpy(finalize_sequence(book_coords)))

        # A cache with the wrong shape on EITHER axis must be ignored, not
        # served: a different sequence_length (axis 0) or a stale feature
        # layout (axis 1) both fall back to recomputing from JSON.
        hello_path, _ = files[1]
        assert hello_path.name == "hello_landmarks.json"
        for stale in [(99, COORD_DIM), (3, 1629)]:
            np.save(cache_path_for(hello_path), np.zeros(stale, dtype=np.float32))
            guarded_x, _ = cached_dataset[1]
            assert guarded_x.shape == (3, TOTAL_FEATURE_DIM), f"stale {stale} cache was served instead of recomputed"

        lazy_loader = make_lazy_dataloader(files, vocab, sequence_length=3, batch_size=2, shuffle=False,
                                            num_workers=0)
        lazy_batch_x, lazy_batch_y = next(iter(lazy_loader))
        assert lazy_batch_x.shape[1:] == (3, TOTAL_FEATURE_DIM)
        assert lazy_batch_y.shape[1] == len(vocab)

    print("self-check passed: one shared body frame per frame (pose shoulders, face fallback) "
          "keeping location/orientation in block A while block B keeps canonical handshape, "
          "lips-only face (40 of 468 points), derived velocity + presence blocks, "
          "label-preserving mirror augmentation, per-video sequences, the lazy path matching "
          "the eager path exactly, and its .npy cache guarded on BOTH axes so a stale layout "
          "gets recomputed instead of silently served")


if __name__ == "__main__":
    _self_check()
