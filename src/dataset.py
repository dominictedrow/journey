"""Turn ingested MediaPipe records (from ingest.load_records) into batched
tensors ready for an LSTM. Each sequence is one whole video (one sign
performance), not an arbitrary chunk of frames.

Coordinate normalization: MediaPipe's `landmarks` are already image-relative
(0-1), but that's relative to the camera frame, not the hand -- the same sign
performed in a different spot in frame, closer/further from the camera, or
tilted at a different angle, produces different raw numbers. Three geometric
steps fix that, covering spatial encoding + scaling + normalization + rotation:
  1. Subtract the wrist (landmark 0) from every point -> position in the
     frame no longer matters (translation invariance).
  2. Divide by the wrist-to-middle-finger-MCP (landmark 9) distance -> hand
     size / distance from camera no longer matters, and the numbers land in
     a small, consistent range (this *is* the scaling + normalization).
  3. Rotate so the wrist-to-middle-finger-MCP vector always points the same
     direction -> in-image-plane tilt of the hand no longer matters.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

VOCAB_PATH = Path(__file__).parent / "vocab.txt"
WRIST = 0
MIDDLE_FINGER_MCP = 9
RIGHT, LEFT = "Right", "Left"
HAND_FEATURE_DIM = 21 * 3
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


def normalize_landmarks(landmarks: list[dict]) -> np.ndarray:
    """21 {x,y,z} dicts -> (21, 3) array: wrist-relative, scale-normalized, rotation-corrected."""
    points = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=np.float32)
    points -= points[WRIST]
    scale = max(float(np.linalg.norm(points[MIDDLE_FINGER_MCP])), 1e-6)
    points /= scale

    # ponytail: only corrects in-plane (x/y) rotation, not full 3D orientation.
    # A second reference axis would be needed to also remove roll around the
    # wrist->middle-finger axis, and MediaPipe's z is a rougher estimate than
    # x/y anyway, so a full 3D fix wouldn't be very trustworthy here.
    angle = np.arctan2(points[MIDDLE_FINGER_MCP, 1], points[MIDDLE_FINGER_MCP, 0])
    target_angle = -np.pi / 2  # hand always ends up pointing "up" in image space
    theta = target_angle - angle
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rotation.T

    return points


def fuse_hands(hand_records: list[dict]) -> np.ndarray:
    """Combine the hand(s) detected in one frame into one fixed 126-vector.

    Each hand is normalized independently (its own wrist, its own scale, its
    own rotation -- one hand's geometry has nothing to do with the other's).
    Slot order is then fixed by handedness, not by detection order: right
    hand's 63 numbers always go in slots 0-62, left hand's in 63-125. A
    missing hand's slot stays zero. Fixed slots mean "which side" is already
    fully encoded by position -- no separate side flag is needed.

    A record with no matching handedness (e.g. process_video.py's placeholder
    for a frame where neither hand was detected, handedness=None) falls
    through both branches untouched, leaving that slot -- or the whole
    vector, if it's the only record for that frame -- zero. normalize_landmarks
    is only called once a branch is known to apply, so a placeholder's empty
    landmarks list is never touched.
    """
    fused = np.zeros(2 * HAND_FEATURE_DIM, dtype=np.float32)
    for r in hand_records:
        if r["handedness"] == RIGHT:
            fused[:HAND_FEATURE_DIM] = normalize_landmarks(r["landmarks"]).reshape(-1)
        elif r["handedness"] == LEFT:
            fused[HAND_FEATURE_DIM:] = normalize_landmarks(r["landmarks"]).reshape(-1)
    return fused


def augment_sequence(coords: np.ndarray) -> np.ndarray:
    """Randomly perturb one (seq_len, 126) fused sequence: one shared random
    rotation + scale for the whole clip (mimics a different camera angle/
    distance), independent per-point noise per frame (mimics MediaPipe's own
    detection jitter), and a random temporal resample (mimics a signer
    performing faster, or MediaPipe dropping frames).

    A hand's 63 slots are all-zero when fuse_hands found no hand that frame --
    that zero means "absent", not "at the origin". Rotation/scale are linear
    so they leave zero at zero, but additive noise would turn an absent hand
    into a fake noisy one, so noise is masked out wherever a hand is absent.
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

    coords = coords.reshape(coords.shape[0], 2, 21, 3).copy()  # (seq, hand, landmark, xyz)
    present = np.abs(coords).sum(axis=(2, 3), keepdims=True) > 0  # (seq, hand, 1, 1)

    angle = np.radians(np.random.uniform(-AUGMENT_ROTATION_DEG, AUGMENT_ROTATION_DEG))
    cos_t, sin_t = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    coords[..., :2] = coords[..., :2] @ rotation.T

    scale = np.random.uniform(1 - AUGMENT_SCALE_RANGE, 1 + AUGMENT_SCALE_RANGE)
    coords *= scale

    noise = np.random.normal(0, AUGMENT_JITTER_STD, coords.shape).astype(np.float32)
    coords += noise * present

    return coords.reshape(coords.shape[0], 126)


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
        frames = defaultdict(list)
        for r in video_records:
            frames[r["frame"]].append(r)

        coords = [fuse_hands(frames[frame_idx]) for frame_idx in sorted(frames)]
        if not coords:
            continue

        sequences.append((resample_sequence(coords, sequence_length), label))

    return sequences


class SignSequenceDataset(Dataset):
    """One item = one video, resampled to (sequence_length, 126) + one-hot label."""

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
    assert len(vocab) == 630
    assert vocab[0] == "book" and vocab[4] == "go"  # fixed WLASL indices, not alphabetical
    one_hot_go = one_hot("go", vocab)
    assert one_hot_go.sum() == 1.0 and one_hot_go[4] == 1.0

    norm = normalize_landmarks(records[0]["landmarks"])
    assert norm[WRIST].tolist() == [0.0, 0.0, 0.0]

    # rotation check: MCP starting out pointing right instead of down should
    # still land on the same canonical position after normalization.
    mirrored = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(21)]
    mirrored[MIDDLE_FINGER_MCP] = {"x": 0.6, "y": 0.5, "z": 0.0}
    norm_mirrored = normalize_landmarks(mirrored)
    assert np.allclose(norm[MIDDLE_FINGER_MCP], norm_mirrored[MIDDLE_FINGER_MCP], atol=1e-5)

    # sequence check: one sequence per video (not per frame-chunk, and not
    # merged with other videos sharing a label), fused to a fixed 126-wide
    # vector (right in 0-62, left in 63-125).
    dataset = SignSequenceDataset(records, vocab, sequence_length=3)
    assert len(dataset) == 4  # book_a, book_b, book_c, go_a -- all three book videos stay separate

    seqs_by_label = defaultdict(list)
    for coords, label in dataset.sequences:
        seqs_by_label[label].append(coords)

    book_seqs = seqs_by_label["book"]
    assert len(book_seqs) == 3  # none of the three book videos collapsed into another
    assert not np.allclose(book_seqs[0], book_seqs[1])  # different takes, different coords
    for book_seq in book_seqs:
        assert book_seq.shape == (3, 126)
        assert not np.allclose(book_seq[0, :63], 0.0)   # right hand present
        assert np.allclose(book_seq[0, 63:], 0.0)        # left hand absent -> zero-filled

    # book_c's middle frame had no hand at all (process_video's None-handedness
    # placeholder) -- it must show up as a zero row, not vanish and pull frame 2
    # into frame 1's slot.
    book_c_seq = book_seqs[2]
    assert np.allclose(book_c_seq[1], 0.0)                  # handless frame -> all zero
    assert not np.allclose(book_c_seq[0, :63], 0.0)         # frame 0's hand still there
    assert not np.allclose(book_c_seq[2, :63], 0.0)         # frame 2's hand still there

    go_seq = seqs_by_label["go"][0]
    assert not np.allclose(go_seq[0, :63], 0.0)   # right hand present
    assert not np.allclose(go_seq[0, 63:], 0.0)   # left hand present too
    assert not np.allclose(go_seq[0, :63], go_seq[0, 63:])  # not the same values

    loader = make_dataloader(records, vocab, sequence_length=3, batch_size=2, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == (2, 3, 126)
    assert y.shape[1] == len(vocab)

    # augmentation: a present hand should move (rotation/scale/noise all
    # apply), an absent hand must stay exactly zero -- not get hallucinated
    # noise -- and the un-augmented dataset must be untouched (val/test path).
    np.random.seed(0)
    augmented = augment_sequence(book_seqs[0])
    assert not np.allclose(augmented[0, :63], book_seqs[0][0, :63])   # right hand: perturbed
    assert np.allclose(augmented[0, 63:], 0.0)                        # left hand: still absent -> still zero

    plain_dataset = SignSequenceDataset(records, vocab, sequence_length=3, augment=False)
    aug_dataset = SignSequenceDataset(records, vocab, sequence_length=3, augment=True)
    plain_x, _ = plain_dataset[0]
    aug_x, _ = aug_dataset[0]
    assert not torch.allclose(plain_x, aug_x)  # augment=True actually perturbs __getitem__ output

    print("self-check passed: per-hand normalization, fixed right/left slot fusion, per-video sequences, "
          "and train-only augmentation (perturbs present hands, leaves absent hands zero) all line up")


if __name__ == "__main__":
    _self_check()
