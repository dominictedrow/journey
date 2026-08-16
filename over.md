# Closing the val-accuracy gap: 6 representation + architecture fixes

## Context

`SignLSTM` plateaus at **train 85.2% / val 47.2%** on 250-word Kaggle asl-signs
(94,477 clips, signer-disjoint split 54,669 / 22,149 / 17,659). Dom has already
added layers, activations, more data, face landmarks, and every regularizer he
could think of; the 38-point gap did not move. Diagnosis (measured on the real
`.npy` caches, not guessed):

| finding | measurement |
|---|---|
| Face mesh is 86% of input dims, 47% of input variance | `FACE_SLOT` = 1404 of 1629 dims |
| Face encodes **signer**, not sign | cos-sim gap: same-signer **+0.0178** vs same-word **+0.0022** (8x) |
| Hands carry 5% of variance and are absent most frames | RH present 36.3%, LH 31.8% of frames |
| Per-part normalization destroys orientation + location | `dataset.py:108` re-anchors every hand to its own wrist, every frame |
| Two of three augmentations are no-ops | rotation/scale applied *after* canonicalization removed rotation/scale |

Root cause: **the input representation, not the LSTM.** A 6.2M-param model was
handed 1404 dimensions of signer fingerprint and 126 dimensions of
handshape-only hand data with location and orientation stripped out. It did the
rational thing and memorized signers.

Intended outcome: rebuild the feature representation so the model sees what
actually defines a sign, then fix the architecture shortcut that lets it bypass
temporal structure. Target: val 65-75% (GISLR-competition range for non-winning
but sound solutions). **These are estimates, not promises, and they do not
simply add.**

## Decisions made

- **I implement, one phase at a time.** Dom reviews and retrains between phases.
- **Short probe per phase** (`--epochs 15`), one full run after all 6 land.
  Baseline to beat, from `logs/train_20260812_204017.log` epoch 15:
  **train 66.32% / val 41.62%.**
- **Keep z.** Dropping it is a one-line A/B for later.

---

## Target feature layout (v2)

694 dims, replacing today's 1629. Face-identity share drops from 86% to 17%,
and those 17% are functional (mouth morphemes are real ASL grammar).

| block | contents | dims |
|---|---|---|
| **A — body frame** | RH 21, LH 21, lips 40, pose 12 pts, all in one shared reference frame | 282 |
| **B — canonical handshape** | RH 21 + LH 21, per-hand normalized (today's `normalize_landmarks`) | 126 |
| **C — velocity** | frame-to-frame delta of block A | 282 |
| **D — presence mask** | 4 flags: right_hand, left_hand, lips, pose | 4 |

**Cached to `.npy`: blocks A+B only (`COORD_DIM = 408`).** C and D are derived at
load time, so phases 3-6 need no cache rebuild — only phase 2 does.

New constants in `src/dataset.py`:

```python
LIP_PAIRS = [(37,267),(39,269),(40,270),(185,409),(61,291),      # outer upper
             (84,314),(181,405),(91,321),(146,375),               # outer lower
             (82,312),(81,311),(80,310),(191,415),(78,308),       # inner upper
             (87,317),(178,402),(88,318),(95,324)]                # inner lower
LIP_MIDLINE = [0, 17, 13, 14]
LIP_INDICES = [i for p in LIP_PAIRS for i in p] + LIP_MIDLINE     # 40 points
POSE_KEEP   = list(range(11, 23))   # shoulders/elbows/wrists/fingers; drops legs
```

`LIP_PAIRS` doubles as the mirror map for phase 5. These indices come from the
standard FaceMesh lip contours — **phase 5's self-check validates them
empirically** rather than trusting my list (see below).

---

## Phase 0 — Guard the cache before touching the layout

`dataset.py:360` checks `cached.shape[0] == self.sequence_length` but **not the
feature width**. Every later phase changes that width, so without this fix a
stale 1629-dim cache would be served silently as if it were correct.

- Add `and cached.shape[1] == COORD_DIM` to the guard.
- `build_landmark_cache.py:79` hardcodes `assert cached.shape == (3, 1629)` —
  import the constant instead of restating it.

**Concept — silent data corruption vs loud failure.** A shape guard that checks
one axis is worse than no guard: it looks like safety. Every cache invariant
should fail loudly, because a wrong-but-plausible tensor produces a
wrong-but-plausible accuracy number you will chase for days.

---

## Phase 1 — Drop the face mesh, keep 40 lip points

**Change.** Reduce the face contribution from 468 points to the 40 `LIP_INDICES`.
This phase runs as a **temporary slice against the existing 1629-dim cache** —
no rebuild — so the highest-value question gets answered in one 18-minute probe.
The slice helper is deleted in phase 2 when the real extraction replaces it.

- Add `_reduce_v1_cache(arr)` in `dataset.py`: keeps hand + pose slots, indexes
  `LIP_INDICES` out of `FACE_SLOT`. 1629 -> 345.
- `LazySignVideoDataset.__getitem__` calls it; `INPUT_SIZE` follows.

**Concept — spurious correlation and the signer-disjoint split.** A model
minimizes loss by any available route. Face geometry perfectly predicts the
signer, the signer partly predicts which words appear in their clips, so face
geometry becomes a *shortcut feature*: high training signal, zero transfer to an
unseen signer. This is why the gap opened by epoch 8 and why no amount of
dropout closed it — regularization limits *how much* a model fits, not *what* it
fits. The fix is removing the shortcut from the input, not penalizing its use.

**Why lips stay.** Non-manual markers (mouth morphemes, "mm", "cha", "th") are
one of the five phonemic parameters of ASL and genuinely disambiguate signs. 40
points is enough to carry mouth shape and far too few to carry face identity.

**Expected:** largest single gain, and the train/val gap should visibly narrow.

---

## Phase 2 — One shared reference frame (the only cache rebuild)

**Change.** Stop normalizing each part into its own private coordinate system.

- New `body_frame(frame_records)` returns one `(origin, axis_vector)` per frame:
  pose shoulder midpoint + shoulder width if pose is present, falling back to
  face nose-tip + inter-eye distance, else `None` (frame stays zero).
- `fuse_landmarks` becomes two-pass: resolve the frame reference, then push
  **every** part through the existing `_normalize(points, origin, axis, 0.0)`
  helper with those same shared parameters -> block A.
- Block B additionally keeps today's per-hand `normalize_landmarks` output, so
  scale/rotation-invariant handshape is still handed over directly.
- Delete `_reduce_v1_cache`, `normalize_face`, `normalize_pose` and the
  now-unused face/pose anchor constants.

**Cache migration — required, in this order:**
```powershell
Remove-Item data/raw_landmarks/*.npy      # build_landmark_cache SKIPS existing caches
python src/build_landmark_cache.py
```
The `*_landmarks.json` corpus is untouched — `extract_kaggle_landmarks.py` still
stores all 468 face points and the subset happens at fuse time, so **no
re-extraction from parquet is needed.**

**Concept — invariance is a trade, not a free win.** Normalization removes
nuisance variation, but every invariance you impose is information you can never
recover. A sign is defined by five parameters: handshape, **location**,
**orientation**, movement, and non-manual markers. Anchoring each hand to its own
wrist and rotating it upright every frame removes exactly two of them —
"mother" and "father" (identical handshape, chin vs forehead) become
indistinguishable, and a rotating hand looks static. Location and movement
currently survive only through two coarse BlazePose wrist points.

Choosing one shared anchor keeps *relationships between parts* — where the hand
sits relative to the body, which way it faces, how it travels — while still
removing what genuinely is nuisance: where the signer sits in frame, how far
they are from the camera, camera tilt. Keeping block B alongside block A is the
deliberate best-of-both: absolute geometry *and* a canonicalized handshape view.

**Per-frame, not per-clip**, so `fuse_landmarks(frame_records) -> vector` keeps
its contract and `mac.py`'s live path needs no restructuring.

---

## Phase 3 — Velocity features

**Change.** In a new shared `finalize_sequence(coords)` (called by both
`_video_to_sequence` and `mac.py`, so training and live inference cannot
diverge): `np.diff` block A along time, prepend a zero row, append as block C.
Zero the velocity wherever either frame's part is absent, so a hand appearing
mid-clip doesn't produce a spurious `0 -> real` spike.

**Concept — feature engineering as a prior.** An LSTM *can* learn a first-order
difference; it has the memory to do it. But it has to spend capacity and data
discovering that `x_t - x_{t-1}` matters, from a 5%-variance corner of a noisy
input. Handing it over explicitly costs 282 dims and zero learned parameters.
This is the standard trade: an architecture that *could* learn something is not
the same as one that *will*, given finite data. Movement is a phonemic parameter
of ASL — make it a first-class input.

---

## Phase 4 — Presence mask

**Change.** In `finalize_sequence`, 4 float flags per frame from
`|block|.sum(-1) > 0` per part -> block D.

**Concept — zero-imputation ambiguity.** `fuse_landmarks` zero-fills a missing
part, but zero is a *legal coordinate*: after normalization it means "at the
reference origin." The network cannot tell "no hand detected" from "hand exactly
at the shoulder midpoint." With hands absent in ~65% of frames, that is not an
edge case — it is the majority of the input. An explicit missingness indicator
is the standard fix, and it converts a corrupting ambiguity into usable signal
(when the hands leave the frame is itself informative about gesture boundaries).

---

## Phase 5 — Mirror augmentation; delete the dead augmentations

**Change** in `augment_sequence`:

- **Add** `mirror_sequence(coords)`, applied with p=0.5: negate x on every point;
  swap RH<->LH in blocks A and B; swap pose left/right index pairs
  (11<->12, 13<->14, ... 21<->22); swap `LIP_PAIRS`, leaving `LIP_MIDLINE` fixed.
- **Delete** `AUGMENT_ROTATION_DEG` and `AUGMENT_SCALE_RANGE` and their code.
- **Keep** jitter and temporal resample.

**Self-check that validates the index maps** (not just the code path): take the
dataset mean lip shape in body frame, mirror it, assert it matches the original
within tolerance. Same for the pose pairs. If my hardcoded `LIP_PAIRS` are
wrong, this fails — that assert is what makes the list trustworthy.

**Concept — augmentation encodes a symmetry you know the label has.** Data
augmentation is not "add noise for regularization"; it is asserting *this
transform does not change the answer.* ASL is handedness-agnostic — a
left-dominant signer produces the mirror image of the same sign, and it is the
same word. That makes mirroring label-preserving, which effectively doubles the
corpus for free and forces the model to learn handedness-independent features.

**Why the other two get deleted.** An augmentation is worthless if the
normalizer undoes it. Body-frame normalization pins shoulder width to 1 and the
shoulder line to horizontal, so a global rotation or rescale produces an input
the pipeline will *never* generate at inference — the model spends capacity
learning invariance to an impossible variation. This is the general rule worth
keeping: **augment in the space the model actually sees, and only with
variations that survive your preprocessing.**

---

## Phase 6 — Architecture

**Change** in `src/model.py`:

| today | new | why |
|---|---|---|
| `residual_proj` re-adds raw input before all 3 ReLUs | single `Linear(694, 384)` -> `LayerNorm` -> ReLU input projection | kills the bag-of-frames shortcut |
| — | residuals between LSTM layers (hidden->hidden) | standard depth residual, no raw re-injection |
| 3x unidirectional `nn.LSTM(384)` | 3x `bidirectional=True`, 192/direction | both ends of a gesture inform its meaning |
| `BatchNorm1d` over flattened timesteps | `LayerNorm` | per-timestep, batch-independent; correct for sequences |
| `mean(dim=1)` pooling | attention pooling: `softmax(Linear(384,1))` weights | model picks the informative frames |

**Concept — the shortcut path.** `mean_t(ReLU(W·x_t + LSTM_out))` with `W·x_t`
re-added at every layer is, at the readout, an **order-invariant MLP over raw
coordinates**. Mean-pooling makes it permutation-invariant in time; the raw
re-injection makes it dominated by the highest-variance input block (the face).
That path fits the training set faster than the LSTM path can, so gradient
descent preferred it — a textbook case of an architecture offering a shortcut
that defeats its own inductive bias. Residuals belong *between layers of the
same width* to ease gradient flow, not as a raw-input bypass around the very
recurrence you built the model for.

**Bidirectional is legal here** and worth calling out: it breaks causal
streaming, but `mac.py` already buffers a complete gesture (hands present ->
absent) before classifying, so no live capability is lost.

**Retune dropout after phase 1.** The current `DROPOUT = 0.5` was compensating
for the face leak. Once that's gone, 0.3 is likely better — treat it as a knob to
sweep, not a mandated change.

---

## Files touched

| file | change |
|---|---|
| `src/dataset.py` | all of it — constants, `body_frame`, `fuse_landmarks`, `finalize_sequence`, `mirror_sequence`, `augment_sequence`, cache guard |
| `src/model.py` | phase 6 architecture; slot imports follow the new layout |
| `src/build_landmark_cache.py` | hardcoded `1629` in `_self_check` -> import `COORD_DIM` |
| `src/mac.py` | call `finalize_sequence` after `resample_sequence`; update the `TOTAL_FEATURE_DIM` assertion |
| `src/train.py`, `src/infer.py`, `src/split_dataset.py`, `src/extract_kaggle_landmarks.py` | **no change** |

`models/checkpoint.pt` becomes unloadable at phase 1 (input dim changes) and is
overwritten on the first retrain. It is tracked in git — worth a commit of the
current one before starting if Dom wants it recoverable.

## Verification

Each phase, before retraining — every one of these already exists:

```powershell
python src/dataset.py                      # self-check
python src/model.py                        # self-check
python src/train.py --self-test
python src/mac.py --self-test              # live path still fuses to the right width
python src/build_landmark_cache.py --self-test
```

Then the probe, and log the result into this table:

```powershell
python src/train.py --epochs 15
```

| phase | train@15 | val@15 | gap | Δ val |
|---|---|---|---|---|
| baseline | 66.32% | 41.62% | 24.7 | — |
| 1 lips only | | | | |
| 2 body frame | | | | |
| 3 velocity | | | | |
| 4 presence mask | | | | |
| 5 mirror | | | | |
| 6 architecture | | | | |

Final: `python src/train.py --epochs 50` (full run), then
`python src/infer.py --test-dir data/raw_landmarks --checkpoint models/checkpoint.pt`
for the held-out test number — **test has never been touched during any of this**,
so it is the only honest final figure.

Watch the **gap** column as much as the val column: phases 1-2 should shrink it
(less memorization available), phases 3-6 should raise both.

## Phase 7 — Artifact

Publish the six fixes and their ML concepts as an Artifact, using the "Concept"
sections above as source, with the filled-in results table as the payoff.

## Explicitly out of scope

Label smoothing, cosine LR schedule with warmup (`lr_scaling=0.99` is currently
near-constant), larger batch size, dropping z, limb-proportion augmentation, and
the Conv1D+Transformer pivot that won GISLR. All are real wins; none are among
the 6, and folding them in would destroy per-fix attribution. Revisit after the
table is full.
