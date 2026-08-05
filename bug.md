# Live camera inference bugs (src/mac.py)

Observed on the Mac after pulling the trained checkpoint: video and camera
wiring work, but the sign predictions are unusable. Three symptoms, all
traced back to `src/mac.py`'s live inference loop.

## 1. Says more than one word at once, rapidly

**Symptom:** words overlap audibly and change many times per second.

**Cause:** `run_live` (src/mac.py:68-77) runs a forward pass and a
speak-if-changed check on *every camera frame*, with no cooldown or
debounce. Every time `predicted_word` differs from `last_spoken`, it spawns
a new daemon thread that calls `speak()` (src/mac.py:76):

```python
if predicted_word != last_spoken:
    threading.Thread(target=speak, args=(predicted_word,), daemon=True).start()
    last_spoken = predicted_word
```

Because the model's output flips frequently frame-to-frame (see bug 2),
`predicted_word` changes rapidly, so many threads get spawned in quick
succession. `speak()` blocks only the thread it runs on, so nothing stops
several of these threads' `say` calls from playing at the same time --
that's the overlapping audio. There is no minimum hold time before a
prediction is accepted, and no lock/queue serializing speech.

## 2. Predicts the wrong word

**Symptom:** the spoken word rarely matches the sign actually performed.

**Cause:** a mismatch between what the model was trained on and what it's
fed live.

Training (`src/dataset.py:_build_sequences`, line 128) takes each *whole,
pre-segmented, single-sign video clip* (14-384 raw frames covering one
complete gesture start to finish) and resamples it with `np.linspace` down
or up to exactly `SEQUENCE_LENGTH` (60) evenly-spaced steps spanning that
entire performance. Every training example the model ever saw is "one
complete sign, normalized to 60 steps."

Live inference (`src/mac.py:51,66-69`) instead keeps a raw sliding window
of the last 60 *camera frames* with no resampling and no segmentation:

```python
window = deque(maxlen=SEQUENCE_LENGTH)
...
window.append(fuse_hands(frame_to_hand_records(result)))
if len(window) == SEQUENCE_LENGTH:
    x = torch.from_numpy(np.stack(window)).unsqueeze(0).to(device)
```

At typical webcam framerates that's roughly 2 seconds of continuous,
unsegmented video -- it can contain the tail of one sign, idle movement,
the start of the next sign, or a fragment of a slower sign that never gets
"normalized to one full gesture" the way training data was. The model
never saw inputs shaped like this during training, so on live video it's
effectively evaluating out-of-distribution input and guessing.

## 3. Speaks a word when no hands are on screen

**Symptom:** predictions and speech continue even with empty frames.

**Cause:** `fuse_hands` (src/dataset.py:80-86) zero-fills a hand's 63
slots when no hand is detected in a frame -- correct and intentional for
training, where "absent" is a valid signal. But `SignLSTM` (src/model.py)
is a plain forced-choice classifier over the 630-word vocab: it has no
"no sign" / background class and produces a full softmax over real words
no matter what it's given (model.py:44-56, confirmed by the self-check
at model.py:76-77, `probs.sum(dim=-1) == 1` always). There's no
confidence threshold in `model.py`.

`mac.py` takes the argmax unconditionally (src/mac.py:72) and speaks it
whenever it changes, with no check on whether any hand was actually
present in the window and no confidence gate to suppress low-confidence
guesses:

```python
predicted_word = vocab[logits[0].argmax(dim=-1).item()]
if predicted_word != last_spoken:
    ...
```

A window with no hands at all just gets classified into whichever word's
statistics happen to be closest to an all-zero input -- there's nothing
in the model or the loop that can say "nothing."

## Common thread

All three symptoms come from the same gap: `mac.py` was written to prove
the pipeline is *wired* correctly (camera -> landmarks -> fused vector ->
window -> model -> speech), which it is -- see its own self-check. It was
never given the additional logic continuous, real-time use needs on top
of that: segmenting a continuous stream into "one sign's worth" of frames,
requiring a stable/confident prediction before accepting it, gating
predictions on hand presence, and serializing speech output. None of that
exists yet in this codebase.
