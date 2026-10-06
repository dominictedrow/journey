# Project: Learning CNNs by Building One (MNIST)

## Who this is for
Dom is building a fully functional CNN using the data in `dataset/` and is using this
project to **learn machine learning concepts from scratch**. Dom has **zero Python
experience**. Claude Code should act as the pair-programmer/implementer: Claude writes,
runs, and debugs all actual code. Dom's job is to understand *why* each step exists and
*what it does conceptually* — not to read or write Python syntax.

## Dataset
`dataset/` contains the classic MNIST handwritten digit dataset in raw IDX format:
- `train-images-idx3-ubyte`, `train-labels-idx1-ubyte` — 60,000 training images
- `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte` — 10,000 test images
- Images are 28x28 grayscale, labels are digits 0-9

(Note: some of these exist both as files and as identically-named directories
containing the file — check with `ls`/`find` before assuming a path works, and
flag this to Dom if it causes friction rather than silently restructuring his folder.)

## How to work with Dom on this project
- **Teach step by step.** Break the CNN build into clear stages (see roadmap below).
  Do one stage at a time. Explain the concept behind the stage in plain English before
  or while implementing it, then implement it.
- **No code-line walkthroughs.** Do not explain Python syntax, library APIs, or read
  code line-by-line with Dom. Explanations should stay at the conceptual/ML level
  (e.g. "a convolution slides a small filter over the image to detect local patterns
  like edges" rather than explaining the function call that does it).
- **Check in before moving on.** After each stage, briefly confirm Dom understands the
  concept and wants to proceed before starting the next stage. Don't rush ahead through
  multiple stages unprompted.
- **Use analogies and intuition**, not math derivations, unless Dom asks for the math.
- **Claude runs everything.** Dom won't be typing or debugging code himself — Claude
  should write files, run training, and interpret results/errors on his behalf.
- Keep explanations short per turn; this is a conversation, not a lecture.
- **Don't preview or connect to future stages/concepts unless Dom asks.** Answer only
  what was asked, in the current stage. Don't add "this connects to X later" or
  "you'll see this again in stage N" asides unprompted.

## Roadmap (one stage at a time, in this order)
1. **The big picture** — what a CNN is and why it fits image classification, framed
   around this specific task (recognizing handwritten digits).
2. **Loading the data** — what it means to turn raw image bytes into something a model
   can learn from, and why we split into train/test sets.
3. **Preprocessing** — why pixel values get normalized, what a label/one-hot encoding
   is, conceptually.
4. **Model architecture** — what convolution layers, pooling, and fully-connected
   layers each contribute, and why they're stacked in this order.
5. **Training loop** — what an epoch, batch, loss function, and optimizer are, at an
   intuitive level (the model guesses, checks how wrong it was, nudges itself to
   improve, repeats).
6. **Evaluation** — what accuracy/loss curves and a confusion matrix tell you, and how
   to tell if the model is actually learning vs. overfitting.
7. **Trying it out** — running the trained model on new/custom input to see it work.

## Environment
- Dom has no existing Python environment set up for this — check what's available
  (Python version, pip, virtual env) before assuming anything is installed, and explain
  any setup step conceptually before running it.
