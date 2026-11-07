# manual-models

> **Part of [journey](https://github.com/dominictedrow/journey)**, Dom's ML and software portfolio.
>
> **How it was built: ✍️ manual.** I wrote the code in this branch myself. Any exception is called out next to the project it applies to.

This branch is where I learn by building things by hand. The vibe-coded projects ([`vibe-projects`](https://github.com/dominictedrow/journey/tree/vibe-projects)) show what I can ship with AI agents. This branch shows what I understand well enough to write myself.

## Where it's going

The [`roadmap/`](roadmap/) stages are the core of the plan. For each stage I build a project manually first, then come back and improve it with inference-engineering techniques (faster kernels, batching, quantization, caching and so on). Each improvement will land on its own branch, `manual/<project>-optimized`, so the before and after can be compared directly.

## Contents

| Project | What it is | When |
|---|---|---|
| [`python_practice/`](python_practice/) | Where it started: daily hand-written Python exercises (`python_basics/days`) and data-science notebooks (Stack Overflow survey and SPY analysis, plus video-course notebooks) | 2026-08 → 2026-10 |
| [`fundamentals/linear_reg_manual`](fundamentals/linear_reg_manual/) | Linear regression from scratch on a house-price dataset | 2026-09 |
| [`fundamentals/mlp_manual`](fundamentals/mlp_manual/) | Character-level MLP that generates names (makemore style) | 2026-10 |
| [`roadmap/Stage_0`](roadmap/Stage_0/) | NumPy fundamentals: softmax/CE, four matmul implementations, an MLP, a latency harness. Exception: the backprop in `Module_3_4.py` isn't mine (the stage README explains why) | 2026-10 |
| [`fundamentals/nano-gpt_manual`](fundamentals/nano-gpt_manual/) | A GPT-style transformer written by hand in PyTorch: data prep, weight init, forward pass (in progress) | 2026-10 → 2026-11 |
| [`roadmap/Stage_1&2`](roadmap/Stage_1&2/) | MNIST MLP in NumPy, plus reteach and test exercises | 2026-11 |
| [`fundamentals/iris`](fundamentals/iris/) | Iris flower classifier: preprocessing, training loop, evaluation | 2026-11 |
| [`bloons/`](bloons/) | **Mixed.** An RL agent that plays Bloons TD 6 from screen pixels on macOS. The game harness (screen reading, input, game-state detection) was built with Claude Code. The RL environment, model, training and scripted baseline are mine (in progress). | 2026-11 → |

## Notes

- Datasets over a few MB, tokenized `.bin` files and model weights are left out because of size.
- `bloons/` includes its full commit history.
