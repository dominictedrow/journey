# journey

My path through machine learning and software, kept as two tracks side by side:

- **✍️ Manual**: code I write myself, to actually understand how things work.
- **🤖 Vibe coded**: projects where AI coding agents (mostly Claude Code) wrote the code. I picked the goal, made the design calls, and steered and tested the work.

Every project is labeled with which track it belongs to, what it's for, and when I built it.

## How to read this repo

| Branch | What's on it |
|---|---|
| **`main`** (you are here) | This map, plus my mentor's code (see below) |
| [**`manual-models`**](https://github.com/dominictedrow/journey/tree/manual-models) | All manual work in one branch: Python practice, from-scratch models, the roadmap stages, and the Bloons RL agent |
| [**`vibe-projects`**](https://github.com/dominictedrow/journey/tree/vibe-projects) | Index of the vibe-coded projects. Each one lives on its own `vibe/<name>` branch with its full history |

## Where it's going

I started out vibe coding: I could ship working apps with AI agents before I could write much Python myself. The manual track is how I close that gap. The plan from here is a staged roadmap: **build each project manually, then improve it with inference-engineering techniques** (faster kernels, batching, quantization, caching). Each optimized version will land on its own branch next to the original, so the before and after can be compared directly.

## Manual track ✍️

| Project | What it is | When |
|---|---|---|
| [python_practice](https://github.com/dominictedrow/journey/tree/manual-models/python_practice) | Daily hand-written Python exercises and data-science notebooks | 2026-08 → 10 |
| [linear_reg_manual](https://github.com/dominictedrow/journey/tree/manual-models/fundamentals/linear_reg_manual) | Linear regression from scratch (house prices) | 2026-09 |
| [mlp_manual](https://github.com/dominictedrow/journey/tree/manual-models/fundamentals/mlp_manual) | Character-level MLP that generates names | 2026-10 |
| [Roadmap Stage 0](https://github.com/dominictedrow/journey/tree/manual-models/roadmap/Stage_0) | NumPy fundamentals: softmax/CE, matmul four ways, an MLP, a latency harness | 2026-10 |
| [nano-gpt_manual](https://github.com/dominictedrow/journey/tree/manual-models/fundamentals/nano-gpt_manual) | A GPT-style transformer written by hand (in progress) | 2026-10 → 11 |
| [Roadmap Stage 1&2](https://github.com/dominictedrow/journey/tree/manual-models/roadmap/Stage_1%262) | MNIST MLP in NumPy | 2026-11 |
| [iris](https://github.com/dominictedrow/journey/tree/manual-models/fundamentals/iris) | Iris flower classifier | 2026-11 |
| [bloons](https://github.com/dominictedrow/journey/tree/manual-models/bloons) | RL agent that plays Bloons TD 6 from pixels. **Mixed:** the game harness was built with Claude Code; the RL model and training are mine | 2026-11 → |

## Vibe track 🤖

| Project | What it is | When |
|---|---|---|
| [instana](https://github.com/dominictedrow/journey/tree/vibe/instana) | Instagram reel analyzer: Grok vision classification and format clustering (Electron) | 2026-05 |
| [news_bot](https://github.com/dominictedrow/journey/tree/vibe/news_bot) | Global Macro market-research dashboard (Flask + PostgreSQL, macOS app) | 2026-05 → 06 |
| [cold_email](https://github.com/dominictedrow/journey/tree/vibe/cold_email) | Gmail mail merge for sponsorship outreach | 2026-06 |
| [polymarket_copy](https://github.com/dominictedrow/journey/tree/vibe/polymarket_copy) | Polymarket wallet scoring, backtesting and paper copy-trading | 2026-07 |
| [cnn_1](https://github.com/dominictedrow/journey/tree/vibe/cnn_1) | My first ML project: MNIST digit CNN | 2026-07 |
| [cnn_2](https://github.com/dominictedrow/journey/tree/vibe/cnn_2) | Cats vs Dogs CNN | 2026-07 |
| [asl](https://github.com/dominictedrow/journey/tree/vibe/asl) | Sign-language recognition from video, with live webcam inference | 2026-07 → 08 |
| [c_work](https://github.com/dominictedrow/journey/tree/vibe/c_work) | Personal automation suite: Claude skills plus a macOS app | 2026-07 → 10 |
| [kaggriculture](https://github.com/dominictedrow/journey/tree/vibe/kaggriculture) | Kaggle simulation-competition agent | 2026-08 → 09 |
| [knees](https://github.com/dominictedrow/journey/tree/vibe/knees) | RSNA Knee Abnormality Detection (Kaggle 2026) | 2026-09 → 10 |
| [envenda](https://github.com/dominictedrow/journey/tree/vibe/envenda) | Enveda CASMI 2026 (Kaggle): molecule ID from mass spectra | 2026-10 |
| [trader](https://github.com/dominictedrow/journey/tree/vibe/trader) | Research trading bot with an after-cost strategy competition | 2026-10 → 11 |

## Mentor's code (not mine)

> **Disclaimer:** The code in [`J_trade_ML/`](J_trade_ML/) isn't my code and I take no credit for writing it. It's my mentor's XAUUSD (gold) trading ML research, kept here only as study material. `J_trade_ML/CODEBASE_GUIDE.md` is the reading map I use to work through it.

## Notes

- Large datasets, model weights and database files are left out because of size. Code, notebooks and small artifacts are included.
- The older standalone repos for these projects are archived and point back here.
