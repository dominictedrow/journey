# quant_trader

> **Part of [journey](https://github.com/dominictedrow/journey)**, Dom's ML and software portfolio.
>
> **How this was built: 🤖 vibe coded.** AI coding agents (mostly Claude Code) wrote the code. I picked the goal, made the design calls, and steered and tested the work.
>
> **What it is:** an ML quant trading pipeline. It collects point-in-time market, macro, crypto and news data and builds a quant feature layer. A homemade LLM plus four base models (Transformer, MLP, two CNNs) and a meta learner score assets. A deterministic rules engine turns those scores into orders, which run on Alpaca **paper** trading.
> **When:** 2026-11 →

## Status
Phase 0 (framework) is done. See [`START_HERE.md`](START_HERE.md) for the current state and next task.

## Map
| Path | What |
|---|---|
| `START_HERE.md` | Current checkpoint and next task (read first) |
| `CLAUDE.md` | Session protocol and hard rules for Claude Code |
| `docs/spec/` | The original system spec (v2) |
| `docs/plan/` | Master plan (phases and gates), API registry (free/paid flags) |
| `docs/interrogation/` | Per-phase question banks and answers |
| `docs/components/` | Per-component build specs |
| `docs/handoffs/` | Session handoffs |
| `docs/setup/` | Setup guides (Windows migration) |
| `DECISIONS.md` / `EXPERIMENTS.md` / `CHANGELOG.md` | Decision log, experiment index, version history |
| `experiments/` | Pre-registered experiment designs and results |
| `configs/` | YAML config (every threshold and model size) |
| `src/qt/` | Code: common, collect, db, features, labels, validation, news, models, portfolio, execution, orchestrator |
| `tests/` | Mirrors `src/qt` |
| `data/ models/ logs/ reports/` | Local only (gitignored) |

Not financial advice. Paper trading only.
