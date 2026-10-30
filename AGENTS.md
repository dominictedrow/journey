# Instructions for Claude Code and other coding agents

## Role and authority
Implement the approved plan incrementally; treat `START_HERE.md` as current checkpoint, `MASTER_PLAN.md` as roadmap, stage files as task contracts, and `DECISIONS.md` as binding approved decisions. **Human approval overrides agent inference.** Disclose uncertainty.

## Authorized without further approval
Read files and public references; run safe read-only diagnostics, lint, tests, backtests on approved local data; implement explicitly approved tasks on their feature branches; make granular local commits. Only push when remote access and pushing have been approved, and avoid exposing private data.

## Must pause and ask the human
Changes to architecture, markets, asset classes, timeframe, vendors/subscriptions, paid resources, data rights, model family escalation, strategy meaning, feature definitions, reward labels, trade frequency, leverage, shorting, order types, exits, risk thresholds, computing budget, persistent infra, live connections, live deployment, external API costs, or scope; inability to corroborate important source details; material test failure requiring a change in method; Git destructive commands, history rewrite, force-push or replacement of uncommitted user files. Also ask if a materially better path emerges. Do not use silence as permission.

## Approval request template
**DECISION REQUIRED [ID]**: observed problem/opportunity; existing plan; proposed deviation; alternatives including stay the course; costs/risks; impact on stage, tests and Git; precise question. Log in `DECISIONS.md` with status PENDING and stop affected work. Unaffected authorized tasks may continue.

## Work loop
CHECKPOINT → VERIFY git and artifacts → READ active stage → IMPLEMENT small task → TEST → COMMIT → UPDATE checkpoint → PUSH if authorized → REPORT. Prefer deterministic, documented and reproducible outputs. Record every experiment's data version, parameters, seed, code SHA and metrics. Reuse saved features and perform inference outside the LLM's loop.

## Engineering rules
Python with type hints, `pytest`, `ruff`, config files, structured logging, UTC-aware timestamps, strict schemas, dependency pinning, reproducible environments. Never use look-ahead features, randomized train/test splitting for time series, or automatically promote a trained model to production. Use idempotent order IDs, exchange calendars, reconnect logic and emergency shutdown. Tests may not call real-money endpoints. No unattended self-modifying live strategy.

## Git rules
- `main` stays stable; protect it with PRs and passing required status checks when available.
- Create one branch per stage or scoped feature: `research/creator-rules`, `feat/data-pipeline`, `feat/svm-baseline`, `test/paper-execution`, `fix/order-dedupe`.
- Commits should be atomic with Conventional Commit names: `feat(data): validate candle timestamps`, `test(risk): reject oversized orders`, `docs(checkpoint): record stage 03 progress`.
- Run checks before commit; commit only relevant paths; never use `git add .` without reviewing status and diff. No secrets/datasets/model binaries in git by default.
- Push the feature branch after meaningful passing checkpoints if remote configured and push approved; open PR, pass CI, get human sign-off at each stage gate, then merge. Do not force-push `main`.
- If the current branch/checkpoint mismatch, stop for approval. Do not automatically reset, rebase, stash or delete.

## Quality and safety
Do not optimize win rate alone: optimize net expectancy and risk-adjusted return subject to loss, drawdown and frequency constraints. Separate model forecasts, positions, execution and risk hard-stops. Crash/restart tests, duplicate fills and API outages are first-class test cases. Always default to NO TRADE on missing/stale data, conflicting state, model error or unknown risk exposure.

## V4 authorization and safety policy
The user approves proposing a new private GitHub repository and approves the broker demo API architecture, but this does **not** mean the repository has been created or external account credentials connected. Request explicit confirmation before account mutations/paid subscriptions. Never infer current Git state from docs: inspect the Mac filesystem and remote first.
No paper orders until Stage 06 gate and approved numeric risk envelope. Every position must have a confirmed protective stop independent of classifier output. Do not assume stops guarantee execution price across gaps/outages. `TradingView` is charting and optional alerts, not the authoritative order/account ledger. Broker demo orders and broker-account state are authoritative.
