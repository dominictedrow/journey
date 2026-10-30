## v13 — 2026-10-30 — B3: TREND02 mode
- IMOM01 shelved per D-022 (removed from default configs). Branches `main`, `feat/b1-framework` and `feat/imom01` pushed to `origin`.
- Branch `feat/trend02`. Added `trend02` multi-speed EMA ensemble with 10 rule tests; generic look-ahead tests cover it. 60 tests pass; ruff clean.
- Competition reporting eras changed to 2004–11 / 2012–19 / 2020+.
- Result: TREND02 fails even gross of costs (D-023, `research/TREND02_RESULTS.md`). Holdout untouched.

## v12 — 2026-10-30 — B3: IMOM01 mode
- Branch `feat/imom01` (from `feat/b1-framework`). Tests: 48 passed; ruff clean.
- Added `imom01` mode (paper #19 adaptation) with 10 rule tests (signal timing, DST, early close, missing bars, forced 16:00 flat, lag, ONFH filter). Engine: optional `Mode.decision_mask` so clock-time modes skip irrelevant bars.
- Competition: labelled variants, per-period breakdown (paper sample vs post-paper), `configs/sensitivity_300k.toml` sizing sensitivity.
- Result: IMOM01 fails after costs (D-022, `research/IMOM01_RESULTS.md`). Holdout untouched.

## v11 — 2026-10-30 — Phase B1 framework built
- Branch/HEAD: `main`, no commits (commit awaits owner OK). Tests: `uv run pytest` → 36 passed; `ruff check` clean.
- Added Python package `src/trader/`: vendor CSV loader + validation + frozen Parquet snapshots with sha256 provenance; causal resampling (daily bars roll at 17:00 ET, DST-safe); NYSE sessions/early closes (`exchange_calendars`); mode interface (`SOURCE/ADAPTATION/INTERPRETATION/BASELINE`); risk layer (mandatory stop, policy sizing formula, min-size veto, daily loss breaker); event-driven backtester (next-bar fills, 1-minute stop/target checks, gap-through fills, stop-first on ambiguous bars, stale-data NO TRADE); metrics; competition runner with sealed holdout and experiment log; CLI.
- Baseline modes: `no_trade`, `always_long`, `random`. Generic no-look-ahead tests run against every registered mode.
- Imported owner data (B2 start): `reports/data_quality.md`. Holdout sealed from 2025-10-01 (`reports/holdout.json`). First competition: `reports/competition_*.md`.
- New pending decisions: D-020 (data deviates from spec), D-021 (MES min size vs risk envelope).
- No accounts, purchases, orders or external connections.

## v10 — 2026-10-19
- D-018: re-sequenced to build-first with switchable strategy modes + competition (owner-approved). Added ACTIVE SEQUENCE section to `MASTER_PLAN.md`; v10 notes on stage files 01–05; checkpoint now Phase B1.
- Docs only. No code, Git, accounts or orders.

## v9 — 2026-10-19
- Branch/HEAD: none (directory is not a Git repo; no Git operations authorized). Tests: none (docs-only).
- 01-02b-3 (partial): full-text methods review of paper #19 Baltussen, Da, Lammers, Martens (2021, JFE 142:377–403) → `research/PAPER_19_IMOM_METHODS_REVIEW.md`.
- Updated inventory #19 access status, IMOM01 register row (EURUSD unsupported; 16:00 ET exit), Stage 01 checklist, START_HERE checkpoint.
- No accounts, purchases, orders, external connections or Git actions.

## v8 — 2026-10-19
- Added source-by-source inventory for 23 academic and institutional resources, including limitations and corrections to mismatched BIS URL.
- Added 11 strategy/overlay concepts organized by genuine return mechanism vs execution and risk controls; registered priority research order and falsification protocol.
- Updated Stage 01 research checklist and canonical resume checkpoint. No code deployment, orders, accounts or Git operations.

# Session and checkpoint log

## 2026-10-18 — initial blueprint
- Created staged research and implementation blueprint.
- No repository, market feed, code, model, video transcription or trades have been executed.
- All substantive scope decisions are pending.

After every session append: date, branch, HEAD SHA, completed task IDs, files changed, exact test commands and results, push outcome, unresolved questions and next task.

## 2026-10-18 — scope refinement
- Recorded approved intraday futures/FX-first, optional crypto, both directions, mini PC, and dashboard foundation.
- Created requirements draft and architecture/provider comparison; updated active task to 00-02.
- No videos were watched or transcribed in this update. The provided links are for creator identification only.
- No Git repository or trades created; no tests executed beyond Markdown/zip packaging verification.


## 2026-10-18 — v3 scoping refinement
- Recorded second-round answers, added TradingView feasibility analysis and official sources, corrected distinction between strategy emulator, built-in paper account and real broker execution.
- Promoted full authorized creator courses to Stage 01 primary corpus; confirmed Mac VS Code developer host and mini-PC target; refined open questions to material gating decisions.
- No source videos watched; no software implemented; no broker or Git repo configured; all order modes remain disabled.

## 2026-10-18 — v4 requirements refinement
- Locked TradingView Essential charting + broker demo API automatic paper execution; superseded native TradingView paper account automation assumption.
- Owner authorized a small futures + forex universe recommendation; MES/EURUSD are research candidates, not approved instruments.
- Public complete creator courses prioritized; require evidence coverage and source logs.
- Broker-confirmed protective stops mandatory for every position.
- New private GitHub repository requested but not created; numerical limits and broker selection still gated.
- Updated checkpoint and Stage 00 acceptance gates.

## v5 — 2026-10-18 PDT
- Added US-specific provider feasibility with dated sources, explicit funded Tradovate API requirement and data-license caveats.
- Proposed MES/EURUSD primary research pilot and optional MNQ/GBPUSD comparisons.
- Added separate draft numeric simulation risk policy, Stage 00 sign-off dossier, and private GitHub setup checklist.
- Refreshed checkpoint for design-ready / not-yet-approved status; no external actions performed.

## v6 — 2026-10-18 PDT
- Recorded explicit Stage 00 design sign-off; checkpoint advanced to Stage 01 task 01-01a.
- Added course source inventory and Stage 01 kickoff task contract with validation and provenance gates.
- Retained separate authorization gates for GitHub, accounts, cost, risk limits, demo/live execution. No repository operations, account activity or video watching claimed.

## v7 — 2026-10-18: partial educational transcript audit
- Updated Stage 01 checkpoint; documented 3 Joovier strategy hypotheses from timestamped third-party transcript.
- Logged Swaggy public-course access gaps. Explicitly **no complete videos watched**; no new account, costs, git or trading actions.
