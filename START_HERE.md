# CURRENT CHECKPOINT — v14 / 2026-10-30 (read before historical sections)

- **Git:** `main`, `feat/b1-framework`, `feat/imom01`, `feat/trend02` pushed. **Active branch `feat/rev01`** (local; push needs owner OK).
- **B3 progress:** IMOM01 shelved (D-022), TREND02 shelved (D-023). REV01 built and run: **fails after costs**, small gross EURUSD reversion edge (D-024 pending, `research/REV01_RESULTS.md`). Holdout (from 2025-10-01) still sealed.
- **Pattern so far:** three institutional candidates, none survives costs at intraday horizons on this data. Assumed costs (1 MES pt, 1.4 EURUSD pips per round trip) are decisive, and real broker numbers + bid/ask data (D-020) are now the highest-value inputs.
- **Open owner decisions:** D-020 (data), D-021 (MES min size), D-024 (shelve REV01?).
- **Next exact task:** Joovier creator modes J-T01/J-T02/J-T03 as `INTERPRETATION` (see `research/JOOVIER_TRANSCRIPT_FINDINGS.md`), unless the owner redirects to data/costs first.
- Venv at `~/.venvs/trader` (`UV_PROJECT_ENVIRONMENT`); run instructions in `README.md`.
- Still prohibited without separate approval: accounts, purchases, data subscriptions, broker connections, paper/live orders.

---

# PRIOR CHECKPOINT — v13 / 2026-10-30

- **Git:** `main`, `feat/b1-framework`, `feat/imom01` pushed to `origin`. **Active branch `feat/trend02`** (local; push needs owner OK).
- **B3 progress:** IMOM01 shelved (D-022). TREND02 built and run: **fails, even gross of costs** (D-023 pending, `research/TREND02_RESULTS.md`). Holdout (from 2025-10-01) still sealed.
- **Open owner decisions:** D-020 (data deviates from spec), D-021 (MES min size vs $25k/0.25%), D-023 resolved: TREND02 **shelved**.
- **Next exact task:** REV01 (short-horizon mean reversion vs rolling mean/VWAP proxy, MES + EURUSD). TREND02's negative gross edge hints at intraday mean reversion.
- Venv at `~/.venvs/trader` (`UV_PROJECT_ENVIRONMENT`); run instructions in `README.md`.
- Still prohibited without separate approval: accounts, purchases, data subscriptions, broker connections, paper/live orders.

---

# PRIOR CHECKPOINT — v12 / 2026-10-30

- **Git:** `main` = docs root (6dc1218). `feat/b1-framework` = B1 framework (997de4f). **Active branch `feat/imom01`.** Nothing pushed (push needs owner OK).
- **Phase B3 started.** IMOM01 implemented, tested and run: **fails after costs** on dev data (D-022, `research/IMOM01_RESULTS.md`). Holdout (from 2025-10-01) still sealed.
- **Open owner decisions:** D-020 (data deviates from spec), D-021 (MES min size vs $25k/0.25%), D-022 resolved: IMOM01 **shelved**.
- **Next exact task:** TREND02 (multi-speed EMA trend, MES + EURUSD) as the next mode, then REV01 and the Joovier `INTERPRETATION` modes.
- Venv at `~/.venvs/trader` (`UV_PROJECT_ENVIRONMENT`); run instructions in `README.md`.
- Still prohibited without separate approval: accounts, purchases, data subscriptions, broker connections, paper/live orders.

---

# PRIOR CHECKPOINT — v11 / 2026-10-30

- **Phase B1 Framework: built**, awaiting owner review of the demo report. Package in `src/trader/`, tests in `tests/` (36 passing), run instructions in `README.md`.
- **Phase B2 Data: started.** Owner data imported to `../data/cache/*.parquet` (outside the repo). Quality report `reports/data_quality.md`. Holdout **sealed from 2025-10-01** for both instruments; do not pass `--unseal-holdout` until final evaluation.
- **Open owner decisions:** D-020 (S&P file is an index CFD, EUR/USD single-sided — accept as research data?), D-021 (MES too large for 0.25% of $25k with ATR stops — equity/risk choice).
- **Next exact task:** Phase B3 — implement IMOM01 (MES-only, 16:00 ET exit, per paper #19 review) as the first real mode, then TREND02 / REV01 / J-T0x, each with unit tests + documented rules.
- **Git:** repo `main` has **no commits**; first commit awaits owner OK. Venv lives at `~/.venvs/trader` (exFAT drive breaks venvs) — set `UV_PROJECT_ENVIRONMENT`.
- Still prohibited without separate approval: accounts, purchases, data subscriptions, broker connections, paper/live orders.

---

# PRIOR CHECKPOINT — v10 / 2026-10-19

- **Plan re-sequenced (D-018, owner-approved):** build-first. Read the "ACTIVE SEQUENCE" table at the top of `MASTER_PLAN.md`. One bot, switchable strategy modes, after-cost competition to pick the best.
- **Active phase: B1 Framework** (not started). Next exact task: scaffold the Python package, mode interface, backtester, risk layer (mandatory stop) and competition report, tested on synthetic data.
- Stage 01 research continues in parallel (IMOM01 review done; see v9 below).
- **Git (owner-authorized 2026-10-19):** local repo initialized on `main`; private GitHub repo `dominictedrow/trader` created; remote `origin` = https://github.com/dominictedrow/trader.git. No commits/pushes yet; each commit/push still needs owner OK. (Supersedes the earlier planned name `trading`.)
- **Data:** owner is sourcing it. Spec in `research/DATA_REQUIREMENTS.md` (1-min ES/MES OHLCV per contract + EUR/USD 1-min bid/ask, 2015+).
- Still prohibited without separate approval: accounts, purchases, data subscriptions, broker connections, paper/live orders.

---

# PRIOR CHECKPOINT — v9

- **v9 session (Claude Code, filesystem verified):** working dir `/Volumes/DOMS_DRIVE/projects/Code Projects/trade_bot/trading_bot_blueprint`; **not a Git repository** (consistent with no Git authorization). No tests, no code.
- Done this session: 01-02b-3 for paper #19 (IMOM01) — full journal text reviewed, page-referenced in `research/PAPER_19_IMOM_METHODS_REVIEW.md`. Key finding: effect supported for ES/MES (Table B1) but **not for Euro FX futures** (Table B4) → recommend IMOM01 MES-only, exit 16:00 ET, lagged entry, cost sensitivity decisive. Recommendation awaits owner approval at the Stage 01 interpretation gate.
- Next exact task: visually re-verify quoted table values; review Gao et al. (2018) predecessor; then TREND02 sources (#6, #1, #2, #10) and REV01 (#18). 01-02a creator visual validation still needs primary video playback or owner-supplied material.

---

# PRIOR CHECKPOINT — v8 / 2026-10-19

- Active stage **01 — strategy/source research**. Current tasks: `01-02a` creator visual-source validation **and** `01-02b` institutional paper research (new parallel research track).
- Completed 01-02b so far: inventoried 23 supplied references with access caveats; audited selected abstracts, publisher previews and accessible portions of primary PDFs; prepared **hypotheses, not validated models**. No claim to have read all 23 full texts. Next: finish primary-method extraction and page/equation checks for highest-priority papers.
- Read next: `research/INSTITUTIONAL_PAPER_INVENTORY.md`, `research/INSTITUTIONAL_STRATEGY_CANDIDATES.md`, `research/PAPER_EXPERIMENT_TEMPLATE.md`, `stages/01_research.md` and existing creator research files.
- Proposed candidates `IMOM01`, `TREND02`, `REV01` first; overlays `REGIME01`, `EXEC01`; **not authorized to code/trade them** until strategy interpretation review and relevant stage gate. Existing Joovier/Swaggy work remains active and incomplete.
- Approval gates: Stage 01 literature interpretation and selection; Stage 02 transition; later data licensing/hardware purchases; paper order activation; live activation. Previous stage-00 research design already approved.
- **No Git operations, account creation, payments, external connections, paper/live orders authorized.** Private repository `trading` remains uncreated. No verified commits or tests; docs/ZIP-only update.
- Owner-authorized direction: MES and EURUSD pilot, local MES simulator design, OANDA practice feasibility evaluation, protective stop always, volatility-based sizing subject to future numeric signoff.
- IMPORTANT: The earlier checkpoint sections below are historical and do not override this header. On each resumed session, check actual filesystem/Git state before action.

---

# CANONICAL CHECKPOINT OVERRIDE — 2026-10-18 22:40 PDT (v7)

**Read this section first.** It supersedes stale historical checkpoint fields later in this file.
- Active stage: **01 — Creator strategy research**; stage approved, no approval to advance to Stage 02.
- Current task: **01-02a — complete primary-source course review and validate chart demonstrations**.
- Completed this session: public source verification and **partial timestamped transcript audit of Joovier six-hour course** (three candidate strategies). **NO complete videos watched**. Swaggy complete-course research still blocked on verified accessible primary instruction.
- Next reads: `AGENTS.md`, `stages/01_research.md`, `research/COURSE_SOURCE_INVENTORY.md`, `research/JOOVIER_TRANSCRIPT_FINDINGS.md`, `research/SWAGGY_ACCESS_REVIEW.md`.
- Required next actions: access full primary playback, cross-check timestamp offsets + chart frames, extract exact entry/exit criteria, continue legitimate public Swaggy full-course search. No creator indicator accepted for implementation yet.
- Status: **research in progress**, not strategy sign-off. Existing candidate findings are hypotheses.
- Owner authorization: research only. **NO** accounts, purchases, Git operations (including repo creation, commit, push, branch), broker connections, automated paper orders, or live orders.
- Repository: not created; branch/HEAD uninitialized; tests not run. If checkout exists later, inspect actual status before operations.
- Safety: protective stops required on every position; quantitative limits are paper-design proposals, not production authorization.

---

# START HERE — canonical resumable trading bot checkpoint (v6)

Paste the absolute path to this file into **Claude Code with repository filesystem access** and instruct: "Read START_HERE.md and resume only the current approved stage." The path alone cannot force chat tools to read the repository.

## Current state — 2026-10-18 19:31 PDT (owner approval recorded)
- Status: **STAGE 00 RESEARCH DESIGN APPROVED; STAGE 01 RESEARCH AUTHORIZED**.
- Current stage: `01`; active file: `stages/01_research.md`.
- Task: **01-01a** — validate creator identities and public course source inventory; start with `research/COURSE_SOURCE_INVENTORY.md`.
- Stage 00 approval scope: MES + EUR/USD pilot *research*, OANDA practice **feasibility assessment**, local MES simulation design, transition to creator-course research. This is **not** authorization to open or use accounts, purchase data or hardware, create or push a GitHub repo, or activate any orders.
- Planned private GitHub repo name: `trading`; **not created**. No Git operations authorized. No verified branch/HEAD/remote or executed tests.
- Research queue: verified Joovier public video candidates, Swaggy C official academy access restrictions, and unverified purported public vault lead. See `research/COURSE_SOURCE_INVENTORY.md`. No video has been fully watched, chart-validated or converted into strategy rules.
- Proposed risk values from `RISK_POLICY_PROPOSAL.md` remain **unapproved** and cannot enable execution. All broker demo/live order placement disabled.
- Next action: inventory and verify all accessible full public courses; document durations/chapters, obtain legitimate transcripts/frames and tag confidence. Do not infer chart setups from titles/descriptions.
- Read first: `AGENTS.md`, `MASTER_PLAN.md`, `DECISIONS.md`, `STAGE_00_SIGNOFF.md`, `stages/01_research.md`, `research/COURSE_SOURCE_INVENTORY.md`, `research/VIDEO_NOTE_TEMPLATE.md`.
- Stop gate: present a creator corpus/inventory for confirmation when source identity/access is ambiguous; Stage 01 strategy interpretations need separate approval before implementation or Stage 02.

## Mandatory resume procedure (run every new session)
1. Read this file, `AGENTS.md`, `MASTER_PLAN.md`, `DECISIONS.md`, `CHANGELOG.md` and **only the active stage file plus referenced dependencies**. Read `QUESTIONS.md` if any scope answers are outstanding.
2. Confirm working directory, `git status --short --branch`, branch, HEAD SHA, uncommitted changes and actual artifacts. Compare to the checkpoint; never trust the checkpoint blindly.
3. If git state or files conflict with the checkpoint, **stop, explain discrepancy and ask permission** before checkout, reset, stash, overwrite or branch switches.
4. Identify the next unchecked task and its acceptance criteria. Complete only authorized tasks in the active stage. Read the relevant research evidence and decision records.
5. Before writing, give a concise task plan and list any decisions needed; ask approval for deviations or material architecture changes.
6. Implement in small commits on a scoped feature branch. Run tests/linters/data-integrity checks. Never treat failed tests as passing.
7. Update `START_HERE.md`, relevant stage checkboxes, `CHANGELOG.md`, `DECISIONS.md` as needed, with actual branch/SHA, test results and next exact task. Commit checkpoint changes together with associated code. Push to remote if configured and authorized; record push result.
8. At stage end: publish outcome, evidence, tests, open risks, and request approval before proceeding to the next stage.

## Non-negotiables
- No real-money trading without a distinct written authorization after passing readiness gates.
- Never store credentials in git; `.env` and secrets are gitignored; do not echo secrets.
- Never invent strategy rules, video contents, benchmark results, performance or test outcomes.
- Do not change scope, strategy, risk limits, broker, dataset or stage sequence without approval.
- Never imply that resuming from a path is automatic in a chat without repo/file access.
- See `AGENTS.md` for full operational rules.

## Historical owner decisions (2026-10-18, second scoping round)
- Begin research/testing across **futures and forex in parallel**; specific contracts and pairs pending. Crypto is phase-later candidate.
- Capital amount is explicitly **undecided**; **adaptive volatility-based sizing** preferred but hard ceilings and risk stops are mandatory and pending.
- Overnight positions permitted **conditionally** when strategy/risk model supports; do not infer unlimited overnight permission. Active positions get first priority on dashboard charts.
- Development: **Claude Code inside VS Code on Mac**; buy a dedicated mini PC later, avoid purchase until hardware acceptance benchmark.
- Success: compare net after-cost returns, drawdown, Sharpe/Sortino, expectancy, profit factor, stability and execution reliability before thresholds are selected.
- TradingView: owner initially proposed **TradingView Paper Trading** then brokerage connection, but later **approved independent broker demo API** as the execution architecture, because TradingView states Pine strategies cannot place orders to its integrated paper account/broker directly. See `TRADINGVIEW_FEASIBILITY.md`.
- Strategy research: prioritize evaluating **full available courses** rather than relying on creator promotional videos. Access rights, course identity, coverage and evidence tracking still need confirmation. Supplied videos remain identity-only.
- Next required reading also includes `TRADINGVIEW_FEASIBILITY.md`, `REQUIREMENTS_DRAFT.md`.

## Latest owner decisions — third scoping round (2026-10-18)
- **APPROVED architecture choice:** TradingView provides charting and optionally indicators/alerts; autonomous paper-order placement belongs to a separately authenticated **broker demo API**, not TradingView's native paper account. Do not route live or demo orders through unverified TradingView browser automation.
- **APPROVED:** Agent may *recommend*, not silently finalize, a small futures-and-forex symbol universe based on verified availability, spreads, commissions, data rights, liquidity, session schedules, and rollover.
- **APPROVED research:** Inventory and study publicly available complete courses for Swaggy C and Joovier; identity-confirmation links must not be mistaken for assigned strategy videos. Document coverage and timestamp evidence; no claim of having watched material before it is actually accessed and analyzed.
- **TRADINGVIEW:** User has Essential subscription; verify specific alert limits and exchange data entitlements at setup; never equate subscription tier to licensed futures quotes.
- **MANDATORY:** Every open position has a broker-supported protective stop independent of inference; block new entries if stop placement cannot be confirmed, and reconcile stop/order status after disconnections. Paper and live policies separate.
- **GIT:** New private GitHub repository is desired but NOT YET CREATED. Initialize local Git/remote when authorized through Claude Code; private repository, PR reviews and stage gates. No credentials committed.
- Current recommended broker *candidates*, not approvals: OANDA v20 practice (forex); Tradovate demo (futures) subject to eligibility and API subscription requirements. See `PROVIDER_EVALUATION.md`.
- Resume notes: actual repository location, branch and commit are **not verified here**; agents must query local Git rather than assuming the checkpoint represents actual Git state.
