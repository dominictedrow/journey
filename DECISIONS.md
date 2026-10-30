# Decision log

Every row is an intentional decision. Do not rewrite history; supersede decisions with a new ID.

| ID | Date | Decision | Status | Approver | Evidence / impact |
|---|---|---|---|---|---|
| D-000 | 2026-10-18 | No live-money deployment authorized by this project blueprint | ACTIVE | Project safety baseline | Gate stage 07 |
| D-001 | 2026-10-18 | Futures and forex priority; crypto optional; long/short intraday | APPROVED SCOPE / BROKER PENDING | User | User response; `REQUIREMENTS_DRAFT.md` |
| D-002 | 2026-10-18 | Define capital, allowable loss and position limits | PENDING | User | `QUESTIONS.md` 8-12 |
| D-003 | 2026-10-18 | Reference videos identify Swaggy C and Joovier; these are NOT strategy-analysis assignments | PARTIAL | User | Links in requirements; priority research corpus pending |
| D-004 | 2026-10-18 | Confirm model label, metric and timeframe | PENDING | User | `QUESTIONS.md` 20-25 |
| D-005 | 2026-10-18 | Autonomous mini-PC runtime, local dashboard foundation, agent pauses for deviations/stage gates | APPROVED SCOPE / IMPLEMENTATION PENDING | User | `ARCHITECTURE_OPTIONS.md` |

Decision template: ID; date; context; previous approach; proposed change; alternatives; cost, tradeoff and risk; tests needed; human response; status APPROVED/REJECTED/PENDING/SUPERSEDED; affected files and stages.

| D-006 | 2026-10-18 | Futures + forex in first research wave; representative instruments TBD | APPROVED SCOPE / SYMBOLS PENDING | User | Second scoping response |
| D-007 | 2026-10-18 | Adaptive volatility sizing and conditional overnight positions, subject to independent hard limits | APPROVED PRINCIPLE / LIMITS PENDING | User | Second scoping response |
| D-008 | 2026-10-18 | Mac VS Code Claude Code development; mini PC purchase later; dashboard active positions first | APPROVED SCOPE | User | Second scoping response |
| D-009 | 2026-10-18 | TradingView native paper first, broker later via TV: feasibility requires proof, no assumption of direct automation | PENDING FEASIBILITY | User preference / engineering review | `TRADINGVIEW_FEASIBILITY.md`, TradingView official docs |
| D-010 | 2026-10-18 | Research full legitimately accessible creator courses and compare multiple performance metrics before setting promotion thresholds | APPROVED RESEARCH SCOPE | User | Course access unresolved |

| D-011 | 2026-10-18 | TradingView for visual analysis; broker demo API for autonomous paper trading | APPROVED ARCHITECTURE | User | Replace D-009; `TRADINGVIEW_FEASIBILITY.md` |
| D-012 | 2026-10-18 | Agent recommends small initial futures/forex universe for owner sign-off | APPROVED TO RECOMMEND | User | Actual symbols not approved yet |
| D-013 | 2026-10-18 | Analyze all publicly accessible full courses by Joovier and Swaggy C where discoverable and feasible, with coverage ledger | APPROVED RESEARCH | User | No course yet claimed analyzed |
| D-014 | 2026-10-18 | TradingView Essential subscription; verify alert and futures market data entitlements | CONFIRMED SUBSCRIPTION / ENTITLEMENTS PENDING | User | Avoid equating plan with exchange feed |
| D-015 | 2026-10-18 | Require independent confirmed protective stop for every position | APPROVED NON-NEGOTIABLE | User | Risk engine and fill reconciliation tests |
| D-016 | 2026-10-18 | Create new private GitHub repository | APPROVED INTENT / NOT EXECUTED | User | Owner must authorize account access and creation |
| D-017 | 2026-10-18 | Provider choice, numeric risk thresholds, pilot instruments and stage advance | PENDING SPECIFIC APPROVAL | Owner | Stage 00 gate |

## v5 owner decisions — 2026-10-18 PDT
- APPROVED preference: brokerage jurisdiction United States; research futures-demo costs rather than purchasing; private GitHub repo name `trading`; have agent recommend quantitative risk limits.
- PROPOSED only: MES + EUR/USD pilot, OANDA v20 Practice FX adapter, local MES simulator, Tradovate as conditional future option, simulation risk numbers in `RISK_POLICY_PROPOSAL.md`.
- PENDING: Stage 00 sign-off, GitHub account operations, any fees/deposits, risk-number approval, demo and live order activation.

## D-Stage00-Approval — 2026-10-18 PDT
**Approved**: Stage 00 research design; MES/EURUSD pilot research, OANDA practice feasibility evaluation, local MES simulator design, and Stage 01 Joovier/Swaggy C public creator-course research. **Explicitly NOT authorized**: any accounts, credentials, GitHub operations, spending, automated demo orders, live orders. Numerical risk policy and creator-rule translations require separate approval.

## 2026-10-19 — research scope addition
Owner asked to analyze 23 institutional/academic sources and add **different profitability attempts**. Research-only extension to Stage 01 recorded, not strategy implementation or capital authorization. Open approval gate: select which hypotheses to carry into Stage 02 after methods review.

## D-018 — 2026-10-19 — Build-first re-sequencing with switchable strategy modes
- **Context:** Owner reviewed the plan and asked for the whole bot to be built first, with strategies added as switchable modes and compared to find the best.
- **Previous approach:** research-first. Stage 01 sign-off gated design (02), data (03), features (04) and backtests (05); one SVM classifier was the central model.
- **Approved change:** build the bot framework now (Phase B1). Each strategy is a self-contained, config-selectable **mode** behind one interface. A chronological after-cost **competition** across all modes and baselines selects the leader. Stage 01 research continues in parallel and refines modes; it no longer blocks building.
- **Ambiguous creator rules:** may be built as modes labelled `INTERPRETATION` (agent-defined parameters, documented). They must never be described as creator-faithful.
- **SVM/ML:** demoted from central model to an optional later mode/ensemble entrant in the competition.
- **Unchanged and still binding:** D-000 (no live money), D-015 (confirmed protective stop on every position), no paper/demo orders before an approved risk envelope, no accounts, purchases, data subscriptions, broker connections or Git operations without separate approval. Leakage, cost and holdout rules in `AGENTS.md` still apply.
- **Status:** APPROVED. **Approver:** Owner (Dom), 2026-10-19 chat.
- **Affected:** `MASTER_PLAN.md`, `START_HERE.md`, `stages/01`–`05`.

## D-019 — 2026-10-19 — Repository created
Owner asked for a new repo named `trader` with a remote. Created private `dominictedrow/trader` (private per D-016); local `git init -b main`; `origin` configured. Supersedes the planned name `trading`. Commits and pushes still require owner approval per AGENTS.md. **Status:** EXECUTED.

## D-020 — 2026-10-30 — Supplied data deviates from the data spec
- **Context:** Owner supplied `data/EURUSD.csv` (2003-05 → 2026-10) and `data/USA500IDXUSD.csv` (2011-09 → 2026-10), 1-minute OHLCV.
- **Spec (`research/DATA_REQUIREMENTS.md`):** per-contract ES/MES futures with real volume; EUR/USD bid **and** ask.
- **Delivered:** S&P 500 **index CFD** (continuous, no contract rolls, fractional tick-volume that is not usable for VWAP, partial overnight coverage before 2018). EUR/USD **single-sided** price (bid or mid, unconfirmed). Both verified UTC; bar-start labelling assumed.
- **Interim handling (agent, reversible):** S&P CFD used as an ES/MES price proxy with MES contract specs for sizing/costs; FX spread modeled (1.0 pip assumed) rather than observed. REV01/J-T03 VWAP on the S&P will be a proxy at best.
- **Question:** accept these files as the B2 research dataset (with the caveats above), or source per-contract futures / FX bid-ask before trusting any ranking?
- **Status:** PENDING owner.

## D-021 — 2026-10-30 — MES minimum size vs proposed risk envelope
- **Observed:** at the proposed $25k paper equity and 0.25% risk/trade ($62.50), one MES contract is too large for typical stops: a 1.5× hourly-ATR stop is ~$79 per contract (needs ≈$34k equity), a 2× daily-ATR stop ~$642 (needs ≈$259k). The risk layer correctly vetoes these as `below_min_size`; the S&P always-long baseline therefore takes **zero** trades.
- **Options:** (a) raise paper equity; (b) raise risk/trade (e.g. 0.5%); (c) restrict MES to modes with tight intraday stops; (d) keep as-is and let MES only trade when stops are small. Not changed by the agent — risk thresholds require owner approval (AGENTS.md).
- **Status:** PENDING owner.

## D-022 — 2026-10-30 — IMOM01 fails after costs on development data
- **Observed:** gross edge +0.7 bp/day (paper sample +1.0, post-2020 +0.3) against ≈2.8 bp assumed round-trip cost. All three pre-registered variants lose at 1×/2×/3× costs at both $25k and $300k equity. Engine reconciles with an independent calculation. Details: `research/IMOM01_RESULTS.md`.
- **Recommendation:** shelve IMOM01 as a stand-alone mode; keep it registered and logged as a failed hypothesis; do not spend the holdout on it. Revisit only with real ES data, real MES costs, or gamma data (IMOM01-G, paid).
- **Status:** PENDING owner.
