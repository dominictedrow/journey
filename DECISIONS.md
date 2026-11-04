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
- **Status:** APPROVED — shelved. **Approver:** Owner (Dom), 2026-10-30 chat. Removed from default competition configs; mode, tests and results kept.

## D-023 — 2026-10-30 — TREND02 (intraday trend) fails, even gross of costs
- **Observed:** all four pre-registered variants (unanimous ensemble; 15m, 60m and 4h alone) lose on EURUSD and the S&P CFD at 1×/2×/3× costs, at both $25k and $300k. At zero cost they are flat to slightly negative, consistently across 2004–11, 2012–19 and 2020+. Details: `research/TREND02_RESULTS.md`.
- **Recommendation:** shelve TREND02 and keep the holdout sealed. The negative gross edge supports testing REV01 (mean reversion) next. Slower trend (TSM01, daily/monthly) is the faithful test of the sources and would be a separate, newly registered hypothesis.
- **Status:** APPROVED — shelved. **Approver:** Owner (Dom), 2026-10-30 chat. Removed from default competition configs; mode, tests and results kept.

## D-024 — 2026-10-30 — REV01 fails after costs; small gross FX reversion edge
- **Observed:** all six pre-registered variants (07db1ec) lose 65–100% at 1× costs on $25k and $300k. Gross per trade is near zero (EURUSD 5m +0.03R; costs ≈0.17R). The EURUSD gross edge grew from −0.30 bp (2004–11) to +0.37 bp (2020+). The S&P shows no consistent gross edge. Details: `research/REV01_RESULTS.md`.
- **Multiple testing:** REV01 was queued after TREND02's negative gross result, so treat it as an informed test.
- **Recommendation:** shelve REV01 (market-order execution) and keep the holdout sealed. Revisit EURUSD short-horizon reversion only with bid/ask data (D-020) plus a passive-execution model (EXEC01).
- **Status:** APPROVED — shelved. **Approver:** Owner (Dom), 2026-10-31 chat. Removed from default competition configs; mode, tests and results kept.

## D-025 — 2026-10-30 — Joovier modes J-T01..J-T03 fail after costs
- **Observed:** all six pre-registered INTERPRETATION variants (4f198e6) lose at 1×/2×/3× costs on $25k and $300k. The 1-minute modes (J-T01, J-T03) have ≈0 gross edge against 0.2–0.5R costs per trade. J-T02 (squeeze breakout, 10–26 trades a year) loses least (−5% to −10% at $25k), but its gross R is ≈0 or negative. Details: `research/JOOVIER_RESULTS.md`.
- **Caveat:** these test our formalisation of a partial transcript, not the creator's method.
- **Recommendation:** shelve all three and keep the holdout sealed. The B3 queue is now done with no survivor, so prioritise real costs and better data (D-020) before B4.
- **Status:** APPROVED — shelved. **Approver:** Owner (Dom), 2026-10-31 chat. Removed from default competition configs; modes, tests and results kept. Owner chose real costs/data (D-020) as the next direction.

## D-020 update — 2026-10-31 — cost check done; data route proposed
- Public broker numbers (`research/BROKER_COSTS.md`) put realistic costs at MES ≈ $0.66–0.96/side all-in plus spread, and EUR/USD ≈ 1.0–1.7 pips per round trip. The assumed costs (≈1 MES pt, 1.4 pips) are realistic. **No realistic cost rescues any shelved mode**: gross edges are 4–15× smaller than best-case costs. `instruments.py` is unchanged.
- The open question is whether the gross signals are real. That needs (1) the **ask side** of the same EUR/USD and S&P CFD files (free, if the source is Dukascopy as the format suggests), and optionally (2) **per-contract ES 1-minute data** (Databento; paid, needs an account).
- **Question:** confirm the files' source and whether you'll export the ask side; approve or decline a Databento ES purchase.
- **Status:** PENDING owner.

## D-026 — 2026-11-03 — REV01 EUR/USD rerun on real bid/ask data
- **Data:** Dukascopy EUR/USD 1-minute bid **and** ask, 2003-05 → 2026-11, fetched free with `scripts/fetch_dukascopy.py` (no account). The supplied `EURUSD.csv` is the **bid** side (99.9995% exact match); timestamps UTC, bar-start confirmed. The S&P CFD ask side is 82% downloaded; Dukascopy has blocked this IP for now.
- **Observed:** the gross reversion edge is identical on mid and bid bars, so it is not a quote-noise artifact. At Dukascopy's own spread (≈0.6 pip RT, no slippage) fx_5m loses 96% over 2004–2025 but is ≈ break-even after 2012 (+0.1 bp/trade, post-hoc slice). At retail costs it fails in every era. Details: `research/REV01_RESULTS.md` addendum.
- **Engine change:** `Backtester.run(..., half_spread=arr)` optional observed per-bar spread; default behaviour unchanged (2 tests added, full suite 105 pass).
- **Recommendation:** keep REV01 shelved. The one open route is EXEC01 (passive limit entries that earn the spread), modelled on this bid/ask data. That is a new pre-registered hypothesis and changes order types, so it needs owner approval.
- **Status:** PENDING owner (approve EXEC01 design, or close the EUR/USD reversion line).

## D-027 — 2026-11-03 — EXEC01 passive-execution test of REV01 EUR/USD (design pre-registered)
- **Context:** owner said yes to designing EXEC01 after D-026. Design and pass criteria are fixed in `research/EXEC01_DESIGN.md` before any code or P&L run.
- **Key finding while designing:** commission-based accounts (OANDA core, ≈1.0 pip RT commission) can't beat a ≈0.3–0.4 pip gross edge however good the execution is. Only spread-only retail pricing is tested, with an ECN floor as an upper bound.
- **Proposed:** implement limit-entry / limit-take-profit orders in the backtester (backtest-only; ≈1 day, tests first), then run variants E1 (passive entry) and E2 (passive entry + exit) once on the dev window.
- **Affects:** order types (AGENTS.md pause item), engine, multiple-testing log (+2 variants). Holdout stays sealed.
- **Question:** approve implementing and running EXEC01 as designed?
- **Status:** APPROVED and run. **Approver:** Owner (Dom), 2026-11-03 chat.
- **Result:** FAIL on every pre-registered criterion except fill rate, for E1 and E2, in every scenario. Adverse selection: filled signals −0.15R over the next hour vs +0.60R for unfilled ones. Even the optimistic fill rule loses in every era. Details: `research/EXEC01_RESULTS.md`. EXEC01-on-REV01 shelved for good, and the EUR/USD reversion line is closed. Holdout untouched.

## D-028 — 2026-11-03 — TSM01 monthly time-series momentum (pre-registered)
- **Context:** after EXEC01 closed the intraday EUR/USD line, owner chose slower trend (TSM01) and approved building it (chat, 2026-11-03).
- **Design:** `research/TSM01_DESIGN.md`. 12-month sign (primary) and 3-month (secondary), monthly rebalance, 3σ_monthly protective stop, carry from free FRED rates (DFF, ECBDFR) plus an assumed 1%/yr retail FX markup.
- **Structural finding:** at a 3σ monthly stop, one MES contract risks ≈ $3,400, so the PROPOSED 0.25% rule needs ≈ $1.4M of equity. The S&P is tested only as a scale-free benchmark (extends D-021).
- **Status:** APPROVED to build and run. **Approver:** Owner (Dom), 2026-11-03 chat.
- **Result:** FAIL for both instruments (L=12). EUR/USD Sharpe −0.09. S&P Sharpe +0.52 but below buy-and-hold's +0.90 (long in 86% of months). EUR/USD L=3 hits 0.30 at 1× only (secondary, can't pass alone). The bot run shows the 0.25% risk rule gives ≈7% exposure at monthly stops, so P&L is negligible. Details: `research/TSM01_RESULTS.md`. Shelved, holdout untouched.

## D-029 — 2026-11-03 — TSM02 diversified trend on 24 ETFs (research-only, pre-registered)
- **Context:** TSM01 showed single-market trend is weak. Owner chose to test whether the published diversified edge exists in free data (chat, 2026-11-03).
- **Data:** Yahoo public chart endpoint (no account, unofficial, personal research), 24 ETFs across equities, bonds, commodities and currencies. Fetched with `scripts/fetch_yahoo.py` into `../data/yahoo/`, outside the repo.
- **Design:** `research/TSM02_DESIGN.md`. MOP-2012 rules (12-month sign, 40%/σ per asset, equal-weighted), 5 bp costs and 0.5% borrow. The criteria include low correlation with SPY.
- **Scope:** research-only. It adds no instruments to the bot, and trading it would need a separate owner decision.
- **Status:** APPROVED to build and run. **Approver:** Owner (Dom), 2026-11-03 chat.
- **Result:** FAIL (2 of 4 criteria). Sharpe +0.29 (needs 0.40) and +0.11 at 3× costs (needs 0.20). It's positive in every era and has −0.17 correlation with SPY, so the diversified edge exists but is modest and cost-sensitive. Commodities carry it. The script was run once just before being committed, which is disclosed in 3b9d3be. Details: `research/TSM02_RESULTS.md`. Trend line closed, holdout untouched.

## D-030 — 2026-11-03 — Goal change: core SPY + trend sleeve portfolio (PORT01)
- **Context:** eight hypothesis families found no standalone edge after costs. The robust findings are the S&P risk premium and a modest diversified trend sleeve (TSM02: Sharpe 0.29, −0.17 correlation with SPY).
- **Approved change (owner, 2026-11-03 chat):** option (b). The bot's goal becomes a risk-controlled core-long portfolio with a diversifying trend sleeve, judged against buy-and-hold SPY. Owner choices: **margin account, ≤ 2× gross**; **SPY ETF** as the core (replacing MES for the equity exposure); **backtest only** for now.
- **Affects:** markets/instruments (ETFs instead of MES/EURUSD, for this portfolio), leverage (≤ 2×), shorting (sleeve), success metric. Still binding: D-000 (no live money), D-015 (stops, see note), and no accounts, orders or paper trading without separate approval.
- **D-015 note:** a monthly-rebalanced ETF portfolio has no per-trade protective stop. If PORT01 passes, the risk controls (drawdown breaker, gross cap) must be agreed as D-015's equivalent before Stage 06.
- **Design:** `research/PORT01_DESIGN.md`.
- **Status:** APPROVED (direction and choices).
- **Result:** PORT01 FAIL (4 of 5). Sharpe 0.57 vs SPY 0.66 and 60/40 0.72. Max DD 23.6% vs SPY 50.8% (the one pass). Retail leverage costs absorb the sleeve's diversification. Details: `research/PORT01_RESULTS.md`. The strategy search ends: no active strategy beat a simple allocation after costs. Holdout untouched.

## D-031 — 2026-11-03 — Plan-only rebalancing bot
- **Context:** PORT01 ended the strategy search (D-030). Owner chose to repurpose the bot as a rebalancer for a plain allocation (chat, 2026-11-03).
- **Built:** `trader rebalance` (holdings + latest prices → proposed trades) and `trader rebalance-backtest` (rule comparison). The default allocation is 60/40 SPY/IEF with a ±5pp drift band. Package: `src/trader/rebalance/`.
- **Safety:** plan-only. No broker connection, credentials or orders; trades are placed manually by the owner. It defaults to NO TRADE on missing, stale (>4 days) or suspicious (>20% one-day move) prices, and never plans shorts, margin or sells above holdings. It has a drawdown alert. `configs/holdings.toml` is gitignored (personal account data).
- **D-015 equivalent:** a long-only, unlevered allocation has no per-trade stop. Its risk controls are the allocation itself, the drift band and the drawdown alert. Owner to confirm this is acceptable before any automation beyond planning.
- **Not authorized:** broker API connection, automated order submission, paper trading (Stage 06 gate unchanged).
- **Status:** APPROVED (owner chose option 2). Built and tested; awaiting owner review.
