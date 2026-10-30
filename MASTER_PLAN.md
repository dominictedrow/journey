# ACTIVE SEQUENCE — v10 build-first plan (D-018, approved 2026-10-19)

**This section overrides the stage order further down.** Safety gates are unchanged.

One bot, many strategies. Each strategy is a **mode** behind a single interface, selected in config. A **competition** runs every mode and baseline on the same data, splits, costs and risk rules, then ranks them.

| Phase | What gets built | Reuses old stage file | Gate to leave |
|---|---|---|---|
| B1 Framework | Python package: data loader + validation, session/calendar (ET, DST, early closes), strategy-mode interface, event-driven backtester with costs/slippage/next-bar fills, risk layer that rejects any position without a stop, position state machine, trade ledger, metrics, competition report. Tested on synthetic data. | 02 (design items), 04-02/04-05 | Tests pass; owner reviews demo report |
| B2 Data | Historical MES/ES and EUR/USD bars from an owner-approved source; quality checks; frozen snapshots; untouched holdout reserved | 03 | Data quality report |
| B3 Modes | Baselines (no-trade, always-long/buy-hold, random) + IMOM01, TREND02, REV01, J-T01, J-T02, J-T03 (creator modes labelled `INTERPRETATION`) | 04 | Each mode has unit tests + documented rules |
| B4 Competition | Walk-forward after-cost competition, cost 1×/2×/3×, per-instrument results, ranking; optional SVM/ensemble entrant later | 05 | Owner picks modes to carry forward |
| 06 Paper | Unchanged: broker demo API paper trading, only after approved risk envelope + account verification | 06 | Owner paper/live decision |
| 07 Live / 08 Ops | Unchanged | 07, 08 | Separate written live permission |

- **Stage 01 research** runs in parallel. Its findings tighten mode rules (e.g. IMOM01 MES-only, 16:00 ET exit), but it no longer blocks building.
- **Mode contract:** `name`, `version`, `instruments`, `timeframe`, `params`, `provenance` (`SOURCE` / `ADAPTATION` / `INTERPRETATION`), `on_bar(context) -> intents`. Every entry intent must carry a stop. Modes see only completed bars.
- **Competition rules:** identical data and costs for every mode, chronological splits, a final holdout touched once, all attempted variants logged, ranked on after-cost expectancy, drawdown and Sharpe/Sortino. Ranking never uses win rate alone.

---

# Master execution plan

## Mission
Build a multi-market, continuously monitoring, low-trade-frequency, auditable trading system. Start with **rule-only and linear SVM baselines**, evaluate a non-linear SVM only if needed, consider recurrent models **only if justified by robust out-of-sample evidence**. Creator-derived patterns are hypotheses to audit, encode and ablate. Backtest and paper trade before a separate live-money permission gate. Profitability is not promised.

## Architecture
Market data adapters (futures/forex first; optional crypto) → UTC-normalized immutable raw feed → validated bars/events → deterministic indicator/creator-rule feature engine → time-split ML classification → position state machine (`FLAT`, `LONG`, optionally `SHORT`) → independent risk veto → paper/live execution adapter → event/audit bus → persistent telemetry store → local read-only dashboard/alerts. Dashboard consumers must never directly control order execution. A schedule continuously monitors eligible markets; a NO_TRADE result is normal.

## Stages and sequential gates
| Stage | Detailed instructions | Deliverable | Gate |
|---|---|---|---|
| 00 | `stages/00_requirements.md` | Signed-off scope, risk, constraints | Human approval |
| 01 | `stages/01_research.md` | Evidence-linked video strategy rulebooks | Human sign-off on interpretations |
| 02 | `stages/02_design.md` | Schemas, architecture, provider POC, threat model | Human architecture approval |
| 03 | `stages/03_data.md` | Validated reproducible history and streaming | Quality gate |
| 04 | `stages/04_features.md` | Creator and control indicators with unit tests | Interpretations approved |
| 05 | `stages/05_models_backtests.md` | Baselines, SVM, walk-forward net-cost report; conditional RNN | Statistical evaluation gate |
| 06 | `stages/06_paper.md` | Long-running paper execution and reliability results | Human paper/live decision |
| 07 | `stages/07_live.md` | Optional explicitly approved limited capital launch | Written live permission |
| 08 | `stages/08_operations.md` | Monitoring, incident response, retraining/release process | Operating review |

## Cross-stage acceptance criteria
- Every important strategy rule is traced to a timestamped source or explicitly tagged an unverified interpretation.
- Feature calculations deterministic and causal; data quality tests pass.
- Baselines include buy-and-hold/flat and simple rules, with fees, spread, slippage and sizing.
- Walk-forward and held-out time periods with appropriate gap/embargo; prevent symbol/date leakage.
- Model superiority is demonstrated with net returns, drawdown, turnover, tail risk and stability, not accuracy alone.
- Training vs inference are separate processes. No LLM in per-tick critical path.
- 24/7 service uptime goal does not imply tradability 24/7 or profitable outcomes.
- Runtime enforces risk limits independent of predictions; paper/live adapters are completely separated.
- Each stage ends with an evidence-based go/no-go and updated checkpoint.

## Approved scope versus pending design
Approved: intraday, both long/short, futures/forex priority, crypto candidate, dedicated mini PC, paper-first, later on-screen dashboard, stage-gated agent autonomy.
Pending: exact contracts/pairs, broker/data entitlements, capital/risk limits, market sessions, dashboard controls, acceptance metrics. See `REQUIREMENTS_DRAFT.md` and `ARCHITECTURE_OPTIONS.md`.

## Suggested initial defaults — NOT approved
Python 3.12; containerized Linux service; PostgreSQL or parquet storage depending on scale; scikit-learn LinearSVC + calibrated validation score; 1/5/15-minute bars for discovery; CPU first; limited symbol universe; long/short paper accounts subject to supported broker instruments; 0 live exposure. These are options, not locked specifications.

## Decision policy
Architecture deviations, paid tools, model switch, new strategies, asset classes, shorting/leverage or real orders require the approval protocol in `AGENTS.md` and an entry in `DECISIONS.md`. Each stage file lists further gates.


## 2026-10-18 refinement / explicit architecture gate
- Stage 00 now includes a TradingView feasibility investigation before adopting any execution route. TradingView-native paper simulation and an external autonomous paper-broker gateway are **different systems**; do not conflate Pine strategy emulator fills with placed account orders.
- Stage 01 research encompasses full accessible course catalogs; source registry must list course/edition, license/access, full video runtime, chapter timestamps, transcript status, extracted objective logic, ambiguities, and example charts. Do not invent viewing or strategy claims.
- Stage 02 requires an executable proof-of-concept matrix: TV Pine/alert → authenticated event receiver (signals only); bot → broker paper API; optional TV connected broker through UI (manual). Verify custody of position truth, feed freshness, orders, fill reconciliation and failover under each.
- TradingView stays useful for charts and manual paper trading even if it cannot serve as the unattended executor. Production core uses a broker adapter interface to avoid strategy rewrite when switching from paper to live; migration still needs new testing, cannot be guaranteed seamless.
- Dashboard uses persisted event projections; automatically focus active positions before opportunity-ranked candidates. Test two simultaneous instruments and multiple positions; graphical design later.
- Futures and forex processed concurrently in research/backtests but live readiness can be gated per instrument and provider. Overnight exposure requires gap, margin and funding simulations.

## V4 execution-gate clarification
Stage 00: owner decisions recorded; broker demo API + TradingView charting is accepted. Complete provider feasibility, instrument recommendation, independent stop specification, numerical risk approvals, success metric experiment plan, and private GitHub repository initialization. **Do not mark Stage 00 finished merely because preferences have been supplied.**
Stage 01: coverage inventory of publicly accessible full creator courses, deduplicated material and timestamped rules, including educational vs promotional context. Gate at end for interpretation of ambiguous strategy rules.
Stage 02: create pluggable execution interfaces; isolated demo/live configs; venue/account reconciliation; server-side protective stop acknowledgment; event bus powering eventual dashboard. Design must include futures expiration and FX market sessions.
Stage 03+: require staged demonstrations: recorded replay, deterministic backtests, paper order tests, recovery drills and independently audited exposure. Continuous monitoring is not permission for continuous order submission.
Stage 06: paper execution only after explicit risk-envelope approval and broker-account verification. Stage 07: a *new* live-money approval independent of all previous approvals.

## v8 research addition (authorized within existing Stage 01; no implementation approval)
Stage 01 now includes primary-source institutional research in parallel with creator video analysis. Stage 02 must accommodate multiple independent, plug-in signal engines and a regime/risk overlay; Stage 05 must run apples-to-apples chronological after-cost competitions and treat long-horizon findings as distinct from intraday evidence. See `research/INSTITUTIONAL_STRATEGY_CANDIDATES.md`. The existing stages and risk/authorization gates stay in force.
