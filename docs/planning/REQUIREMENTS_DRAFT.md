# Requirements draft v0.2 — awaiting sign-off

## Confirmed by owner 2026-10-18
- Multi-asset design, initially futures and forex; crypto optional depending on research.
- Intraday positions (minutes to hours); long and short; continuous market monitoring but selective entries.
- Paper account before any separately authorized live trading.
- Always-on autonomous mini PC; connected screen with charts that change based on selected/active instruments, portfolio state, per-trade and total P&L, and operational stats.
- Prioritize execution speed/efficiency and realized after-cost performance; SVM first, RNN only if justified.
- Agent executes within approved stage, pauses at stage gates or significant deviations; disciplined branches/commits/pushes.
- Budget flexible when justified with measured benefit and owner approval.
- Creator identity references (not assigned for strategy extraction): Swaggy C https://www.youtube.com/watch?v=2ncnCT0Eoiw ; Joovier https://www.youtube.com/watch?v=XWNi136ybAE .

## System separation
- Development/research agent: reads checkpoint and stage files; researches strategy; trains/evaluates; deploys only authorized signed releases.
- Unattended runtime: deterministic pipeline for ingest/features/inference/risk/execution/reconciliation; it does not require Claude availability or API credentials.
- Monitoring: event telemetry API, append-only audit records, read-only local dashboard; disable execution if risk controls cannot evaluate.
- Training and video analysis do not run on the trading critical path.

## Candidate acceptance tests (numbers not approved)
- No live orders in default configuration; mode is explicitly PAPER with separate credentials/ endpoints.
- On reboot, recovers persisted state and reconciles orders/positions before fresh entries.
- No duplicate orders after retry, reconnect, or process restart.
- Stale quotes, feed outages, margin uncertainty or clock errors block new entries.
- Data model supports exchange/receipt timestamps, symbol, asset class, currency, futures expiry, contract multiplier, and broker order IDs.
- Dashboard shows instrument, candlesticks, annotated signal, status, net liquidation, realized/unrealized P&L, orders/fills and health indicators; charts render from shared immutable events rather than strategy globals.
- Research reports net fees/spreads/slippage, out-of-sample drawdown, turnover, uncertainty, and comparison to non-ML baselines.

## Explicitly unresolved
Pilot instruments; brokerage(s)/API/data subscriptions; applicable region and derivatives eligibility; mini-PC OS and network; position sizing, leverage, max daily loss; overnight holdings and weekend crypto; rollout metrics and live trading authority; screen size/dashboard operator permissions.


## Owner decisions: scoping round 2 (2026-10-18)
- First research universe covers **both futures AND forex**, not an either/or pilot. Select limited representative contracts/pairs as an implementation decision after examining data cost/availability.
- Adaptive position sizing from observed market volatility is desired; use contract multiplier, stop distance, margin requirements, currency conversion, volatility state and independent maximum-loss constraints. No sizing without risk caps.
- Overnight holding may be allowed per-strategy when supported, but session, gap, rollover, weekend/holiday financing, swap and liquidation exposure must be modeled and approved.
- Dashboard screen prioritizes **active positions**, then actionable signals/watchlist; telemetry/event contracts designed early; graphical UI may be late.
- Development host: Mac + VS Code + Claude Code; production host: to-be-purchased mini PC (Linux x86-64 candidate, not yet approved hardware).
- Compare performance metrics before setting quantitative promotion thresholds; still define minimum safety gates in Stage 00.
- TradingView Paper Trading first is a **user preference pending technical validation**, not a supported autonomous execution API. Test TradingView charts/Pine/alerts and broker demo API separately, preserving one strategy/state/risk core.
- Analyze the **full accessible strategy courses** from both creators, recording chapters/timestamps and verifying whether each source actually contains codifiable trading rules. Course materials require legitimate access; no claims of having watched unseen videos.

## V4 binding additions (2026-10-18)
- Separate UI from execution: TradingView Essential used for charting; autonomous paper orders through broker demo API; live endpoint entirely isolated, disabled and requires Stage 07 permission.
- Start with **both futures and forex**, but prioritize a deliberately small representative universe. Suggested study candidates: MES (Micro E-mini S&P 500) and EUR/USD, subject to market data licensing, demo API availability, contract mechanics, and owner approval. Alternatives such as MNQ and GBP/USD only after cost/liquidity analysis.
- Every executed position must have confirmed broker-side protective stop. If the venue lacks native attached/bracket or reliable protective stop capability, fail closed and reject that provider/order mode until an approved protection plan exists. Account for partial fills, gaps, outages, overnight margins, and futures expirations.
- Volatility-based sizing (ATR or realized-volatility candidate), with separately configurable hard contract/unit exposure ceilings and max drawdown/loss breakers. **All numerical limits TBD and require owner approval before any paper order is enabled.**
- Public complete creator course research: official identity evidence, authorized publicly accessible sources, URL and publication date, episode coverage, timestamps, concrete entry/exit/invalidations, contradictions, confidence levels and strategy reproduction tests. No claim of complete viewing without actual access.
- Private GitHub repo not yet created. Mac VS Code + Claude Code for development; Linux mini-PC deployment after profiling.
- Dashboard telemetry contract begins in design: open positions first, current chart, order/fill states, stop state, realized/unrealized P&L, equity/drawdown, model and data-health alerts.
