# Questions for the project owner

**2026-10-18 update:** Q1 broad asset class, Q3 directions, Q4 horizon, Q8 deployment path, Q12 budget flexibility, Q14 creator identities, Q27 mini PC, Q29 autonomy style and Q30 dashboard preference are partly/fully resolved. Exact provider, instruments, risk parameters and quantitative targets remain open. No assumed numerical settings.

Answer what you can. Unanswered decisions stay OPEN and must not be silently guessed when material.

## Markets and behavior
1. Which exact markets? US stocks/ETFs, futures, forex, spot crypto, options? Which symbols or categories first?
2. Does 'multiple markets' mean compare signals across markets or actually trade each market?
3. Long-only or short too? Spot versus derivatives? Is leverage prohibited or allowed?
4. Intended holding period: seconds, minutes, hours, overnight, days?
5. Maximum desired trades per day/week; how many simultaneous positions?
6. Monitor 24/7 or actually trade 24/7 where available? Is weekend crypto desired?
7. Order style: market, limit, stop; is fully autonomous entry and exit essential?

## Money and controls
8. Is this initially research/paper only, or eventual real money? With what minimum paper-test duration?
9. Approximate eventual bankroll (range is enough), currencies, brokerage country/jurisdiction?
10. Maximum loss per trade, max daily/weekly loss and max peak-to-trough drawdown?
11. Whether a stop-loss is mandatory even if model keeps predicting BUY?
12. Budget for feeds, cloud, LLM/video analysis and data storage per month?
13. Are broker account and API chosen? Which platforms have you used?

## Strategy research
14. Confirm exact Joovier and Swaggy C accounts/links; any priority videos/playlists?
15. Do you own/access any private course material? If yes, can authorized notes/transcripts be supplied?
16. Does 'integrate pattern recognition' mean reproduce deterministic rules, label examples for ML, or both?
17. Which terms/features matter to you (liquidity sweeps, market structure, order blocks, trend, volume, session windows, etc.)? Do not assume creators use any unverified item.
18. Do you want screenshots and chart examples documented alongside timestamps?
19. Do we have a target number of videos, or a research cutoff based on saturation and rule completeness?

## ML and evaluation
20. What specifically should classifier predict: next-bar direction, horizon return after costs, or actionable entry quality?
21. Is the proposed BUY logit intended as an open/close signal, or position permission? What should happen on neutral/conflicting signals?
22. Optimize expected dollars, return on equity, Sharpe/Sortino, drawdown, or win rate with constraints?
23. Is deterministic low-latency more important than ML complexity? Expected inference deadline?
24. Historical testing window and what regimes should be required (high volatility, bear, sideways)?
25. What constitutes success sufficient to spend months paper trading? What is failure/stop rule?

## Environment/workflow
26. Claude Code CLI, Claude Desktop, VS Code, or another environment? Does the coding agent have repo and terminal access?
27. Mac local development or Linux remote server? Dedicated always-on machine/cloud budget?
28. GitHub repo existing? Private/public, and are you the only reviewer?
29. Should Claude request approval for every merge/push, or only unexpected changes and stage gates?
30. Do you want Telegram/Discord/email alerts, a dashboard, or both? What should trigger alerts?
31. Do you want models retrained on a schedule or only after human evaluation?
32. Are trades to be run in your own personal account only, or will the system be used for others?


## Updated response status (2026-10-18 second round)
Resolved in principle: both futures and forex in initial research; long/short intraday; Mac + VS Code Claude Code; future mini PC purchase; adaptive volatility sizing; conditional overnight positions; active-position-first dashboard; multi-metric evaluation before threshold selection; investigate complete authorized creator courses. Capital undecided intentionally.

### Remaining decisions that genuinely block architecture or safe execution
1. **TradingView usage:** Is the built-in TradingView paper account a strict requirement for *automated* execution, or is TradingView charting + a broker's paper API acceptable? The former is unsupported by official direct Pine trading integration.
2. **Paper broker/API:** Are there existing TradingView or broker accounts? Which country/jurisdiction will the trading account be registered in (affects futures/FX eligibility, demo APIs and shorting)?
3. **Pilot symbols:** Choose initial *small* futures + forex universe (e.g., one micro index futures contract and one major FX pair), or explicitly authorize the agent to recommend using data/commission evidence.
4. **Hard safety envelope:** Regardless of future capital, define what happens with loss limit, exposure cap, stop-loss policy, maximum concurrent positions, overnight position reduction and emergency kill switch. Exact capital-sized numbers may remain pending for simulation, but no live approval without them.
5. **Creator course access:** Which full courses/editions do you legally have access to, can you supply recordings/transcripts or course URLs, and may the coding agent access them? Public playlists can be used without private access.
6. **TradingView subscription/data permissions:** Current plan and whether you have real-time exchange market data and alert/webhook entitlements. Market-data costs and provider restrictions must be checked.
7. **Trading automation responsibility:** Are you willing to accept broker-paper execution independent of TradingView-connected broker UI, with TradingView alerts only as an optional input/visual overlay?
8. **Git remote:** GitHub repo already created or should agent initialize private repo? No pushes until remote and authorization validated.
9. **Mini PC procurement:** Budget range, display resolution and power/network constraints before buying; initial benchmark can run on Mac.
10. **Notifications:** What failures demand immediate alerts (disconnections, stale quotes, overnight holds, order rejection, daily loss cap)? Defaults can be proposed at design stage.

## V4 decision state — do not ask resolved questions again
Resolved: **broker demo API is acceptable** alongside TradingView charting; user **authorizes recommendations** for small futures/FX universe; use **public complete courses**; user has **TradingView Essential**; every position **must carry an independent protective stop**; new **private GitHub repository** to be created when authorized.

### Only material outstanding approvals
1. **Eligibility & permissions:** Which country/jurisdiction will the broker accounts be registered in? (Necessary to verify API eligibility.) Provider selection and whether a funded account/paid API entitlement is acceptable for futures demo API.
2. **Pilot instrument approval:** Approve or reject recommended MES + EUR/USD *after* checking provider-specific liquidity/data costs; candidates aren't committed choices.
3. **Risk contract:** exact paper-account notional, per-position maximum risk, daily loss and drawdown breaker, portfolio gross exposure and max concurrent positions; overnight risk, stop gap/slippage and market-reopen handling. Agent drafts candidate defaults; owner approves before Stage 06. Live caps set afresh at Stage 07.
4. **Provider/data cost authorization:** which paid demo API subscription and exchange data entitlements (if any) are acceptable once validated; do not purchase automatically.
5. **Strategy source completeness:** establish accessible public course corpus and whether creator rules are sufficiently operational to encode; source gaps are *research findings*, not user blockers until stage 01 review.
6. **Project account authorization:** repo name/owner, GitHub credentials/CLI auth, remote-create operation, push policy and human reviewer. Repository creation is not yet complete.
7. **Stage 00 sign-off:** owner approves scope, paper-vs-live safety isolation, baseline evaluation protocol and success gates before Stage 01.

Deferred legitimately: capital amount (undecided); hardware purchase SKU; cosmetic dashboard design; crypto onboarding; model promotion thresholds (metrics first); retraining schedule.
