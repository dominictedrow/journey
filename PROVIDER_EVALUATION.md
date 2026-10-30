# Provider and market-data feasibility — v5 (2026-10-18 PDT)

**Recommendation, not activation:** OANDA v20 Practice for EUR/USD forex simulation and a deterministic local MES futures simulator until a compliant, affordable futures API/data arrangement is approved. Preserve an execution-adapter interface; TradingView Essential is charting only. Do not create accounts, purchase access, or transmit orders.

| Candidate | What official docs establish | Outstanding evidence / blocker | Decision |
|---|---|---|---|
| OANDA v20 Practice | Independent practice REST (`api-fxpractice.oanda.com`) and pricing stream (`stream-fxpractice.oanda.com`); order management and historical prices documented | US account eligibility, available EUR_USD instrument, actual price/depth/data cadence, stop-loss order semantics, financing and effective costs, practice-vs-live differences, data use terms | Preferred FX feasibility spike; not yet connected |
| Tradovate | `demo.tradovateapi.com` supported; API key prerequisite of live funded account >$1,000 and $25/month API add-on. API add-on does *not* include real-time API market data; CME ILA/sub-vendor requirements may apply to API streaming. | Need provider-written confirmation of market data licensing, entitlement, total costs, demo access and Linux compatibility before purchase | Conditional futures candidate; do not fund or subscribe |
| Internal deterministic MES simulator | Can use validated, licensed historical data in local backtest / replay; no broker account required | Never interpret local fills as proof of broker demo execution quality or exchange realism; independently source authorized price data | Recommended interim MES engine |

## Pilot research instruments
1. MES (CME Micro E-mini S&P 500), front-month contract **by explicit contract month**, multiplier $5/index point, outright minimum price step 0.25 index point = $1.25/tick. Must model rollover, market sessions, commissions, exchange fees, slippage, stop gaps, margin and position liquidation. CME contract calendar applies.
2. EUR/USD (OANDA instrument identifier often `EUR_USD`; confirm with account instrument listing). Model bid/ask, variable spread, financing and weekend closure. No inference of broker permissions from public documentation.
3. MNQ and GBP/USD: comparative research only, do not enable in pilot without evidence and owner approval.

## Cost decision
- Tradovate API access as of Sep 16, 2026: **$25/month**, contingent on funded live account balance **at least $1,000**. Funding is not a fee but exposes capital and requires consent. Real-time market data through the API is **not included**. Exchange/clearing/NFA charges may apply. Do not quote a reliable all-in budget absent entitlement confirmation.
- OANDA practice API costs/terms and data rights should be confirmed with support and account documentation, not presumed free for all uses.
- Prefer no financial commitment during Stage 01 research.

## Exact official sources
- Tradovate updated API access: https://support.ninjatrader.com/s/article/Tradovate-API-Access?language=en_US
- Tradovate API hosts and requirements: https://api.tradovate.com/
- Tradovate price/fees: https://tradovate.com/pricing/
- OANDA practice and live API endpoints: https://developer.oanda.com/rest-live-v20/development-guide/
- OANDA API capabilities: https://developer.oanda.com/rest-live-v20/introduction/
- CME MES specifications: https://www.cmegroup.com/articles/faqs/frequently-asked-questions-micro-e-mini-equity-index-futures.html
- CME hours and expiry: https://www.cmegroup.com/education/courses/micro-e-mini-futures/micro-e-mini-futures-products-overview
- TradingView webhook limitations: https://www.tradingview.com/support/solutions/43000529348-how-to-configure-webhook-alerts/
