# Architecture recommendation and provider tradeoffs (research snapshot 2026-10-18)

## Recommended pattern
Linux x86-64 mini PC (provisionally 16–32 GB RAM, NVMe, wired network and UPS); Python ingestion/feature/inference services; local PostgreSQL/Parquet according to measured load; broker-specific adapter; Docker Compose and systemd restart policies; FastAPI read-only telemetry endpoint; browser dashboard displayed in kiosk mode. Headless operation must continue when the monitor fails. A second machine can train models and inspect videos, then deliver versioned artifacts after approval. No requirement for an inference GPU in MVP.

## Asset-provider options requiring a proof of concept
- Futures: NinjaTrader Trader API is a candidate with REST/websocket market data and automated order interfaces; validate API product/account availability, entitlements, production support, and simulation. https://developer.ninjatrader.com/products/api . Another candidate is Interactive Brokers TWS API; validate Gateway/session operational constraints and API paper behavior. https://www.interactivebrokers.com/docs/tws-api/contracts/futures ; https://www.interactivebrokers.com/docs/tws-api/doc/md/introduction . IBKR continuous futures are historical-only, not live/orderable; explicit contract roll mapping is mandatory.
- Spot forex: OANDA v20 is a practice/live candidate with dedicated streaming endpoints. Verify account jurisdiction and tradeable instrument permissions. Its price stream can coalesce data (up to 4 prices per second/instrument): suitable to test minutes-scale, not full tick reconstruction. https://developer.oanda.com/rest-live-v20/development-guide/ ; https://developer.oanda.com/rest-live-v20/pricing-ep/ .
- Crypto: broker/exchange/venue to be selected only after confirming compliant long/short products in user's jurisdiction. Alpaca spot crypto is not shortable and cannot alone satisfy crypto long/short requirement. https://docs.alpaca.markets/us/docs/crypto-trading-1 .

## Architecture boundaries
1. Exchange calendar/market sessions with time zones and DST, plus futures expiry/roll and FX weekend gaps.
2. Data ingestion -> normalized event schema -> causal feature engine -> model/thresholds -> portfolio allocator -> independent exposure/margin/risk veto -> broker adapter -> fill reconciler -> immutable ledger.
3. Separate telemetry subscriber -> read-only chart/dashboard; do not let rendering stall ingestion or order processing.
4. Baselines: no-trade, fixed deterministic rule, simple directional benchmark; SVM plus probability calibration if warranted. Investigate RNN only if better net out-of-sample performance despite training and maintenance costs.
5. Long/short transitions need flat/close handling, reduce-only where available, pre-trade risk checks, stops, pending-order and partial-fill states; classifier output is not an order.

## Hardware and reliability POC checklist
- Mini PC idle/inference CPU, peak RAM, disk write endurance, temperature/throttling, network uptime, restart/reconciliation, time sync and watchdog.
- Disconnect display during simulated operation and verify trading service unaffected.
- Rehearse network outage, broker loss, partial fill and delayed market events; confirm safe recovery.

## Sources
See `docs/evaluations/SOURCES.md`; verify current terms and region-specific eligibility before payment or account setup.
