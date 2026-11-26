# Decisions Log

This log is append-only. Never edit a past entry. To change a decision, add a new entry that supersedes it.
**Status** is one of: `accepted`, `proposed` (waiting for Dom), or `superseded by D-###`.

| ID | Date | Decision | Status | Approver | Evidence |
|---|---|---|---|---|---|
| D-001 | 2026-11-25 | FRED truncated `BAMLH0A0HYM2` to about 3 years (it starts 2023-11-27, and ALFRED is truncated too). Use `BAA10Y` as the long-history credit-stress proxy for training and the HMM. Archive the ICE series daily from day one. The rules for switching between the two are settled in Phase 3. | accepted | Claude (research), Dom to confirm in Phase 3 | FRED series page note; live probe 2026-11-25 |
| D-002 | 2026-11-25 | FINRA retired the PDT rule on 2026-06-04. The spec's "never sell same session" rule is now a design choice, not a legal limit, so it becomes the config flag `rules.block_same_session_round_trip` (default `true`). | proposed | Dom (Phase 5) | Alpaca blog and changelog 2026-06-03 |
| D-003 | 2026-11-25 | Alpaca fractional/notional orders are DAY-only, so GTC broker-side stops can't cover fractional positions. Choose between whole shares only and re-arming stops every day. | proposed | Dom (Phase 6) | Alpaca orders docs |
| D-004 | 2026-11-25 | Alpaca crypto supports only market, limit and stop_limit orders, with gtc and ioc time-in-force, and no trailing stops. Crypto trailing logic lives in the executor and moves the stop_limit each cycle. | accepted | Claude (research) | Alpaca crypto docs |
| D-005 | 2026-11-25 | Data source priorities. Primary: Alpaca prices, FRED `VIXCLS`, EDGAR fundamentals, Nasdaq earnings calendar. Fallback: yfinance (fragile, unofficial) and Finnhub free. | accepted | Claude (research) | yfinance 429 and v7 endpoint issues |
| D-006 | 2026-11-25 | Coinglass is paid, so it is replaced. Funding comes from OKX and Hyperliquid public endpoints, and open interest from Deribit and Kraken Futures, aggregated in-house. Binance and Bybit are geo-blocked. | accepted | **Dom** | Coinglass pricing page; live probes |
| D-007 | 2026-11-25 | BTC ETF flows: try the SoSoValue API in Phase 2. If its key isn't free, drop ETF flows from v1. Farside scraping is rejected as too fragile. | accepted | **Dom** | Farside returns 403 to scripts |
| D-008 | 2026-11-25 | News sources: Cointelegraph and Decrypt (crypto); CNBC Markets, BBC Business and NPR Economy (economics). | accepted | **Dom** | RSS probes 2026-11-25 |
| D-009 | 2026-11-25 | Stack: Python 3.13 + uv; Docker Compose TimescaleDB (same on Mac and Windows); alpaca-py; APScheduler 3.x with XNYS gating; scikit-learn; hmmlearn pinned (it is in limited maintenance); an in-house 64-bit SimHash (the `simhash` package is stale); PyTorch `cu130` on Windows. | accepted | Claude (research), Dom may revisit in Phase 1 | Research 2026-11-25 |
| D-010 | 2026-11-25 | DOMS_DRIVE is exFAT. The venv lives at `~/.venvs/quant_trader` (`UV_PROJECT_ENVIRONMENT`), and the Postgres data sits in a Docker named volume, never on the drive. | accepted | Claude | vibe/trader precedent |
| D-011 | 2026-11-25 | Alpaca paper fills have no slippage model. Price-transformer training data from paper fills is therefore optimistic, and its experiments must say so. | accepted | Claude (research) | Alpaca paper docs |
