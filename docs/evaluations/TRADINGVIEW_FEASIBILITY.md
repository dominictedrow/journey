# TradingView integration feasibility — decision gate

## Verified platform constraints (official documentation, accessed 2026-10-18)

- TradingView **Paper Trading** provides simulated manual orders and positions over multiple asset classes: https://www.tradingview.com/support/solutions/43000516466-paper-trading-main-functionality/
- TradingView says Pine Script strategies **cannot place orders in its built-in Paper Trading account or connected Trading Panel brokers**. Pine Strategy Tester and alerts use a broker emulator, not live broker execution: https://www.tradingview.com/pine-script-docs/faq/strategies/
- TradingView explicitly says automated strategy trading with a brokerage account is not available natively: https://www.tradingview.com/support/solutions/43000481026-how-to-autotrade-using-pine-script-strategies/
- TradingView webhooks send HTTP POST to external applications but may fail/delay; only 80/443 ports, 3-second receiver deadline, no credentials in payload, two-factor authentication required: https://www.tradingview.com/support/solutions/43000529348-how-to-configure-webhook-alerts/
- TradingView explicitly warns that alerts are **not designed for automated trading**: https://www.tradingview.com/support/solutions/43000722015-using-credentials-for-webhooks/
- TradingView broker connections permit **manual order placement** through chart/Trading Panel, not automatic Pine-to-broker execution: https://www.tradingview.com/support/solutions/43000784715-finding-your-perfect-broker-on-tradingview/
- TradingView real-time displayed quote packages and broker order-panel quotes can differ in delay: https://www.tradingview.com/support/solutions/43000739323-how-does-the-source-of-real-time-data-affect-the-trading-experience/

## Alternatives (not yet approved)
A. **Recommended**: TradingView for research, charting and discretionary paper validation; **broker's official paper/demo API** for unattended orders. Own shared signals/risk/event schema, independent of chart UI. Supports automated paper-to-live migration through adapters, subject to broker qualification and tests.
B. TradingView Pine alerts → authenticated webhook receiver → validated, idempotent broker paper API. Treat alerts as untrusted advisory events; do not execute blindly, protect from duplicates/outages, independently verify price, risk, broker account, fill and state; evaluate delay/failure rates. Must check platform terms and security.
C. Pine Strategy Tester only + TradingView built-in manual Paper Trading. Useful for discretionary setup testing, **does not satisfy unattended paper trading**.

## Required POC acceptance before selection
- Chart data entitlement and quote delay; deterministic indicator parity Pine vs Python; webhook delivery latency/loss/retries; no order duplication; broker demo order submission/ack/fill/cancel; bidirectional state reconciliation; FX and futures permissions; overnight/restart recovery; crash, stale price and partial-fill behavior; audit logs and manual emergency disable.
- Do **not** build a browser-clicking robot to automate order entry through TradingView UI or promise that adding a linked broker makes the autonomous execution pipeline work automatically.
- Default state remains PAPER_DISABLED until explicitly authorized; LIVE_DISABLED until separately approved.

## Open decision
Owner must decide whether **native TradingView Paper Trading account** is essential or whether TradingView charting + **broker paper API** meets the intent. No broker/vendor commitment before POC.

## V4 owner resolution
**Choice approved:** TradingView charting and optional Pine/alerts; independent broker demo API executes automatic paper orders. Treat TradingView as *non-authoritative* for fills, positions, account equity, margin or protective stops. Never scrape/click TradingView to simulate unsupported brokerage API functionality. Essential subscription does not establish real-time CME exchange data access. Alerts are optional and lossy: deduplicate signals, validate signatures/allowlists as applicable, monitor delivery, and always enforce broker-side risk. See `docs/evaluations/PROVIDER_EVALUATION.md`.
