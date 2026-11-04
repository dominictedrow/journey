# Stage 02 — architecture, schemas and provider proof of concept
> **v10 (D-018):** Design items here are folded into **Phase B1 Framework** in `MASTER_PLAN.md`. The broker/provider POC (02-01, 02-08) and mini-PC bench (02-07) move to just before Stage 06.

- [ ] 02-01 Decide exchange/broker/data adapters by instruments, entitlement, pricing, history and supported paper/live endpoints.
- [ ] 02-02 Design time-aligned, UTC-anchored multi-symbol schema: source, exchange timestamp, receipt timestamp, currency, corporate actions, bid/ask/trade/OHLCV, quality flags.
- [ ] 02-03 Specify actor boundaries: ingestion, immutable storage, causal feature construction, inference, strategy policy, risk veto, execution, reconciliation, monitoring.
- [ ] 02-04 Define `FLAT/LONG/(SHORT)` state machine: buy/hold/sell distinction, hysteresis thresholds, minimum hold/cooldown, neutral state, hard stops overriding ML, end-of-session behavior.
- [ ] 02-05 Document failure modes: stale/missing ticks, feeds down, partial fills, duplicated events, API outages, clock drift, exchange halts, market closure, price gaps.
- [ ] 02-06 Define durable event schema, backend telemetry, screen-connected kiosk dashboard, chart rotation and operator-safe UI API; implement dashboard skeleton but defer visual iterations.
- [ ] 02-07 Bench mini-PC CPU/memory/thermal/UPS/network with simultaneous scanning and local rendering.
- [ ] 02-08 Deliver thin read-only provider POC and architecture decision record; explicit owner approval before full implementation.

**Done when:** schema, tests, budgets and adapter architecture approved. Next 03.
