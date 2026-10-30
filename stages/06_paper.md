# Stage 06 — continuous paper trading and resilience
- [ ] 06-01 Human authorizes paper endpoint access, symbols and risk caps; guardrail prevents real endpoint execution.
- [ ] 06-02 Deploy a supervised 24/7 *monitor* with market calendars; alerts and dead-man's switch; NO TRADE when feed/broker is unavailable.
- [ ] 06-03 Implement order idempotency, state reconciliation, partial fill, cancel/replace, exchange reject and retry tests.
- [ ] 06-04 Collect forward out-of-sample trades and missed setups for owner-approved period; compare fills and backtest assumptions.
- [ ] 06-05 Incident drills: crash/restart, incorrect clock, stale feed, API timeouts, position discrepancy, kill switch.
- [ ] 06-06 Score objective gates against predeclared limits, not anecdotal winning streaks. Review drift and latency.

**Done when:** approved paper period completed with reliable evidence; owner determines stop/iterate or requests separate live gate.

- [ ] Verify dashboard remains informational during paper operation and the trading engine continues safely if display/browser crashes.
- [ ] Exercise unattended mini-PC reboot, secure credential loading, network failure and broker position reconciliation.
