# Stage 08 — operations and controlled learning
- [ ] 08-01 Alert on missing market data, stale predictions, unexpected positions, drawdowns, drift, recurring errors and expired credentials.
- [ ] 08-02 Dashboard for candidate setups, reasons rejected, exposures, orders/fills, latency and net P&L.
- [ ] 08-03 Release protocol: freeze model/feature/config version, shadow-test, paper A/B, approval before any live promotion.
- [ ] 08-04 Backups, reproducible disaster recovery, audit log retention, key rotation, incident review and maintenance window.
- [ ] 08-05 Periodic performance/overfitting audits; retire model on threshold failure rather than assuming profits continue.

**Done when:** documented controlled operations. Permanent ongoing review; no unattended architectural self-modification.

- [ ] Verify dashboard remains informational during paper operation and the trading engine continues safely if display/browser crashes.
- [ ] Exercise unattended mini-PC reboot, secure credential loading, network failure and broker position reconciliation.
