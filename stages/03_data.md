# Stage 03 — historical and streaming data integrity
> **v10 (D-018):** This is now **Phase B2 Data**, after the framework is built. Streaming (03-04) is deferred to just before Stage 06.

- [ ] 03-01 Build approved historical data adapters, schema validation and append-only raw storage; save lineage and entitlements.
- [ ] 03-02 Normalize units/timezones/market sessions; detect duplicate, out-of-order, missing/stale and malformed data.
- [ ] 03-03 Implement point-in-time symbol membership, survivorship adjustments, splits/dividends or rollover handling as appropriate.
- [ ] 03-04 Implement streaming with reconnect/backoff, backfill, heartbeats and measured latency.
- [ ] 03-05 Unit and integration tests: causal ordering, clock skew, feed outage, stale-price no-trade and cross-market synchronization.
- [ ] 03-06 Freeze versioned research snapshots; produce data quality report.

**Done when:** end-to-end historical and mock real-time feed reliable, reproducible and measurable; gate quality sign-off.
