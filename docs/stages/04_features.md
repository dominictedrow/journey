# Stage 04 — deterministic pattern features and strategy harness
> **v10 (D-018):** This is now **Phase B3 Modes**. Each strategy is built as a self-contained mode. An owner-approved rulebook is no longer a prerequisite; unverified creator logic is labelled `INTERPRETATION`.

- [ ] 04-01 Translate approved creator rulebooks into exact indicator contracts (inputs, outputs, windows, warm-up, invalidations); keep creator-specific provenance.
- [ ] 04-02 Compute vectorized indicators outside Claude/model inference loop; cache and incrementally update where correct.
- [ ] 04-03 Create baseline features: returns, volatility, range, trend, volume/liquidity, session and optionally cross-market context; all causal.
- [ ] 04-04 Test known annotated examples/counterexamples against code; annotate disagreement instead of silently changing strategy meaning.
- [ ] 04-05 Benchmark latency per symbol and incremental recomputation, monitor compute budget.
- [ ] 04-06 Perform feature leakage audit and present feature inventory for approval.

**Done when:** features are documented, deterministic, versioned, causal and approved.
