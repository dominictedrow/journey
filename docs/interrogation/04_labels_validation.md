# Questioning: Phase 4 · Labels + validation harness

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/04_labels_validation.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Labels, Validation

1. **Triple barrier:** keep 1.5σ√h symmetric barriers, or let them differ per asset class?
2. **Crypto altcoin benchmark:** compare altcoins against BTC (spec). Confirm BTC, ETH and SOL are benchmarked against cash.
3. **Walk-forward start:** when does the first training window end (e.g. 3 years minimum)? Should test blocks be quarterly?
4. **Final holdout:** seal the last 12 months and never touch them until the go-live decision (the vibe/trader pattern)?
5. **Costs:** 10 bps per side for equities and 30 bps per side for crypto (Alpaca taker is 25 bps). Also run 2× and 3× cost stress tests?
6. **IC significance:** use a Newey-West t-stat, because overlapping 5-day labels are autocorrelated?
7. **Metric report:** what goes in each fold's report, and in what format (Markdown plus CSV in `reports/`)?
