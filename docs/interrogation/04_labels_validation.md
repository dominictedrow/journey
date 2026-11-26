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
8. **"All models should use k-fold training" (Dom, `docs/spec/model_descriptions_v1.md`):** CLAUDE.md bans random splits. Which reading do you mean? (A) K sequential walk-forward folds with purge + embargo; these build the out-of-fold set for the meta learner and the final evaluation. (B) (A) plus combinatorial purged CV (CPCV) for model selection and overfitting checks only. CPCV folds are contiguous and purged, but some train on later data. (C) Shuffled k-fold, which would break the hard rule. Research recommends (B). See `docs/plan/MODEL_PLAN.md` §6.1.
