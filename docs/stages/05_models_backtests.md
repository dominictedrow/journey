# Stage 05 — baseline ML, rigorous validation and conditional RNN
> **v10 (D-018):** This is now **Phase B4 Competition**. Deterministic modes compete first. The SVM (05-03/05-04) becomes an optional competition entrant; the RNN permission gate (05-08) is unchanged.

- [ ] 05-01 Define prediction horizon/label and transaction-cost-aware tradability; position state is NOT equivalent to the label.
- [ ] 05-02 Establish baselines: flat/no trade, buy-and-hold appropriate to asset, simple rule-only systems, majority/random and no-creator-features controls.
- [ ] 05-03 Train standardized linear SVM on chronological folds; consider kernel SVM only if compute/data support it. Calibrate decision scores on isolated calibration periods if probabilities are needed; never call a raw decision score a probability/logit.
- [ ] 05-04 Add creator feature groups incrementally; ablate Joovier-only, Swaggy-only, both and neither to test marginal predictive value.
- [ ] 05-05 Backtest an explicit state machine and order model; realistic spreads, fees, slippage, fill uncertainty, latency, trade size and market hours.
- [ ] 05-06 Walk-forward testing with training-only preprocessing, purge/embargo if labels overlap, no future data in features; final untouched holdout.
- [ ] 05-07 Report net return, expectancy, profit factor, win rate, drawdown, tail loss, turnover, trade count, risk-adjusted return, calibration, regime/symbol stability, sensitivity to trading costs and inference latency.
- [ ] 05-08 If SVM fails predetermined gates, first diagnose features, labels, costs, data and risk rules; **request permission** before RNN/GRU/LSTM experiment. Compare under the same splits and resources; account for additional compute and overfit risk.
- [ ] 05-09 Formal go/no-go report. No cherry-picking or test-set-driven retuning without new holdout.

**Done when:** results are reproducible and survive out-of-sample net-cost risk gates; otherwise NO LIVE TRADE and re-plan with approval.
