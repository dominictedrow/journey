# PORT01: core SPY + diversified trend sleeve (design + pre-registration)

**Status:** pre-registered 2026-11-03 before any PORT01 return was computed. **Backtest only.**
**Owner choices (2026-11-03 chat):** margin account with **≤ 2× gross** exposure; **SPY** as the core; backtest only.
**Goal change (D-030):** stop looking for a standalone edge. Judge a portfolio on risk-adjusted return and drawdown against simply holding SPY.
**Inputs, both fixed earlier:** the core is SPY, the market return every test found. The sleeve is **TSM02 primary exactly as pre-registered** (dc7f575): 24 ETFs, 12-month sign, 40%/σ per asset, equal-weighted. Its later post-hoc observations (commodities, long/flat) are deliberately **not** used.

## Portfolio (rebalanced monthly, last trading day)
- **Core:** 60% of equity in SPY.
- **Sleeve:** TSM02 primary weights `w_i = sign_i × 40%/σ_i / N_live` × **m = 0.5** (a fixed round number; not fitted).
- **Gross cap:** if core + Σ|sleeve weights| > 2.0, scale the sleeve down to fit. The core is never scaled.
- SPY's sleeve position is added to the core weight (netted), not held separately.

## Retail margin accounting (total returns, not excess)
- **Return** = Σ weight × asset total return (Yahoo adjusted close).
- **Cash** = 1 − Σ long weights. Positive cash earns `DFF`. Negative cash (borrowed) pays `DFF + 1.5%`/yr (ASSUMPTION, typical retail margin rate).
- **Short proceeds earn nothing**, and shorts pay a 0.5%/yr borrow fee (ASSUMPTION).
- **Costs:** 5 bp one-way on turnover of target weights (ASSUMPTION); intra-month drift is ignored.

## Comparisons (same accounting, same months)
- **B1:** 100% SPY
- **B2:** 60% SPY / 40% cash
- **B3:** 60% SPY / 40% IEF (classic 60/40)

## Window
Dev = the TSM02 window: 2005-11 → 2025-09 (the first month the sleeve has ≥ 12 assets). Holdout sealed from 2025-10-01. Eras: start–2012, 2013–19, 2020–25.

## Pass criteria (all at 1× costs unless stated)
1. Sharpe ≥ B1 + **0.10**.
2. Max drawdown ≤ **0.75 ×** B1's max drawdown.
3. Sharpe ≥ B3's Sharpe (it beats the classic 60/40).
4. At **3×** costs, borrow and margin spread, Sharpe still ≥ B1.
5. In **every** era, Sharpe ≥ B1 − 0.10 (no era where it's much worse than holding).

Passing would make PORT01 a candidate for the one-shot holdout check and, separately, a Stage 06 paper-trading plan; both need owner approval. Failing ends the strategy search with the documented finding that nothing here beats holding SPY on a risk-adjusted basis.

## Multiple testing
This is the 8th test. There's one portfolio (no weight search); m, the 60% core and the 2× cap are fixed round numbers, and the benchmarks are standard.
