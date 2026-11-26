# Questioning: Phase 10 · Meta learner (10a) + Market Transformer (10b)

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/10_meta_market.md`, covering interfaces, config keys, tests and acceptance criteria.
> Sources: `docs/spec/model_descriptions_v1.md` (Meta Learner, Market Transformer), spec "Model contracts", "Execution layer", `docs/plan/MODEL_PLAN.md` §6.
> Out of scope: the rules engine (Phase 5) and the executor (Phase 6). They stay as specified.

## 10a · Meta learner
1. **Inputs:** "all 4 votes, all data from the LLM output, and raw numbers". Keep the spec's extra inputs too: regime probabilities, and each asset's current portfolio weight?
2. **"12 layers deep of paged attention, 8 heads":** paged attention only manages a KV cache during token generation, and the meta learner scores in one forward pass, so there is nothing to page. Proposal: 12 standard attention layers (SDPA), with attention **across the day's assets** so each score sees the others. Keep the paged KV cache in the LLM only. OK?
3. **Output dim 500, one slot per S&P 500 stock:** membership changes over time, so slot *i* would mean different companies in different years, new names have no trained slot, and history would be survivorship-biased. Research recommends a **shared-weight scorer**: the same network scores every asset present that day (any count, crypto included), which also matches the spec. Pick one:
   - (A) a shared scorer over the day's universe, masked to the weekly screened list afterward;
   - (B) a fixed 500-slot head as written, with slots keyed to point-in-time membership.
4. **Output meaning and loss:** cross-entropy means a class per asset (bullish vs bearish, or 3-class). The spec wanted score *s* (target: excess return / σ√h) plus conviction *c* = calibrated P(up), which the rules engine's thresholds (c ≥ 0.65) depend on. Pick one:
   - (A) CE vote only, with *c* = the calibrated probability and *s* = the logit;
   - (B) CE for *c* plus a regression head for *s*.
5. **Screening:** score the whole universe, then keep only the names on that week's screened list (~80 names, D-025). Do crypto coins bypass the stock screen?
6. **Activation + init:** ReLU in the FFN parts with Kaiming init, the default init for attention, and LayerNorm (pre-norm). OK?
7. **Stacking rules (hard rule):** trained only on out-of-fold walk-forward predictions. EXP-002 (ablation) and EXP-003 (votes + regime only) are kept. Confirm.

## 10b · Market Transformer
8. **What does it replace?** It outputs (price, ticker, buy/sell). That overlaps the Phase 10 price transformer, the executor, and the rule that **only the rules engine creates orders**. Pick one:
   - (A) it replaces the price transformer: for each order the rules engine already chose, it outputs the limit price, and buy/sell is a check;
   - (B) it is an extra signal into the rules engine;
   - (C) something else.
9. **Inputs:** all raw numeric data, the LLM's words and price mods, and the order book for the ticker being evaluated. Alpaca's free tier gives **top-of-book quotes only for stocks** (IEX, not L2; no free historical L2 exists). Crypto has a live L2 stream, but no free history, so we'd have to record our own going forward. For stocks, accept quote, spread and imbalance features in place of a book? Start recording crypto L2 now so there is history later?
10. **Outputs + loss:** "output dim 1" is the price. The ticker is the input passed through, and buy/sell needs its own head. What price exactly (next session's limit price; the low for buys and the high for sells, in ATR units, as in the spec's executor target)? Cross-entropy fits buy/sell. Should the price use (A) quantile loss (spec: q10…q90), (B) MSE, or (C) CE over price bins?
11. **Size:** 6 layers, 16 heads at d = 256 gives head dim 16, which is small. Keep as written?
12. **Training data:** historical minute bars back to 2016 for the timing/price part, plus paper fills after ~300 fills (D-011: paper fills have no slippage, so they are optimistic). Promotion rule: it replaces q70 only if the fill rate stays within ±5% of target and measured slippage is lower (EXP-009)?
