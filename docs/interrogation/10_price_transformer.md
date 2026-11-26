# Questioning: Phase 10 · Price transformer (execution)

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/10_price_transformer.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Model contracts (price transformer), Execution layer

1. **Timing head:** how far back should training on historical minute bars go (Alpaca free has history since 2016)? Label = the best slot by realized low/high?
2. **Limit head:** quantile loss over 10/30/50/70/90. Train on paper fills plus historical bars?
3. **D-011:** paper fills have no slippage model. How should we discount that bias?
4. **Promotion rule:** which metric decides when the model replaces q70 (fill rate within ±5% of target and lower slippage)?
