# Questioning: Phase 7 · Numeric models (MLP, CNN #1) + meta learner

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/07_numeric_models_meta.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Model contracts, Validation

1. **MLP:** depth and widths (spec start 512→32), dropout, normalization, loss (3-class CE), class weights?
2. **MLP input:** 5 days of adjusted features flattened. One model for stocks and crypto plus a flag. Confirm.
3. **CNN #1:** image size (about 150 features × 64 days); kernel sizes; row order by feature family vs correlation clustering (EXP-005)?
4. **Meta learner architecture:** gradient boosting (LightGBM, a strong tabular baseline), or a small shared-weight MLP as written?
5. **Calibration:** isotonic per the spec. Use sigmoid (Platt) if a fold has fewer than about 1,000 points?
6. **Training budget:** max epochs, early stopping on the walk-forward validation fold, seeds (one or an ensemble of seeds)?
7. **Should the Mac (MPS) train small versions now, or wait for the Windows GPU?**
8. **Model registry:** versions saved to `models/<name>/<version>/` with config, metrics and git SHA. OK?
