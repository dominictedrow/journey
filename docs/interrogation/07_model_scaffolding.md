# Questioning: Phase 7 · Pipeline completion + shared model framework

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/07_model_scaffolding.md`, covering interfaces, config keys, tests and acceptance criteria.
> Sources: `docs/spec/model_descriptions_v1.md` ("General" section), `docs/plan/MODEL_PLAN.md`, spec "Model contracts" and "Validation".
> This phase builds **no real model**. It finishes the pipeline structure: every model slot is wired end to end with a stub, and one shared training framework applies the "General" rules to every model built in Phases 8–10.
> Research notes are in `docs/plan/MODEL_PLAN.md` §6. Each question says which note applies.

## Pipeline structure
1. **"Pipeline structure complete":** does it mean Phases 4–6 are done *and* the full daily cycle (collect → features → every model slot → rules → executor) runs on paper with **stub models**? Stubs: the baseline momentum vote for each vote model, Δ = 0 for the LLM, "keep all" for the SVM, the baseline score for the meta learner, and the q70 rule for the Market Transformer. Real models then replace stubs one at a time.
2. **News pipeline plumbing (from old Phase 8):** exact dedup, 64-bit SimHash near-dedup (Hamming ≤ 3 over 30 days) and the ~150-article quota are pipeline structure, not models, so they move into this phase. Confirm.
3. **The vote format, which every model shares:** your description says each model "only outputs a vote on bullish or bearish". The spec says P(up), P(flat), P(down), and the triple-barrier labels have a 0 class. Pick one:
   - (A) a 3-class softmax, with the vote derived as P(up) / (P(up) + P(down)) (research's pick);
   - (B) a binary sigmoid, with 0-labels dropped from training;
   - (C) a binary sigmoid, with 0-labels as a soft 0.5 target.
   Also confirm the vote is **per asset**, not one market-wide vote (spec gap #2).
4. **Hardware timing:** real training needs the RTX 5070. D-018 keeps the Mac to collectors and smoke tests, and this Mac has **8 GB** of RAM (your note said 16 GB; is that the Windows PC?). Should Phase 12 (Windows migration) move up to run right after Phase 7, before any real model is trained?

## The "General" rules (one shared framework, `src/qt/models/train/`)
5. **"All operations fused":** `torch.compile` on CUDA (it needs `triton-windows` on Windows), fused Adam (`fused=True`), and SDPA for attention. Eager mode on MPS. OK?
6. **Mixed precision:** bf16 autocast on the 5070 (no GradScaler needed). The project rule is fp32 on MPS. Keep that rule, or allow MPS autocast for experiments?
7. **Optimizer + L2:** you listed "Adam" and "L2 regularization" separately. Do you want (A) Adam with a true L2 penalty (the `weight_decay` argument of `torch.optim.Adam`, which is coupled), or (B) AdamW (decoupled weight decay), which is the modern default? The defaults would be lr and weight decay in config, per model.
8. **TP / PP:** tensor and pipeline parallelism split a model across several GPUs, and there is one GPU. Proposal: no TP. **PP** becomes one experiment (EXP), a micro-batched pipeline over the model's stages on one device, or a CPU + GPU split, to measure memory and throughput. Is that what "experiment with this one" means?
9. **Z-score on every number:** Phase 3 already stores a rolling 252-day z-score (winsorized ±4) for every time-series feature, plus cross-sectional ranks for stocks. Should models read those, or should the framework also z-score every model input with statistics fit **on the training fold only**? Does this include model outputs passed to other models (votes, LLM Δ) and regression targets (Market Transformer price)?
10. **Gradient clipping:** clip the global norm at 1.0 by default (config per model). OK?
11. **Dropout on every model:** the default rate is 0.1, set per model in config. OK?
12. **Data augmentation + temporal augmentation:** which methods are allowed? Research says the label-safe set is small noise (jitter), mixup with mixed labels, and block bootstrap, all on training rows only, with triple-barrier labels recomputed on any changed price path. Time warping breaks the 5-day label. Pick the set, or have it be a per-model EXP.
13. **Speculative decoding:** it only applies where a model *generates tokens* (needs a small draft model plus the LLM). The vote models, meta learner and Market Transformer do one forward pass, so it can't apply to them. Confirm it belongs only to the LLM (Phase 8b).
14. **Memory / custom kernels:** standard tools are gradient checkpointing, fused cross-entropy (e.g. Liger kernels), memory-mapped Parquet/Arrow loading with float16 storage, and small batches with gradient accumulation. Which ones are mandatory everywhere, and which are only used when a model doesn't fit?
15. **Batching:** a shared `DataLoader` over the walk-forward folds (batch size and accumulation steps in config, one asset-day per sample unless the model says otherwise). OK?

## Shared conventions (carried over from the old Phase 7 file)
16. **Training budget:** max epochs, early stopping on the walk-forward validation fold (patience in config), and seeds: one seed or a 3-seed ensemble?
17. **Calibration:** isotonic regression on a held-out fold for every vote, or sigmoid (Platt) when the fold has fewer than about 1,000 points?
18. **Model registry:** versions saved to `models/<name>/<version>/` with config, metrics, git SHA and pinned `(feature, version)` set. OK?
19. **Transform registry:** your "data transforms at collection time" experiments are stored as **new feature versions** (raw tables stay append-only), fitted on the training fold only, and each gets a no-lookahead test. The framework only builds the hook here, and the experiments run in Phase 9c. OK?
