# Questioning: Phase 8 · News pipeline (SVM) + homemade LLM

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/08_news_llm.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: News intake and filtering, LLM design

1. **SVM:** use LinearSVC with Platt calibration, or SGD? How is source type encoded?
2. **SimHash:** our own 64-bit implementation (D-009). Hamming distance ≤ 3 over 30 days, as in the spec. Confirm.
3. **LLM size:** start at about 50M or about 125M? Context length? Tokenizer (train our own BPE, or reuse GPT-2's)?
4. **Pretraining corpus:** which general English set (e.g. FineWeb-Edu sample, OpenWebText), and which finance text (SEC filings, our archive, free financial news datasets)? How many tokens?
5. **Your nano-gpt_manual code on `manual-models`:** should the LLM reuse it as its base?
6. **Task heads:** tickers (token classification), sentiment (regression), event type (classification), Δ (regression per feature). One model with several heads?
7. **KV-cache experiment (EXP-006):** what exactly do you want to test (eviction, quantized KV, paged blocks, prefix reuse)?
8. **news_shock:** how is it defined precisely (max |sentiment| × relevance, or the norm of the predicted Δ)?
9. **Event type taxonomy:** use the spec's list (earnings, guidance, M&A, regulation, protocol upgrade), or more?
