# Questioning: Phase 9 · Text models (Transformer, CNN #2)

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/09_text_models.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Model contracts

1. **Transformer:** d=256, 6 layers, 8 heads to start? How many keyword tokens per asset-day (top-K by tf-idf)?
2. **Feature tokens:** project each feature's 64-day series with a linear layer or a small 1D conv?
3. **CNN #2:** K keywords, and what are the "5 embedding channels" exactly (counts, sentiment, recency, source mix, tf-idf)?
4. **Training start:** require at least 6 months of news, and do we keep only folds where `news_available = 1`?
