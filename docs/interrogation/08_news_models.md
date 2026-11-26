# Questioning: Phase 8 · News models (8a tokenizer + SVM, 8b LLM)

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/08_news_models.md`, covering interfaces, config keys, tests and acceptance criteria.
> Sources: `docs/spec/model_descriptions_v1.md` (Tokenizer, SVM, LLM), spec "News intake and filtering" and "LLM design", `docs/plan/MODEL_PLAN.md` §6.
> Dedup, SimHash and the quota were moved to Phase 7 (pipeline structure).

## 8a · Tokenizer
1. **Tokenizer:** your description has the heading but nothing under it. Pick one:
   - (A) train our own BPE on our corpus (finance + general);
   - (B) reuse GPT-2's BPE;
   - (C) character-level.
   Is it shared by the SVM, the LLM and the keyword vocabulary? And what vocab size?

## 8a · SVM (relevance classifier)
2. **What kind of model?** The description (Kaiming normal init, ReLU, MSE loss) describes a small neural net, not a classic SVM. Classic SVMs have no init or activation and use a hinge loss. Pick one:
   - (A) a classic LinearSVC (squared hinge) on TF-IDF n-grams, ignoring init, ReLU and MSE;
   - (B) a neural net with ReLU hidden layers and Kaiming init, using an **SVM head** (hinge or squared-hinge loss, as in Tang 2013);
   - (C) exactly as written: a neural net with ReLU, Kaiming and **MSE** on a 0/1 target. Research: it works, but it trains worse than BCE or hinge.
   For (B) or (C): how many layers and how wide?
3. **"N-gram made from all training":** rebuilt each week from every article released up to that week (point in time), with 1–3-grams, TF-IDF and chi² top 5,000, as in the spec. Confirm. Never from articles after the training cutoff.
4. **Labels:** keep the spec's automatic label, where "relevant" = an asset it mentions had a next-day excess return beyond 2× its daily vol?
5. **"Append a valid / non-valid tag" to each article:** `articles` is a raw table and append-only, so the tag goes in a new table, `article_relevance(article_id, svm_version, prob, tag)`, without editing the article row. Keep the spec's keep rule (P ≥ 0.5, always keep the top 20)?
6. **Source type** as an input feature (one-hot of the 5 feeds)? Confirm.

## 8b · LLM
7. **"Literally all of the data from the data collection pipeline":** every article in the day's pool plus the full numeric snapshot. Numbers enter as one token per feature through a learned projection (spec), not as text. Which assets go in each pass: the asset the article mentions, the ~80 screened names, or every asset? What context length?
8. **The 2-expert MoE:** your text reads as two experts:
   - (1) a **language** expert that outputs the passage digestion (tickers, keywords, sentiment, event type);
   - (2) a **price-modification** expert that outputs Δ per feature and also reads expert (1)'s output.
   Since (2) consumes (1)'s output, this is a two-stage model, not a classic learned-router MoE. Pick one:
   - (A) a shared trunk with two expert heads and a fixed route by task;
   - (B) a learned top-1 router over 2 expert FFNs in every layer, plus the two-stage output;
   - (C) two separate stacked models.
9. **"Number data zeroed when articles are passed":** in the language pass the numeric tokens are zeroed (masked), and in the price pass both are present. Is that the meaning?
10. **Size:** 16 layers, 8 heads, d = 256 is about 12.6M parameters in the blocks plus the embeddings. That's ~25M with a 50k vocab, well under the spec's 50–125M. Accept it? This closes the carried "LLM size" question, and EXP-007 becomes a sweep around it.
11. **PagedAttention:** it manages the KV cache during **generation** (block tables, as in vLLM), and training uses ordinary causal attention (SDPA/flash). Proposal: build a paged KV cache for inference, which is also the frame for the KV-cache experiment EXP-006. OK?
12. **"Attention Residual":** see research note §6.2. Do you mean (A) the standard residual connection around each attention block, or (B) the published *Attention Residuals* method, where each layer attends over earlier layers' outputs in place of a plain residual add? If (B), is the plan baseline first, then (B) as an EXP?
13. **Loss:** cross-entropy fits next-token pretraining and the classification heads (tickers, event type). Sentiment (−1..1) and Δ are numbers. Should they be (A) binned and trained with CE, or (B) regression (MSE/Huber) heads?
14. **Init:** Kaiming for the MLP/FFN parts, PyTorch default for attention. Is this as written, and which Kaiming mode (fan_in, ReLU gain)? Note the FFN activation isn't specified: GELU?
15. **Pretraining corpus:** which general English set (e.g. a FineWeb-Edu sample), which finance text (SEC filings, our archive), how many tokens? Reuse your nano-gpt_manual code on `manual-models` as the base?
16. **Outputs carried over from the spec:** Δ clipped to ±1.0 and applied only to the top 20% by news_shock. How is news_shock defined (max |sentiment| × relevance, or the norm of Δ)? Keep the event taxonomy (earnings, guidance, M&A, regulation, protocol upgrade) or add types? Keep the injection defenses as written?
17. **Speculative decoding:** it needs a small draft model (e.g. a 2–4 layer copy) and only speeds up generation. Make it an EXP after the LLM works?
18. **Text transforms (your list):** random word order, letters only, Spanish or Mandarin translation. Are these augmentations for the SVM and LLM training sets (training fold only, a separate feature version), run as EXPs in Phase 9c? Research ranks back-translation (Spanish first) as the only one likely to help. Free local translators exist (opus-mt, NLLB-600M (non-commercial license), M2M100).
