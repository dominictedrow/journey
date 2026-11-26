# Questioning: Phase 9 · Vote models (9a MLP + CNN #1, 9b Transformer + CNN #2, 9c transform experiments)

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/09_vote_models.md`, covering interfaces, config keys, tests and acceptance criteria.
> Sources: `docs/spec/model_descriptions_v1.md` (Transformer, MLP, CNN #1, CNN #2, data transforms), spec "Model contracts", `docs/plan/MODEL_PLAN.md` §6.
> The vote format (3-class vs binary) and every "General" rule are settled once in Phase 7 and apply here.

## 9a · MLP
1. **Inputs:** "all number related data from the LLM and not" = raw z-scored features **and** LLM-adjusted features (raw + Δ), for the last 5 trading days, flattened, plus an asset-class flag. One model for stocks and crypto, as in the spec?
2. **Shape:** "embed dim 256, 6 layers" read as an input projection to 256, then 6 hidden layers of width 256. Each layer is Linear → LayerNorm → tanh → dropout, with Xavier-normal init, then the vote head. Correct?

## 9a · CNN #1
3. **Inputs:** "all of the price data from the LLM" means which? (A) only the LLM-adjusted **price** features (returns, vol, trend), or (B) every adjusted numeric feature? The table is features on rows × days on columns. How many days (spec: 64)?
4. **Blocks:** 4 × (conv → activation → pool). The conv-layer activation isn't specified (sigmoid is only on the vote head). ReLU? Kernel sizes? Where does LayerNorm go (after each conv, or before the head)?
5. **Spatial pyramid pooling:** research says SPP's purpose is variable input sizes, and this table has a fixed size, so it adds little over global pooling. Keep SPP as written (with levels 1×1, 2×2, 4×4?), or make it an EXP against global pooling?
6. **Row order and 2D vs 1D:** keep EXP-005 (feature-family order vs correlation clustering; 2D convs vs 1D over time)?

## 9b · Transformer
7. **Inputs:** "number related data including the raw data collection and the LLM output for words and numbers" means raw features, adjusted features, and keyword tokens from the closed n-gram vocabulary (LLM keywords). Use the spec's token design: one token per feature, made by projecting its 64-day series, plus keyword tokens and a type embedding? How many keyword tokens per asset-day (top-K)? Linear projection or a small 1D conv for each series?
8. **"Sigmoid activation function":** do you mean (A) sigmoid inside the FFN blocks in place of GELU, or (B) sigmoid only on the vote output?
9. **Size + attention:** 8 layers, 4 heads (head dim 64), d = 256. Flash attention through `F.scaled_dot_product_attention` on CUDA (SDPA's math path on MPS). Attention residual as decided in Phase 8 Q12. OK?

## 9b · CNN #2
10. **Table:** stocks on rows (sorted by sector so neighbours are related) × **keyword pairs** on columns. Are "keyword pairs" the 2-grams of the closed vocabulary, or pairs of keywords that co-occur in the same article? What goes in each cell: count, sentiment-weighted count, tf-idf? One channel or several (the spec had 5)? One day, or a window of days?
11. **Blocks / SPP / LayerNorm:** the same questions as CNN #1 (Q4, Q5). It stays fully convolutional over rows, so each stock gets its own vote. Correct?
12. **Crypto:** CNN #2 covers stocks only, per your table. Do coins get a vote from it, or does the meta learner see "missing" for them?

## Shared
13. **Training start:** MLP and CNN #1 train now on 10+ years with Δ = 0 and `news_available = 0`. Transformer and CNN #2 wait for 6+ months of filtered news. Only news-available folds count for them. Confirm.
14. **Keep rule:** keep each model only if its out-of-sample IC beats the baseline (EXP-001), and the EXP-002 ablation still decides what the meta learner keeps. Confirm.

## 9c · Data transform experiments (your list, run on the 9a/9b models)
15. **Which to run, and in what order?** Research ranking:
   - **Test first:** quantile/rank-gauss and multi-frequency periodic encodings (the defensible form of your circle idea; a full circle makes the 0th and 100th percentiles neighbours); ratio and anchor scaling (S&P vs NQ, GDP-anchored, vintage-correct); soft per-regime normalization; OHLC chart images (Jiang, Kelly & Xiu 2023) as an extra view.
   - **Cheap look:** GAF/MTF/recurrence images; spectrograms of returns.
   - **Fibonacci:** only as a falsification test against equally spaced placebo levels.
   - **Likely dead ends:** colour-splotch canvas (no locality); sonification into a pretrained audio encoder.
   Run all of them as pre-registered EXPs, or drop the dead ends?
16. **"At data collection time":** each transform is computed in the feature layer as a new feature version, fit on the training fold only, and never written into raw tables. OK?
