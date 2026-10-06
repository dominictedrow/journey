# XAUUSD Dataset — Preprocessing Pipeline

## Dataset size

- **Main dataset**: `model_training/data/new_multi_stacked_XAUUSD_Train_10_new.db` — single table `XAUUSD`, **2,168,598 rows × 120 columns**, 1.5GB on disk (~1GB as float32 in memory). Comfortably analyzable; avoid a naive `pandas.read_sql("SELECT *")` without chunking/dtype-casting since it'll balloon in memory, but SQL aggregation/sampling is trivial.
- **KMeans artifact**: `model_training/data/data_scalers/kmeans_model_multi_stacked_large_9216.pkl`, 360MB — a fitted `MiniBatchKMeans` with **9,216 clusters**, dim 26. This is a model, not raw data. It's a *larger, newer* clustering model than what actually produced the current DB — the DB's `Pattern_Class` only ranges 0–359, meaning it was built with an earlier 360-cluster version.

Total footprint: 1.9GB.

## Pipeline, step by step

The XAUUSD 1-minute data goes through this ordered pipeline (traced through `data_transformation/data_building/*.py`, confirmed against the DB's actual schema/values):

### 1. Multi-timeframe pooling
`data_transformation/data_building/1_1_csv_to_sqlight_multi_timeframe_pooling_multi_stacked.py`

Raw 1m CSV bars get resampled and aligned against each target timeframe (5m/15m/30m/60m) with forward-fill for gaps, producing columns like `1m_Close_0..4`, `5m_High_0..2`, etc. — each row carries a short lookback window of sub-bars from multiple timeframes simultaneously. The whole process repeats per target timeframe and the results are **stacked vertically** (tagged by `Group`), which is why the table has 2.17M rows instead of ~1 timeframe's worth (confirmed: Group 5→1.36M rows, 15→460K, 30→231K, 60→116K).

- **Why it matters:** gives the model multi-resolution market context (micro-structure + higher-timeframe trend) in one row instead of forcing separate models per timeframe.
- **Architecture link:** the `Group` value survives all the way to training as `input_2`, embedded and broadcast across the sequence — the model literally conditions on which timeframe regime a row belongs to.

### 2. Datetime decomposition
`data_transformation/data_building/2_multi_timeframe_format_multi_stacked.py`

`DateTime` is split into `Day`, `Month`, `Hour`, `Minute` and dropped.

- **Why it matters:** raw timestamps aren't learnable; calendar components expose cyclical/seasonal structure (session times, day-of-week effects).
- **Architecture link:** these become ordinal-encoded categorical inputs concatenated into the feature vector.

### 3. Forward-looking target construction
`data_transformation/data_building/multi/5_multi_target_multi.py` + `data_transformation/data_building/5_2_multi_target_multi_stacked_last_row_delete.py`

`Target1/2/3` = next bar's **High/Low/Close**, built via a self-join shifted by one row; the final row per group (no "next" row to look ahead to) is deleted.

- **Why it matters:** this is the supervised-learning label — without it there's nothing to predict. Deleting the dangling last row prevents a leaked/undefined target.

### 4. Candle-pattern clustering
`data_transformation/data_building/multi/8_1_new_kmeans_features_multi_stacked.py`

For each row, 26 binary up/down comparisons between consecutive 1m sub-bar OHLC groups are computed (`Pattern_1..26` — essentially candlestick microstructure motifs), then a `MiniBatchKMeans` clusters these 26-dim binary vectors into discrete regimes, written as `Pattern_Class` (this DB uses a 360-cluster model; a newer unused 9,216-cluster version also sits in `data_scalers/`).

- **Why it matters:** unsupervised discovery of recurring short-term candle "shapes" the model wouldn't otherwise get from raw OHLC.
- **Architecture link:** `Pattern_Class` becomes `input_3`, fed through a dedicated learned `Embedding` layer (`SequencePatternEmbedding`) — treated like a vocabulary of pattern "words," analogous to token embeddings in NLP, concatenated with the raw feature and Group embeddings before the Transformer stack.

### 5. Technical indicators + peak/valley labeling
`data_transformation/data_building/9_features_multi_stacked_main_11.py`

~25 `New_*` indicators computed per timeframe with window lengths scaled by `Group` (SMA, RSI, MACD/signal, Bollinger Bands+width, EMA fast/slow/trend, ATR/ATR%, Stochastic K/D, ROC, momentum, trend-strength z-score, volatility regime ratio, support/resistance range, price position, etc.). Rows still `NaN` after indicator warm-up are dropped. Then a **rule-based scoring heuristic** (RSI extremes + Bollinger touches + MACD cross + proximity to local long-MA extrema + higher-timeframe confirmation) detects turning points on a long MA, forward-fills a `Target4` regime label (5000 = peak/sell zone, 1000 = valley/buy zone), and merges signal runs shorter than a timeframe-dependent minimum distance to kill flicker.

- **Why it matters:** converts a continuous, noisy price series into a much richer feature set plus a denoised categorical regime label. In this DB the two classes end up fairly balanced (1,043,045 vs 1,125,553 rows).
- **Architecture link:** `Target4` is what the model actually predicts; the technical indicators are the bulk of the continuous feature vector.

### 6. Label remap at load time
`load_data_from_sqlite()` in the training script

Rows with `NaN` `Target4` dropped; `5000→0` (hold/majority), `1000→1` (signal/minority) — binary classification framing (`binary_crossentropy`).

### 7. Feature pruning
`preprocess_and_combine_data()` in the training script

Raw multi-timeframe OHLC price columns (the stage-1 output) and several redundant `New_*` indicators (SMA, BB bands, long MA, EMA fast/slow, price-volume-trend, support/resistance levels) are dropped, keeping `Group` and a decorrelated indicator subset.

- **Why it matters:** manual dimensionality reduction — raw price levels aren't stationary/comparable across time and are redundant with the indicators already derived from them.

### 8. Ordinal encoding

`Day`, `Month`, `Hour`, `Minute`, `Group`, `Pattern_1..26`, `Pattern_Class` are encoded via `OrdinalEncoder` with a **fixed, pre-registered category vocabulary** (so train/val stay consistent even if a category is missing/rare in one split).

### 9. Min-Max scaling
`AdaptiveMinMaxScaler`

Continuous columns are globally MinMax-scaled (one scaler fit across all symbols/tables); the fitted scaler + encoder + category schema + `seq_len` are `joblib`-persisted (`scalers/*.pkl`) and reloaded unchanged for validation — preventing train/val leakage or scale mismatch.

### 10. ID remapping

`symbol_id` (per source table/instrument) and `pattern_class_id` (dense remap of the raw KMeans cluster IDs) are assigned via small JSON mapping files kept stable across runs, so embedding indices stay consistent.

### 11. Sequence windowing
`ShiftingSequenceGenerator`

Rows are grouped by `Group` (never crossing timeframe boundaries), then sliding windows of `seq_len=8` consecutive bars are built; the label comes from the **last row** of each window — classic many-to-one sequence classification.

- **Architecture link:** this literally produces the `(seq_len, features)` tensor shape (`input_1`) the Transformer expects.

### 12. Class balancing + augmentation at batch time
`ChronologicalBalancedSequenceGenerator`

Minority-class windows are oversampled in place (replication factor ≈ majority/minority ratio, with optional randomized jitter on the factor), and replicated copies get Gaussian noise + random uniform scaling applied so duplicates aren't identical. A separate epoch-cycling variant randomly resamples a capped number of majority-class sequences per epoch rather than permanently discarding the excess.

- **Why it matters:** keeps the binary classifier from just predicting the majority class, without literally throwing away 90%+ of majority data forever.

## End-to-end architecture tie-in

The three model inputs — `input_1` (scaled indicator sequence), `input_2` (Group/timeframe id), `input_3` (Pattern_Class id per timestep) — map directly to preprocessing stages 1, 4, and 5/9 respectively, then get concatenated (raw features + pattern embedding + broadcast group embedding) before entering the Transformer/"sphere transform"/conv-mixer stack, ending in a softmax over the binary `Target4` label from stage 5/6.
