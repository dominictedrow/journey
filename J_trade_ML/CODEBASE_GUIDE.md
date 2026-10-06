# J_trade_ML — Codebase Guide

A reading map for this repository, written for working through it file-by-file to build Python fluency. It covers what the project does, how the folders relate, and what every script contains (functions, classes, imports), grouped so the heavy duplication in this repo doesn't drown you.

## 1. What this project actually is

This is a personal research repo for building a machine-learning trading system on **gold (XAUUSD)**, 1-minute bar data. The pipeline in broad strokes:

1. **`model_training/data_transformation/`** — raw 1-minute CSV price data gets turned into a large SQLite database of engineered features + a target label, through a numbered sequence of scripts.
2. **`model_training/model_concepts/`** — a lab of ~10 different neural-network architecture *families* (spiking neural nets, gated MLPs, transformers, CNNs, ensembles...) tried out against that data, each iterated many, many times.
3. **`model_training/training_models/`** — the "production" training entry points: much larger, more complete scripts that actually train and evaluate a chosen architecture end-to-end (data loading → sequence windowing → class balancing → training loop → checkpointing → live-trading model export).
4. **`docs/preprocessing.md`** — an existing, well-written narrative of the feature/label pipeline. **Read that file before/alongside this one** — it explains *why* each preprocessing stage exists and how it ties into the model's three inputs. This guide focuses on the code; that doc focuses on the data science reasoning.

This is a research/experimentation repo, not a clean library — expect duplicated logic, versioned filenames (`_v2`, `_testing`, `_low`, `.2`...), dead code, and no shared `import`-able modules. Every script is close to self-contained (copy-paste-and-modify style), which is actually convenient for learning: you can read one file top to bottom without chasing definitions across the codebase.

## 2. Top-level layout

```
J_trade_ML/
├── docs/preprocessing.md          ← read this first for the "why"
├── graphify-out/                  ← auto-generated knowledge-graph tooling output, not project code — ignore
└── model_training/
    ├── data/                      ← the actual data artifacts (a 1.5GB SQLite DB, a KMeans .pkl) — not code
    ├── data_transformation/       ← ETL pipeline: raw CSV → engineered SQLite table
    │   ├── data_building/         ← the numbered pipeline stages (start here for code)
    │   │   ├── multi/             ← "multi-timeframe stacked" variant of stages 2, 5, 8, 9
    │   │   ├── others/            ← abandoned/alternative feature approaches
    │   │   └── 9_features_multi_stacked/  ← a refactor of stage 9
    │   └── tools/                 ← tiny one-off utility scripts
    ├── model_concepts/            ← experimental model-architecture lab (10 families, many iterations each)
    │   ├── best_models/, clustering_models/, cnn_models/, ensemble_models/,
    │   │   g_mlp_models/, lion_heart_models/, lionheart_spiking_nn/,
    │   │   old_(softmax)/, scgMLP_models/, transformer_models/, snn/
    └── training_models/           ← the "real" training entrypoints, biggest/most complete scripts
        ├── model_builds/
        │   ├── ASTA_MHA_versions/, Qg_MLP_versions/, TTCM_SEQ/, old_models/,
        │   │   backup_current_models/, live_models/#1_small_37/
        ├── new_model_testing/, tools_&_notes/
        └── (7 loose top-level TTCM_SEQ_* scripts)
```

## 3. Suggested reading order

Given ~140 Python files and ~99,000 lines with a lot of near-duplication, reading everything front-to-back in file order is not a good use of time. A better path, easiest concepts to hardest:

1. **Tiny standalone utilities** (10–80 lines, one clear job each) — the fastest way to see clean, isolated `sqlite3`/`pandas` patterns without wading through ML code:
   - `model_training/data_transformation/tools/test.py`, `DB_to_cvs.py`
   - `model_training/data_transformation/data_building/others/PolynomialFeatures.py`
   - `model_training/data_transformation/data_building/5_2_multi_target_multi_stacked_last_row_delete.py`
   - `model_training/model_concepts/lionheart_spiking_nn/reset_folders.py`
   - `model_training/training_models/model_builds/live_models/#1_small_37/data/databases/tools/database_data_delete.py`

2. **The data pipeline, in numeric order** (Section 5 below), reading `docs/preprocessing.md` alongside each stage. This is where the `sqlite3`, `pandas`, `numpy`, chunked/`multiprocessing`/`joblib`/`numba` patterns live.

3. **One model_concepts file per family**, cheapest-to-understand first (Section 6 gives a recommended pick per family). This is where Keras/TensorFlow custom `Layer`/`Model`/`Callback` subclassing lives — the single most repeated Python pattern in this repo.

4. **One full `training_models` script**, e.g. `training_models/TTCM_SEQ_stacked_multi_main_testing_new_16b.py` (Section 7). This is the "everything at once" file: custom training loops, learning-rate schedules, data generators, logging, GPU config, checkpointing. Treat every other file in `training_models/model_builds/` as a *diff* against this one rather than reading each independently.

## 4. A note on the duplication

Almost every directory here is really **one script copy-pasted repeatedly with small tweaks** (a hyperparameter, a new layer, a bugfix, a different loss). The AST scan behind this guide confirms it: dozens of files share the exact same function/class name lists. Practical implication for you as a reader: once you've read the "baseline" file for a family, skim the diffs of a couple of variants (`diff -u file_a.py file_b.py` in a terminal is the fastest way) rather than reading each one in full — you'll learn much more per minute that way.

---

## 5. `data_transformation/` — the ETL pipeline (read in this order)

Cross-reference `docs/preprocessing.md`, which documents *why* each stage exists; this section is *what's in the file*.

| # | File | What it does | Key functions |
|---|------|---------------|----------------|
| — | `data_building/1_csv_symbol_exstraction_multi.py` (18 lines) | Small pre-step: walks a folder and pulls out per-symbol CSVs. `import os` only. |  |
| 1 | `data_building/1_1_csv_to_sqlight_multi_timeframe_pooling_multi_stacked.py` (173 lines) | Stage 1 of the pipeline. Resamples raw 1-minute CSV bars against multiple target timeframes (5m/15m/30m/60m), forward-fills gaps, and writes the result into SQLite, tagging each row's source timeframe via a `Group` column. Uses `concurrent` (parallelism), `tqdm` (progress bars). | `parse_timeframe`, `load_data`, `integrate_timeframes`, `process_chunk`, `fill_forward`, `process_files`, `main` |
| 2 | `data_building/2_multi_timeframe_format_multi_stacked.py` (60 lines) / `data_building/multi/2_multi_timeframe_format_multi.py` (58 lines, near-identical variant) | Stage 2. Splits `DateTime` into `Day`/`Month`/`Hour`/`Minute` columns and drops the original timestamp. | `process_table` |
| 3 | `data_building/multi/5_multi_target_multi.py` (67 lines) + `data_building/5_2_multi_target_multi_stacked_last_row_delete.py` (53 lines) | Stage 3. Builds `Target1/2/3` (next bar's High/Low/Close) via a self-join shifted by one row, then deletes the dangling final row per group (no "next" row exists for it). | `copy_and_modify_tables`, `delete_blank_target_rows` |
| 4 | `data_building/multi/8_1_new_kmeans_features_multi_stacked.py` (188 lines) — this is the version actually used to build the current DB | Stage 4. Computes 26 binary up/down comparisons between consecutive 1-minute sub-bars (`Pattern_1..26`), then fits/applies `MiniBatchKMeans` to cluster these into a `Pattern_Class` regime id. Uses `sklearn`, `joblib` (persist the fitted clusterer), `multiprocessing`. | `get_table_names`, `load_data_from_sqlite`, `reshape_data`, `identify_patterns_group_1min`, `classify_patterns`, `save_to_new_db`, `ensure_unique_column_names`, `parallel_load_data`, `process_table`, `main` |
| 5 | `data_building/9_features_multi_stacked_main_11.py` (502 lines) — the version actually used | Stage 5, the biggest/most important feature-engineering script. Computes ~25 technical indicators (SMA, RSI, MACD, Bollinger Bands, EMA, ATR, Stochastic, ROC, momentum, volatility regime...) via `pandas_ta`, then runs a rule-based peak/valley detector to derive `Target4` (the actual classification label: peak/sell vs. valley/buy zone), forward-filling and denoising short flicker runs. | `add_new_features`, `detect_peaks_valleys`, `process_table`, `process_database` |

**Other files in this area, and how they relate:**

- `data_building/multi/8_new_kmeans_features_5m_multi.py`, `... - Copy.py`, `..._1.py`, `..._2.py` — earlier/parallel iterations of the KMeans clustering stage above; `_2.py` is the most evolved (adds `numba`-accelerated pattern computation and `psutil`-based resource-aware parallelism — worth a look if you want to see `numba.jit` in action).
- `data_building/multi/9_new_2_features_multi_stacked_{1,2,3,5,6}.py` and `data_building/others/9_new_2_features_multi_stacked_main_main.py` — earlier iterations of stage 5, same function names (`add_new_features`, `detect_peaks_valleys`, `process_table`, `process_database`), growing from 156 → 504 lines as more indicators/logic get added. `9_features_multi_stacked/9_features_multi_stacked_new_method.py` (693 lines) is a later refactor that adds `add_slope_features` and `_nearest_extreme` helpers and switches to `typing` annotations.
- `data_building/others/5_classification.py` and `6_classification_live_conversion.py` — an alternative/earlier labeling approach (`process_pricing_data`, `preprocess_data_for_ml`) not part of the main numbered pipeline.
- `data_building/others/8_new_dbscan_features_5m_multi.py` — an abandoned alternative to the KMeans clustering stage, using `DBSCAN` instead (`identify_patterns`, `classify_patterns`, `save_to_new_db`).
- `data_building/others/PolynomialFeatures.py` (63 lines) — a small standalone experiment with `sklearn.preprocessing.PolynomialFeatures` (`add_features`, `create_polynomial_features`). Good short read for basic `sklearn` feature-engineering syntax.
- `data_building/10_1st_100_rows delete.py` (34 lines) — trivial `sqlite3` housekeeping script, deletes the first 100 rows of a table.

**`data_transformation/tools/`** — standalone utilities, not part of the numbered pipeline:
- `DB_to_cvs.py` (77 lines, `main`) — exports a SQLite table back out to CSV.
- `test.py` (36 lines, `get_feature_names`) — lists the column/feature names of a table; good tiny first read.

**Note on stage 6 (label remap) and stage 7 (feature pruning):** per `docs/preprocessing.md`, these aren't separate files — they live inside `load_data_from_sqlite()` and `preprocess_and_combine_data()` in the *training* scripts (Section 7), not in `data_transformation/`.

---

## 6. `model_concepts/` — the architecture lab

Ten sub-folders, each a family of experiments. Nearly every file in here defines the same handful of Keras/TF building blocks (`RealTimeDashboard(Callback)` for live-plotting training progress, `SpatialGatingUnit`/`gMLPBlock` for gated-MLP layers, `load_data_from_sqlite`/`create_overlapping_sequences`/`preprocess_data` for data loading) and then a family-specific model architecture on top. Recommended one-file-per-family reads, cheapest concepts first:

### `g_mlp_models/` — start here for family reads
Gated-MLP ("gMLP") architecture: `SpatialGatingUnit` (a custom `Layer` that splits the sequence dimension and gates one half against the other) feeding `gMLPBlock`. Four versions (`g_mlp_#_all.py`, `_upgrade.py`, `_upgrade_4.py`, `_upgrade_5.py`, 280–352 lines each) — read `g_mlp_#_all.py` first since it also defines `PositionalEncoding` and `load_all_tables_from_sqlite`, then skim the diffs. This is the cleanest example in the repo of subclassing `tf.keras.layers.Layer` (`__init__`/`build`/`call`/`get_config`).

### `ensemble_models/`
`ensemble_model.py` (333 lines) combines a gMLP block with a `CustomModelCheckpoint(ModelCheckpoint)` callback. `ensemble_model_5.py` (612 lines) is a later, larger version that adds `PositionalEncoding`, an LSTM path (`create_LSTM_model`), and `calculate_metrics`/`save_best_model` helpers.

### `best_models/`
`cross_dual_model_testing.py` (594 lines) and `LSTM_cross_dual_model_test.py` (612 lines) — near-identical "current best" snapshots combining gMLP with either a second gMLP head or an LSTM head, plus `calculate_metrics`/`save_best_model`. These are essentially checkpoints of `ensemble_model_5.py`.

### `transformer_models/`
Standard Transformer-encoder building blocks as custom layers: `FeatureAttention`, `TemporalFusion`, `TransformerBlock` (and `SEBlock` — a squeeze-and-excitation block — in the `_upgrade_0` version). `transformer_model_0_#.py` (125 lines, plain functions rather than classes: `positional_encoding`, `build_transformer_model`) is the simplest — good first read if you want to see a Transformer built the "functional API" way before the class-based versions.

### `cnn_models/`
Three different input modalities: `cnn_gated_(single_image).py` and `cnn_lstm_(model_2)_main.py` treat candlestick charts as **images** (loaded via `PIL`, walked via `glob`) fed through CNN/ConvLSTM2D layers; `trasformer_ViT.py` uses the `transformers` library (Vision Transformer) on the same image data. Good reads if you want to see `tf.keras.layers.ConvLSTM2D`/image-pipeline code (`process_path`, `load_images`, `create_sequences`) rather than tabular data.

### `clustering_models/`
`clustering_model.py` (101 lines) — a small standalone example of `tensorflow` data generators over image sequences (`preprocess_sequence`, `data_generator`), unrelated to the KMeans clustering in `data_transformation/`.

### `scgMLP_models/` — read after `g_mlp_models/`, this family builds on it
"Stacked/combined gMLP" — extends the gMLP idea with a second and third parallel gMLP branch (`gMLPBlock2`, `gMLPBlock3`, `MiddleLayer`, `ThirdgMLPModel(Model)`) that get combined via a `combined_step()` training function. Ten versioned files (`_1` through `_6`, plus `_test`/`.1`/`.2` suffixes, 631–1032 lines) escalate in complexity — later ones (`_4` onward) add a hand-rolled prime-number-based learning-rate schedule (`is_prime`, `get_prime_numbers`, `dynamic_min_learning_rate`) which is a fun, unusual bit of code to read. `scgMLP_model_2.py` is a reasonable representative middle-complexity read.

### `lion_heart_models/` and `lionheart_spiking_nn/` — the spiking-neural-network family, most advanced material here
This is the repo's biological-inspired architecture: neurons that accumulate a membrane potential and fire discrete spikes, trained with an STDP-like (spike-timing-dependent plasticity) local update rule alongside normal backprop. Core recurring classes: `SpikingNeuronLayer`, `CustomDenseLayer` (weight updates via `stdp_weight_update`), `PopulationEncodingLayer`/rate-coding functions to turn continuous features into spike trains, and hand-written `train_step`/`validation_step` functions (i.e., NOT `model.fit()` — a custom `tf.GradientTape` loop, which is worth studying if you haven't seen one).
  - Read `lion_heart_models/new_nn_test_1.py` (291 lines) first — it's the simplest, `CustomMLPTensorFlow` is a plain Python class (not even a Keras `Layer`) with `forward`/`train_step`/`compute_loss`/`apply_gradients` written by hand, the clearest place to see what Keras normally does for you.
  - Then `snn/new_nn_main_test_1.py` → `..._8.py` → `..._24.py` (the `snn/` folder has 14 versions, `_1` through `_25.2`, each adding a bit: `ReversedDense`, then `SpikingNeuronLayer`, then STDP updates, then dual-input models) — pick 2–3 spread across that range rather than all 14.
  - `lionheart_spiking_nn/example_test.py` (116 lines, pure `numpy`, no TensorFlow) — a from-scratch spike-train simulation, good for understanding the *concept* before the Keras version.
  - `lionheart_spiking_nn/lion_heart_hybrid_test_{2,3,4}.py` (~1200 lines each) and `lion_heart_LLM.py` (1156 lines) are the most elaborate versions — add a `Preprocessor` class, `SequentialSpikeTrainModel(Model)`, `DebuggingCallback`. Skim rather than deep-read unless SNNs specifically interest you.
  - `lionheart_spiking_nn/neuronal_activities/*.py` (3 near-identical files) — not model code; render/export the spiking activity as 3D visualization videos (`imageio`, `moviepy`, `matplotlib`). Interesting if you want to see video-generation code.
  - `lionheart_spiking_nn/reset_folders.py` — trivial cleanup utility.

### `old_(softmax)/` — lowest priority, historical/abandoned approach
An earlier numeric-softmax labeling approach the author moved away from (folder literally named `old_numberical_softmax_(not_good)`). Contains its own miniature copy of the data pipeline (`data_transformation_1/2/3/` — `pandas_(csv_sqlite).py`, `pandas_(sqlite_create).py`, `pandas_(peaks_valleys).py`, `pandas_(vacuum_database).py`, `pandas_(timestamp_remove).py`, `pandas_(database_spliter).py`) plus early model attempts `g_mlp.py`, `g_mlp_(main).py`, `g_lstm_(history_step).py`, `g_rnn.py`.
  - **Heads up:** the three `pandas_(peaks_valleys).py` copies (in `data_transformation_1/2/3/`) have a syntax error — a stray trailing `8` after `.shift(window)` on line 27 — so they won't run as-is. Read for the pandas logic, don't try to execute them.

---

## 7. `training_models/` — the full training entrypoints

These are the largest, most complete files in the repo (900–2550 lines) — everything from `model_concepts/` plus data loading, sequence generation, class balancing, GPU configuration, logging, and checkpointing in one script. Three architecture families, each with many versions, plus loose top-level files and a `live_models/` deployment snapshot.

### Recommended single read: `TTCM_SEQ_stacked_multi_main_testing_new_16b.py`
(Appears three times, identically, at `training_models/TTCM_SEQ_stacked_multi_main_testing_new_16b.py`, `training_models/model_builds/TTCM_SEQ/TTCM_SEQ_stacked_multi_main_testing_new_16b.py`, and `.../TTCM_SEQ/old/...` — just read one copy.) This is the most complete, representative "final" training script in the repo. Notable pieces, in the order you'll hit them reading top to bottom:
- `Tee`/`Tee2` — classes that duplicate `stdout` to both the console and a log file (a real-world pattern for `sys.stdout = Tee(...)`).
- `AdaptiveMinMaxScaler` — a hand-rolled scaler (`fit_transform`/`transform`/`inverse_transform`), not `sklearn`'s, because it needs `joblib` persistence across separate train/inference scripts (see `docs/preprocessing.md` §9).
- `AdaptiveSphereTransformLayer` / `InvertedSphereTransformLayer` — an unusual custom geometric transform layer (projects features onto/off a hypersphere) — worth reading `build`/`call` closely as an example of a nontrivial custom Keras layer with learnable parameters.
- `AdaptivePositionalEncoding`, `RotaryPositionalEmbedding` — two different positional-encoding schemes side by side (the second is a from-scratch RoPE implementation, `_rotate_half` + rotation math).
- `CustomMultiHeadAttention`, `DecoderLayer`, `MarketAwareTransformer(Model)` — the actual Transformer stack.
- `SpatialGatingUnit`/`gMLPBlock`, `SequencePatternEmbedding` — the gMLP + pattern-embedding path (this is `input_3` from `docs/preprocessing.md`).
- `CustomSchedule(LearningRateSchedule)` — a custom learning-rate schedule class (the standard Transformer warmup/decay formula).
- `PrintLR`, `PrintPredictionsAndLoss_0`, `PrintUnscaledLoss` — `Callback` subclasses for per-epoch logging.
- `ShiftingSequenceGenerator(Sequence)` / `ChronologicalBalancedSequenceGenerator(ShiftingSequenceGenerator)` — the sequence-windowing + minority-class oversampling generators described in `docs/preprocessing.md` §11–12. This is the best place in the repo to see a custom `tf.keras.utils.Sequence` (`__len__`, `__getitem__`) plus inheritance between two of them.
- `configure_gpus`, `load_data_from_sqlite`, `preprocess_and_combine_data`, `combined_data` — top-level driver functions tying it all together.

### `model_builds/TTCM_SEQ/` and the 7 loose top-level `TTCM_SEQ_*` files
All variants of the file above. Differences worth knowing about rather than re-reading in full:
- `TTCM_SEQ_stacked_multi_main_final.py` — swaps `RotaryPositionalEmbedding` back out for a plain `PositionalEncoding` + adds a `SymbolEmbedding` layer.
- `..._32b_testing.py` / `..._32b_class_2_testing_0.py` — replace the sphere-transform + rotary-attention stack with `SphericalDenseLayer` and `FractalTemporalGate`.
- `..._32b_class_2_testing_1.py` — the most architecturally different one: drops the Transformer stack entirely for `NeuroplasticEmbedding`/`CorticalAttention`/`DualPredictionHead`/`CorticalTransformer(Model)` (adds a custom `_kl_loss` — KL-divergence regularization term, if you want to see that pattern).
- `model_builds/TTCM_SEQ/old/` — earlier drafts of all of the above; skip unless diffing.

### `model_builds/ASTA_MHA_versions/` — the Transformer-with-multi-head-attention family
Ten files (904–1420 lines), all built around `CustomMultiHeadAttentionModel(Model)` with `EncoderLayer`/`DecoderLayer`. Read `ASTA_MHA_v1_Standard.py` first (the plainest version); then note what the others add:
- `ASTA_MHA_v1_MaxMin_v2.py` swaps in exotic layers: `QuantumFractalTransform`, `HypercomplexAttention`, `HolographicEncoding`, `SOMLayer` (self-organizing map) — interesting/unusual names but same `Layer` subclassing pattern.
- `ASTA_MHA_v1_predictions_stacked_multi*.py` (3 variants) add `SymbolEmbedding` and swap the encoder/decoder for a plain `Encoder`/`Decoder` + `CustomTransformerBlock`.
- `ASTA_MHA_v1_15m_predictions.py` and `backup_current_models/ASTA_MHA_v1_5m_predictions*.py` add `ConvolutionalPositionalEncoding` and `AttentionAugmentedConv2D` (attention mixed into a conv layer).
- `model_builds/old_models/ASTA_MHA_v1_*` — earlier, much shorter drafts (just `RealTimeDashboard` + loading/preprocessing functions, no custom attention yet) — actually a decent "before" snapshot to compare against `_Standard.py`'s "after".

### `model_builds/Qg_MLP_versions/` — quantum-flavored gMLP family
`QgMLPBlock` — a gMLP variant whose `build`/`call` methods are named `phi_x_alpha`, `phi_x_beta`, `Information_Entropy`, `Markov_Chain`, `QuantumLayer` (physics/info-theory-flavored function names dressing up what are still standard tensor ops — good exercise in reading past unusual naming to the actual math). `Qg_MLP_1.py` → `_2.py` → `_3.py` are near-identical (429→455 lines); `Qg_MLP_v2_experiments.py` (656 lines) is a bigger rewrite adding `fft_KAN` (FFT-based Kolmogorov-Arnold Network layer) and `QuantumEquation`.
  - `current_test/ASTAF_fractals_MHA_v{1,2}_MaxMin.py` (~1220 lines each) — a *merge* of the ASTA_MHA attention stack with a `FractalDenseCell` recurrent layer (`fractal_rnn` function) — read after both `ASTA_MHA_v1_Standard.py` and a `Qg_MLP` file to see how two families get combined.
  - `current_test/Dyna_MLP_v1_testing_0.py` (728 lines) — a related but distinct idea: `DynaDense`/`DynamicOutputModel` build a network whose output layer size is chosen dynamically (`compute_output_shape` computed at runtime).

### `new_model_testing/SNAKEMLP_stacked_multi_main_testing_new_32b_testing.py` (2248 lines)
The single largest file in the repo (credited "By: JD (ThinkBe)" in its docstring — worth noting if you're tracking authorship). A `TTCM_SEQ`-style training script (same `Tee`, `AdaptiveMinMaxScaler`, `ShiftingSequenceGenerator` infrastructure) built around two novel layers: `SDMLP` (`get_focus_blocks`) and `SnakeMind` (`snake_dynamics`) — a different gating/routing mechanism than the gMLP `SpatialGatingUnit` used elsewhere.

### `model_builds/live_models/#1_small_37/` — the deployment snapshot
A self-contained copy of one specific trained model plus its supporting artifacts, structured for live/production use rather than experimentation:
- `models/TTCM_SEQ_stacked_multi_main_1_live.py` (898 lines) — a **trimmed** version of the main training script: keeps `AdaptiveMinMaxScaler`, `load_data_from_sqlite`, `preprocess_and_combine_data` but drops most of the training-loop machinery in favor of `initialize_model`/`load_data_and_initialize_model` — i.e., this is closer to an *inference* script. Good file to read right after the main `TTCM_SEQ` training script to see what changes when you go from "train" to "serve".
- `models/model_testing/Qg_MLP_v1_live_trading_15m.py` (538 lines) — a live-trading variant of the Qg_MLP family (`ExtendedPositionalEncoding`, `CustomActivation`, `DynamicDenseLayers`).
- `data/data_scalers/8_new_kmeans_features_multi_stacked.py` (216 lines) — a live-serving copy of the KMeans pattern-clustering stage (adds `suppress_specific_warning`, uses `contextlib`).
- `data/databases/tools/database_data_delete.py` (47 lines) — tiny SQLite cleanup helper.
- Non-code artifacts here worth knowing about (not scripts, but explains what the code above operates on): `models/predictions/predictions_multi_stacked_1.txt` (live prediction output), `models/symbol_id_mapping.json` (the ID-remapping table from `docs/preprocessing.md` §10), `models/scalers/old/*.pkl` (persisted `AdaptiveMinMaxScaler`), `data/databases/Processed_multi_stacked_1.db` (live feature DB), `data/data_history/*.txt` (raw history dumps).
- `#2_small_37/` exists as a sibling folder but is currently empty/placeholder.

### `training_models/tools_&_notes/`
- `reset_folders.py` (22 lines) — trivial cleanup utility (duplicate of the one in `model_concepts/lionheart_spiking_nn/`).
- `quantum states_testing.py` (47 lines) — a standalone, pure-`numpy` scratch file (`phi`, `state_alpha`, `state_beta`, `psi`, `symbolic_x`) exploring the quantum-mechanics-flavored math that shows up in the `Qg_MLP`/`QgMLPBlock` naming — read this if the "quantum" layer names elsewhere confused you, it's the underlying idea in isolation.
- `fractals_gpt.txt` — not code, a text note; the graph tooling in this repo flagged it as conceptually related to the `FractalDenseLayer`/fractal-gated ideas used in `ASTAF_fractals_MHA` and `SNAKEMLP` — worth a skim if you're tracing that idea across files.

---

## 8. Non-code items you'll see but can ignore for Python reading

- `model_training/data/*.db`, `*.pkl` — the actual dataset and a fitted KMeans model; binary data, not scripts.
- `graphify-out/` — output of an automated code-analysis/knowledge-graph tool (`graph.html`, `graph.json`, `GRAPH_REPORT.md`) run over this repo. Not part of the project itself.
- `.DS_Store` / `._*` files — macOS filesystem metadata, not project files.
- `.claude/` — this AI assistant's local settings.

## 9. If you want a "spot the pattern" exercise

Because so much of this repo is the same handful of building blocks re-combined, a good fluency exercise once you've read a handful of files: search across the whole `model_training/` tree for one recurring class name and read every definition of it back to back, e.g.

```bash
grep -rn "class SpatialGatingUnit" model_training/
grep -rn "class SpikingNeuronLayer" model_training/
grep -rn "class RealTimeDashboard" model_training/
```

Watching the *same* class slowly gain methods and complexity version over version is a genuinely good way to see how a codebase (and a single author's Python style) evolves.
