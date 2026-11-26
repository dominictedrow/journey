# Model Plan: changes to MASTER_PLAN for Dom's model descriptions

> **Status: accepted by Dom 2026-11-26 (D-041) and applied to `MASTER_PLAN.md`.**
> Source: [`docs/spec/model_descriptions_v1.md`](../spec/model_descriptions_v1.md) (Dom, 2026-11-26). Scope: every model except the execution algo and the rules algo (Phases 5 and 6 are untouched).
> Method: the same two-step process as every phase. Each item in Dom's description becomes a question in `docs/interrogation/07–10`, which are asked one at a time, and the build specs are written only from his answers. Nothing is added beyond his description except common ML conventions (batching, early stopping, seeds, calibration, model registry), and each of those is marked as one.

## 1. Analysis of the current plan
- **Done or running:** Phases 0–2 and Phase 3 (build step 5a is in progress, uncommitted).
- **Pipeline phases still to build:** 4 (labels + walk-forward harness), 5 (rules engine + baseline), 6 (executor on paper). These are the "pipeline structure".
- **Current model phases:** 7 (MLP, CNN #1, meta learner), 8 (news pipeline + SVM + LLM), 9 (Transformer, CNN #2), 10 (price transformer).
- **Problems with the current order once Dom's description applies:**
  1. **The LLM comes after the numeric models.** In Dom's description, every vote model and the meta learner read the LLM's output, so the LLM must be built first. (Before news exists, its output is Δ = 0 with `news_available = 0`, as in the spec.)
  2. **Phase 8 mixes structure with models.** Dedup, SimHash and the quota are pipeline structure. They should be done before any model, so "pipeline complete" really is complete.
  3. **No shared training framework.** Dom's "General" section applies to every model (mixed precision, Adam, clipping, dropout, augmentation, k-fold, fused ops). Built once, it keeps every later model consistent. Built per model, the models drift apart.
  4. **The Market Transformer is new.** It overlaps the old Phase 10 price transformer and the executor.
  5. **Real training needs the RTX 5070,** but Phase 12 (Windows) is last, and D-018 keeps the Mac (8 GB RAM) to smoke tests.

## 2. Proposed phase table
Phase numbers 0–6, 11 and 12 are unchanged, so every existing reference stays valid. Phases 7–10 are re-scoped.

| # | Component | Main deliverables | Exit gate |
|---|---|---|---|
| 4 | Labels + validation harness | As before, **plus** settling Dom's "k-fold" against the no-random-splits rule (new Q8) | As before |
| 5 | Rules engine + baseline | Unchanged | Unchanged |
| 6 | Executor (Alpaca paper) | Unchanged | Unchanged |
| **7** | **Pipeline completion + shared model framework** | News dedup/SimHash/quota (moved from 8). Model-slot contracts and tables. A **stub** in every model slot. The shared trainer (`models/train/`) that applies the "General" rules. Model registry. Transform-registry hook | The full daily cycle runs on paper with stubs in every slot, and the shared trainer trains a toy model on CPU, MPS and CUDA code paths with identical config |
| *(12?)* | *Windows migration, possibly moved here* | *See Q7.4* | |
| **8** | **News models**: 8a tokenizer + SVM, 8b LLM (2-expert) | Tokenizer, relevance classifier + `article_relevance` table, LLM (language expert + price-modification expert), paged KV cache, Δ adjustment, injection defenses | SVM and LLM validation metrics logged; Δ clip and risk-event tests pass; the LLM stub is replaced |
| **9** | **Vote models**: 9a MLP + CNN #1, 9b Transformer + CNN #2, 9c data-transform EXPs | Four vote models on the shared trainer. 9b is gated on 6+ months of news. 9c runs Dom's transform list as pre-registered EXPs | Each model is kept only if its OOS IC beats the baseline (EXP-001) |
| **10** | **10a meta learner, 10b Market Transformer** | Meta learner on out-of-fold votes. Market Transformer (its role is settled in Q10.8) | Meta beats the best single model OOS; the Market Transformer beats q70 on measured slippage (EXP-009) |
| 11 | Go-live gate | Unchanged | Unchanged |
| 12 | Windows migration | Unchanged (unless moved up) | Unchanged |

**Ordering rule (Dom):** no model is built until Phases 4–7 pass, so the pipeline structure is complete first.

## 3. Questioning files (the question algo)
| File | Covers | # Qs |
|---|---|---|
| `docs/interrogation/04_labels_validation.md` | + Q8 "k-fold" | 1 new |
| `docs/interrogation/07_model_scaffolding.md` | Pipeline completion, vote format, hardware timing, every "General" rule, carried conventions, transform hook | 19 |
| `docs/interrogation/08_news_models.md` | Tokenizer, SVM, LLM | 18 |
| `docs/interrogation/09_vote_models.md` | MLP, CNN #1, Transformer, CNN #2, data transforms | 16 |
| `docs/interrogation/10_meta_market.md` | Meta learner, Market Transformer | 12 |

The old unanswered questions in 07–10 are merged into these, not lost (calibration, budget, registry, SimHash, corpus, nano-gpt reuse, news_shock, taxonomy, L2 data, D-011, promotion rule).

## 4. Where Dom's description meets a hard rule or the spec (asked, not decided)
| Item | Conflict | Question |
|---|---|---|
| "k-fold on all models" | CLAUDE.md: no random splits; walk-forward with purge + embargo only | 04 Q8 |
| "vote on bullish or bearish" | Spec: P(up/flat/down) per asset; labels have a 0 class | 07 Q3 |
| Mixed precision everywhere | CLAUDE.md: fp32 on MPS | 07 Q6 |
| TP / PP | One GPU, so neither is meaningful | 07 Q8 |
| Speculative decoding everywhere | Only token-generating models (the LLM) | 07 Q13, 08 Q17 |
| "16 GB of RAM" | The Mac has 8 GB; the PC's RAM is unknown | 07 Q4 |
| SVM with Kaiming / ReLU / MSE | That describes a neural net, not an SVM | 08 Q2 |
| "Append a valid tag to each article" | Raw tables are append-only | 08 Q5 |
| LLM 16 × 256 (~25M) | Spec said 50–125M | 08 Q10 |
| PagedAttention (LLM, meta learner) | Paging only exists for a generation-time KV cache | 08 Q11, 10 Q2 |
| 2-expert MoE where one feeds the other | That makes it a two-stage model, not routing | 08 Q8 |
| SPP on a fixed-size table | SPP exists for variable sizes | 09 Q5 |
| Meta output dim 500 | Membership changes; survivorship; no crypto | 10 Q3 |
| Meta CE loss | Spec: score *s* (regression) + conviction *c* | 10 Q4 |
| Market Transformer outputs (price, ticker, side) | Only the rules engine creates orders; no free stock L2 | 10 Q8, Q9 |

## 5. What the plan does not add
Nothing outside Dom's text except these conventions, each one listed as a question so Dom can strike it: batching / DataLoader, early stopping, seeds, calibration, model registry, stubs (needed for "pipeline complete"), `article_relevance` table (needed by the append-only rule).

## 6. Research notes (three research agents, 2026-11-26)
Citations are as reported. Items marked *unverified* came from the agents' background knowledge, not a fetched source.

### 6.1 Validation and labels
- **k-fold:** only purged, time-ordered schemes comply. Translate "k-fold" as K sequential walk-forward folds with purge + embargo; that is the OOF stacking set and the final evaluation. Use **CPCV** (López de Prado, AFML ch. 7/12; N=6, k=2 → 15 splits, 5 paths) only for model selection and overfitting diagnostics (PBO, deflated Sharpe), because it trains on the future for some folds. Shuffled k-fold is banned. *(from memory)*
- **0-class:** 3-class softmax is the most faithful. Dropping 0 biases toward volatile samples; a soft 0.5 target blurs the signal. Sample weights for label overlap (AFML ch. 4) matter more than the 0-class choice.
- **Augmentation:** evidence for return prediction is thin (Wen et al. 2021 survey). The label-safe set is jitter, mixup with soft labels, and block bootstrap, on training rows only, with labels recomputed on changed paths. Time warping breaks the 5-day barrier.

### 6.2 Training infrastructure (1 × RTX 5070 sm_120, MPS)
| Item | 5070 | MPS | Use |
|---|---|---|---|
| torch.compile | Yes, via `triton-windows` (officially supports RTX 50xx) | Prototype (*unverified*) | CUDA only, behind a config flag, eager fallback, parity test |
| Fused Adam(W) | `fused=True` | Default `foreach` | Per device in config |
| bf16 autocast | Yes, no GradScaler | Keep fp32 (rule) | Per device in config |
| TP / PP | Not meaningful on 1 device | No | PP only as a learning EXP (micro-batched stages, parity check) |
| flash-attn package | sm_120 unreliable (open build issues) | No | Use `F.scaled_dot_product_attention`; benchmark its flash/cuDNN/efficient backends once |
| PagedAttention | Inference KV cache only (FlexAttention paged helpers, experimental, CUDA) | No | LLM generation + EXP-006 only |
| Speculative decoding | Autoregressive generation only | — | LLM EXP only |
| Memory | Weights + Adam state for 50M params < 1 GB; **activations dominate** | — | bf16, batch/accumulation, checkpointing on OOM, float16 memmap data. bitsandbytes/Liger not worth it |
| Init | `kaiming_normal_` suits ReLU; `xavier_normal_` suits tanh/sigmoid. PyTorch defaults: Linear = kaiming_uniform(a=√5), MHA in_proj = xavier_uniform | — | As Dom wrote, in config; GPT-2 residual-projection scaling worth noting for the 16-layer LLM |
| 2-expert MoE | A fixed route or a serial A→B is not MoE routing; a learned top-1 + aux balancing loss is, but expect no gain at this size | — | 08 Q8 |
| Attention Residuals | Kimi Team, arXiv 2603.15031 (2026-03): softmax attention over earlier layers' outputs replaces the fixed residual sum; Block AttnRes (~8 blocks) recovers most of the gain; extra activation memory | Yes | 08 Q12; proven only at 3B-active scale |

### 6.3 Model-specific
- **SVM:** a real SVM is LinearSVC (squared hinge). Kaiming + ReLU means a neural net; an "SVM head" (Tang 2013, L2-SVM) is the coherent hybrid. MSE on 0/1 works but trains worse than BCE/hinge. Fit vocab/IDF inside each training window.
- **SPP** (He et al. 2014) adds nothing for a fixed-size table; global pooling gives the same fixed vector. Features have no spatial adjacency unless ordered (EXP-005).
- **Meta 500 slots:** slot instability, survivorship, dead/new slots, no weight sharing. Shared per-asset scorer + attention across the day's assets is the standard pattern (e.g. MASTER, AAAI 2024 *(from memory)*).
- **Order book:** Alpaca stocks are top-of-book quotes only (IEX on free; no depth channel; checked in Alpaca docs). Crypto has a live `orderbook` stream; no free historical L2 found for either, so record crypto L2 ourselves from now if wanted.

### 6.4 Data transforms (ranked)
- **Test first:** quantile / rank-gauss and multi-frequency periodic encodings (Gorishniy et al., NeurIPS 2022, arXiv 2203.05556), the defensible form of the circle idea; a full circle wraps the tails together, so use half a circle or several frequencies. Ratio / anchor scaling (vintage-correct GDP, SPX/NDX), soft per-regime normalization with filtered HMM probabilities. OHLC chart images (Jiang, Kelly & Xiu, JF 2023).
- **Cheap look:** word-order shuffle as a regulariser; GAF/MTF/recurrence plots; STFT/CWT spectrograms of returns.
- **Falsification only:** Fibonacci levels vs equally spaced placebo levels.
- **Likely dead ends:** colour-splotch canvas (no locality), sonification into a pretrained audio encoder, letters-only text, Mandarin pivot.
- **Back-translation** (Sennrich 2016) is the one text transform with support. Free local MT: opus-mt (~75M, CC-BY), M2M100 (MIT), NLLB-600M (CC-BY-NC; fine for personal research). *(licenses from memory; check model cards)*
- **Rules for every transform:** fit inside the training fold, use `released_at`-correct vintages, write a no-lookahead test, compare against z-score with the same seeds, and correct for multiple testing (deflated Sharpe).

## 7. New experiments to pre-register (on acceptance)
EXP-010 PP on one device · EXP-011 2-expert MoE (learned router) vs dense · EXP-012 Attention Residuals vs standard residual · EXP-013 SPP vs global pooling · EXP-014 speculative decoding on the LLM · EXP-015+ one per data transform Dom keeps in 09 Q15.

## 8. Steps to apply this plan (after Dom accepts)
1. Rewrite the Phases 7–10 rows in `MASTER_PLAN.md` from §2, and decide whether Phase 12 moves up (07 Q4).
2. Mark D-041 accepted.
3. Add the EXP rows from §7 to `EXPERIMENTS.md` as `PLANNED`.
4. Run Phase 4 questioning (with Q8) as normal, next after Phase 3 finishes.
