# Experiment log

One entry per run (training, labeling, or submission). Newest at the top. A result with no entry here doesn't exist.

## Template
```
### E<NNN> — <short name> — <YYYY-MM-DD> — session S<NN>
- Track: A (pretrained) | B (scratch ViT) | L (labels) | D (data)
- Git commit: <sha>   Config: <path or inline>
- Data: cache version <vN>, labels version <vN>, folds <file>
- Hardware / wall-clock: <GPU, h>   Peak VRAM: <GB>
- Metrics: gold-58 macro AUC <x> [bootstrap 95% CI a–b]; report-label CV macro AUC <y>; per-finding table (link)
- Kaggle: public LB <z> (submission id), notebook runtime <s>
- Verdict: keep / discard / needs follow-up — <one line why>
```

## Runs

### E004 — Track A preflight before Kaggle (AB.md Part 2) — 2026-10-07 — session (Windows box)
- Track: A (correctness). Machine: Windows box, RTX 5070, bf16, `grad_checkpointing [0,1]` + channels_last, 2 loader
  workers (Dom's `prepare.py` held ~9 GB of the 16 GB RAM). Git: `37f0e69` + uncommitted tree (AB.md work).
  Config `configs/trackA_coatnet.yaml` + `--set` labels_v0 / custom folds files in `C:/knee_cache/preflight/`.
- **2.1 overfit: inconclusive, not a wiring bug.** 32 studies × 40 ep (320 steps, eff. batch 4, backbone 0.1×) → train
  AUC 0.677; 16 studies × 80 ep (640 steps, eff. batch 2, lr 3e-3, wd 0, no rank loss) → train AUC 0.681. Train mode
  and eval mode give identical outputs; `last.pt` = `model_avg.pt`. The logged loss is confidence-weighted
  (`w_<label>`, mean 0.70): the 16-study run reached 0.42 against a weighted floor of 0.243 and a weighted prior of
  0.454, i.e. it barely fit in training either. Pooled window features of different studies are 0.95–0.999 cosine-similar,
  and random-window variation within a study exceeds the variation between studies. Wiring (labels, mask, weights, loss, head)
  is covered by `tests/test_overfit_sup.py` on the same `train_sup` path.
- **2.2 small real run: weak pass.** 300 train / 150 val (fold-0 studies), 2 epochs = 24 steps: val macro AUC
  0.505 → **0.580**; LR warmup/cosine correct; `val_n_retried_fp32` = `val_n_nonfinite` = 0; best-epoch fallback picked ep 1.
- **2.3 resume: pass.** 48 studies, 24 steps; STOP after step 8 → resumed at epoch 0 batch 32; step counter and LR
  identical at all 24 steps vs an uninterrupted run; losses track within run-to-run GPU noise (which starts at step 4,
  before the stop).
- **2.4 inference = training eval: pass.** `notebooks/infer.py` on the 150 val studies' raw DICOMs (bundle from 2.2's
  `model_avg.pt`) vs the trainer's `oof.parquet`: max |diff| 3.0e-3, mean 2.6e-4, per-finding Spearman ≥ 0.9957. 87 s
  total for 150 studies on the 5070 (decode included).
- **2.5 folds: pass.** folds_v0: 0 of 4,257 report hashes span folds, no duplicate studies, per-fold prevalence within
  ±0.03 per finding, gold 12/12/11/12/11.
- Bug found and fixed: `--set optim.lr=3e-3` stayed a string (PyYAML wants `3.0e-3`); `common.parse_set_value`.
- **Full fold 0, 2 epochs (go/no-go for Kaggle, Dom's OK):** 3,525 train / 882 val, 146 steps/epoch, 3.7 studies/s,
  5.0 GB peak, ~50 min total. Val macro AUC **0.735 → 0.774** (best epoch used; top-2 average 0.755); train loss 0.51 →
  0.43; 0 non-finite. Gold-12 0.650 [0.50, 0.76] (n = 12, noise). Per finding: Synovitis 0.93, Effusion 0.84, Medial OA
  0.82 … MCL 0.70 (`C:/knee_cache/preflight/fold0_2ep/auc_table.md`). The whole cosine schedule ran in 2 epochs, so this
  is a floor, not the 16-epoch number.
- Verdict: **pass**. The pipeline is correct end to end and Track A learns on real data; cleared for the Kaggle T4
  profile + smoke session.

### E003 — Track A memory switches on the RTX 5070 (smoke + equivalence) — 2026-10-07 — session (Windows box)
- Track: A (performance/safety check; the AUCs are noise). Machine: Windows box, RTX 5070 12 GB, bf16 autocast.
- Git commit: `37f0e69` + uncommitted tree (CRLF-only diffs). Config: `configs/trackA_coatnet.yaml` with `--set` overrides
  `labels.file=<cache_root>/labels/labels_v0.parquet folds.file=.../folds_v0.csv gold.file=<data_root>/train.csv`
  (the config has them null, so every launch needs them), `data.limit_studies=160 optim.epochs=1 optim.effective_batch=8
  optim.accum=8 data.workers=4 log_every=2`, batch 1 study. Fold 0, cache v1, labels v0.
- Peak VRAM / train studies/s (run twice; the repeat matched): ckpt all stages **4.57 GB / 3.2**; ckpt [0,1] +
  channels_last **5.04 GB / 4.0 (+24%)**; ckpt off + channels_last **12.0 GB (at the ceiling) / 0.3–0.7** (48-study
  repeat; spills, unusable on 12 GB; it only fit on the T4's 15 GB at 11.3 GB). `val_n_retried_fp32` and
  `val_n_nonfinite` were 0 in every run (bf16).
- Val macro AUC after one epoch (n=31): 0.660 / 0.688 (ckpt all) vs 0.621 / 0.646 (ckpt [0,1] + cl). Noise-level;
  consistently ~0.04 apart, **not explained**; two seeds on a full fold would settle it.
- Equivalence check (real coatnet_rmlp_2_rw_384, same weights, 10 real cached slices, seeded RNG; script was in the
  scratchpad, now `tests/test_gpu_equiv.py` on the nano model): ckpt [0,1] vs all = **bit-identical** features,
  grad rel-L2 at the noise floor. channels_last = exact up to rounding with TF32 off (feat 4e-5, grad 3e-4 rel) but
  ~10% off with **TF32 on** (PyTorch's default for convs; kernel choice differs by layout; the T4 has no TF32).
  bf16, both layouts equally far from an fp32 reference (feat 5.5% vs 5.6% rel-L2), so channels_last is no worse than
  the default. bf16 gradients vs fp32 are ~50–60% off in both (synthetic loss; not compared on real training).
- Setup note: a leftover Jupyter kernel (conda env `ai`, 44 GB committed) had exhausted Windows' commit limit and
  made the dataloader workers fail with OpenBLAS allocation errors; Dom approved killing it. Keep notebooks closed
  during long runs (16 GB RAM).
- Verdict: keep `grad_checkpointing: [0, 1]` + channels_last for Track A on this box; do not turn checkpointing off
  here (Kaggle-only). Follow-ups: bigger `data.batch_size` fits (7 GB headroom, untested); compare AUC with 2 seeds.

### E002 — T4 profile of every workload — 2026-09-30 — session P02
- Track: A + B + MAE (performance only; no training, no metric). Machine: Kaggle GPU kernel `knee-profile`, Tesla T4, 411 s.
- Git commit: code at `519ca33` (knee-src), script `notebooks/kaggle_profile/gpu_profile.py`. Synthetic inputs, random weights.
- Results: `docs/plan/03_performance_plan.md` §2b. Track A train 971 ms/micro-batch (704 with ckpt off + channels_last,
  11.3 GB), Track A inference 1,067 ms / 60 windows (673 compiled), Track B train 288 ms (235 ckpt off), Track B
  inference 185 ms (96 compiled), MAE 406 ms / 128 windows (6.7 GB).
- Verdict: keep. Conv+BN fold turned off by default (slower); compile, ckpt off and channels_last are next.

### E001 — labels v0: blend of 4 public report-label sets — 2026-09-26 — session S04 part 1
- Track: L (labels). Machine: Mac (no GPU).
- Code: `python -m knee.labels.build_labels --version v0 --no-claude` (commit of S04 part 1). Readers: pilkwang,
  steven_v1, lixin, laymond (raw); all CC0-1.0; no leak (docs/labels.md §1). Blend: silence = 0.5 vote, agreement ×
  certainty weight, fills Synovitis ← Effusion and PF OA ← Medial OA.
- Output: `<cache_root>/labels/labels_v0.parquet`, `folds_v0.csv` (5 folds 882/882/881/881/881, gold 12/11/11/12/12,
  report-hash groups, no site proxy yet).
- Metrics: gold-58 macro AUC **0.889 [0.849, 0.924]** (1,000× bootstrap). Best single reader steven_v1 0.878; best
  public non-gold-tuned reference steven_v4 0.893. Leave-one-reader-out 0.888–0.890 (no reader is measurably
  needed). Fills on vs off: 0.889 vs 0.881. Silence "vote" vs "skip": 0.889 vs 0.883 (Fracture 0.817 vs 0.779,
  Lateral OA 0.821 vs 0.793). All inside the ±0.04 CI. Per-finding table: docs/labels.md §3.
- Verdict: keep as the fallback label set for S05 until v1 (with our Claude reader) exists. "vote" became the default
  because it's the convention each public set uses on its own; the gold numbers were seen before the choice (one
  global switch, not per finding).
