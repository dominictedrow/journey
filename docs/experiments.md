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

### E013 — Track B 5-fold submission — 2026-10-12 — session (Windows box + Kaggle)
- Bundle: private dataset `knee-bundle-b` (5× `trackB_vits_mil/fold*/model_avg.pt`, mean, prior = labels_v0, staged in
  `C:/knee_cache/sub3/up`); kernel `knee-submit-b` v1 (= `notebooks/infer.py` with `BUNDLE_SLUG = "knee-bundle-b"`), T4.
  `knee-bundle` (Track A) left untouched. Stub run: 3 studies, 0 errors, 37 s total.
- Submission 57127904. **Public LB 0.860** (val mean 0.849 → +0.011; Track A 0.909 → 0.924, +0.015). (Track A folds 0+1 = 0.924, 57082082.)

### E012 — Track A folds 0+1 finished; A+B blend check; 2-fold submission — 2026-10-11 — session (Windows box + Kaggle)
- Kaggle session 3 (`knee-train` v2): fold 0 final **0.910** (ep 15; top-3 avg 0.909), fold 1 **0.907**; gold-12 0.87 / 0.94.
  Runs copied to `C:/knee_cache/trackA_runs/fold{0,1}`.
- Track B folds 3: val 0.848, gold-12 0.84 (fold 4 relaunched after a loader OOM). Folds 0–3 mean 0.847.
- Track B fold 4 (2026-10-11, 1 loader worker, 4.04 h): val **0.860** (ep 19 alone beat top-3 avg 0.8596), gold-11 0.82.
  **5-fold Track B: val mean 0.849** (0.857/0.831/0.852/0.848/0.860), gold mean 0.80. Likely cause of both fold-4 OOMs (UNVERIFIED): a
  zombie fold-2 `train_sup` (PID 17520, from 2026-10-08) still holding 13.5 GB of committed memory that taskkill
  could not free; reboot to clear it.
- **A+B rank blend on fold 0–1 OOF (labels_v0, lo/hi 0.4/0.6): B adds nothing.** w_B 0 → 0.9085 mean, 0.1 → 0.9081,
  0.3 → 0.9034, 0.5 → 0.8928; gold-24 falls likewise (0.913 → 0.889 at 0.3). Only Effusion/Synovitis gain (+0.002/+0.004 at 0.3).
- Submission 57082082: `knee-bundle` v2 (folds 0+1 `model_avg.pt`, mean), `knee-submit` v3, T4 ×2. Public LB: pending.
- Kaggle runs one GPU session at a time on this account: `knee-train-2` (folds 2+3) queued behind `knee-submit`.

### E011 — First submission: Track A fold 0 epoch 9, single model — 2026-10-10 — session (Windows box + Kaggle)
- Checkpoint: `knee-train-2` fold0 `epochs/ep009.pt` wrapped with its config (`C:/knee_cache/sub1/trackA_f0_ep9.pt`), val 0.903.
- Bundle: private dataset `knee-bundle` (wrapper folders p_src/p_w/p_wh kept by zip mode → `infer.py` `in_bundle`).
  Kernel `knee-submit` v2, T4, internet off, fp16 autocast, no compile. Kaggle image is Python 3.13: the cp312 decoder
  wheels don't install (uncompressed test DICOMs don't need them).
- **Public LB 0.915** (submission 57033689). Top of the public LB on 2026-10-10: 0.964.
- Verdict: valid baseline. Gap to the top is ~0.05.

### E010 — Track A folds 0+1, Kaggle session 2 (`knee-train-2` v1) — 2026-10-09 — session (Windows box + Kaggle)
- Resumed from the smoke output (E008) via `kernel_sources` (alternate slug); 8.5 h; 0 retries / non-finite in every epoch.
- Val macro AUC by epoch (of 16): fold 0 0.728, 0.803, 0.830, 0.850, 0.865, 0.880, 0.888, 0.897, 0.901, **0.903** (ep 9);
  fold 1 0.721 … **0.896** (ep 8). ~1.3 studies/s per T4. Gold-12 at fold 0 ep 9: 0.850.
- Session 3 pushed 2026-10-09 as `knee-train` v2 (mounts `knee-train-2`) to finish both folds.

### E009b — Track B folds 1–2 (MAE init, 2 loader workers) — 2026-10-09 — session (Windows box)
- Fold 1: val **0.831**, gold-12 0.822. Fold 2: val **0.852**, gold-12 0.804. (Fold 0: 0.857, E009.) 0 retries / non-finite.
- 2 workers (a stuck exited process held 13.5 GB commit) → 7.4 h (fold 1) and 14.7 h (fold 2: 8.9 → 1.3 studies/s mid-run).

### E009 — Track B fine-tune fold 0: MAE init vs random init — 2026-10-08 — session (Windows box)
- Track: B. `configs/trackB_vits_mil.yaml` (20 epochs, eff. batch 16, LLRD 0.75, drop path 0.1), labels_v0 / folds_v0 fold 0
  (3,525 / 882), gold = train.csv. RTX 5070 bf16, ~13 studies/s, 7.6 GB peak, ~2.1 h (MAE) / ~2.3 h (scratch). 0 retries, 0 non-finite.
- Init `encoder_final.pt` (E007; strict load, 0 missing/unexpected): **val macro AUC 0.857** (ep 19; top-3 avg 0.856), gold-12
  0.697 [0.58, 0.79]. Still creeping up at the end (0.855 → 0.857 over the last 5 epochs).
- Random-init control (`run_name=trackB_vits_mil_scratch`): **val 0.769**, gold-12 0.583 [0.44, 0.72].
- MAE buys **+0.088 val AUC** and wins on all 12 findings (largest: Fracture +0.15, MCL +0.12, Medial Meniscus +0.12, Baker's +0.12).
- Per-finding tables: `C:/knee_cache/runs/trackB_vits_mil{,_scratch}/fold0/auc_table.md`. Gold-12 is n = 12, noise.
- Verdict: **keep**. Beats Track A's 2-epoch floor (0.774, E004); the real gate comparison waits for Track A fold 0 at 16 epochs.

### E008 — Track A smoke session on Kaggle T4 ×2 (`knee-train` v1) — 2026-10-08 — session (Windows box + Kaggle)
- Track: A. Folds 0+1, one per T4, `HOURS=0.75`, fp16, ckpt off + channels_last + compile (E006), workers 1, eff. batch 24.
- 146 steps/epoch; both folds reached step 100 (loss 0.50 → 0.46 / 0.50 → 0.48), 0 non-finite, loss scale 32768,
  peak 10.3 GB. STOP saved `last.pt` (878 MB) in both folds.
- Steady state: **1.25 studies/s per GPU** (0.8 s/study vs the profile's 0.53 s). GPU util 99% (the loader keeps up); the
  T4s sit at the 70 W **power cap** (throttle reason 0x4, SW power cap), SM clock 525–1260 MHz. → ~52 min/epoch incl. val, ~14 h per fold for 16 epochs.
- Not yet tested: resume in a second session from this output, `already finished`, `val_n_retried_fp32` (no epoch finished).
- Verdict: **pass, with a speed caveat** (power-limited, not a bug). The resume check rides on the first long session.

### E007 — MAE pretraining, SliceViT-S/16 @256, 60 epochs (Track B) — 2026-10-08 — session (Windows box)
- Track: B. Config `configs/mae_vits.yaml` (dec_depth 2, mask 0.75, bs 128, blr 1.5e-4, workers 4, gpu_crop), epochs 60
  (Dom). RTX 5070, bf16, ~950 windows/s, 6.05 GB peak, 07:42 → 10:17 (2.6 h incl. probes). 0 non-finite.
- Loss 1.08 → 0.57. Probe macro AUC (labels_v0, fold 0, 1500/600): ep5 0.753, 10 0.767, 20 0.776, 30 0.781, 40 0.782,
  45–60 0.783 (flat from ~ep 40).
- Output: `C:/knee_cache/runs/mae_vits_p16_256/encoder_final.pt` (+ snapshots every 5 epochs).
- Verdict: **keep** — next is the Track B fine-tune on fold 0 from `encoder_final.pt` plus the random-init control.

### E006 — Kaggle T4 profile, step 2 (`knee-profile` v2) — 2026-10-08 — session (Windows box + Kaggle)
- Track: A/B/MAE (performance). Kaggle T4, fp16 autocast, random weights. Output: `kaggle kernels output dominictedrow/knee-profile`
  (`profile/summary.md`, `results.json`). (E005 = the MAE loader bench in `configs/mae_vits.yaml`; entry not yet written.)
- **A_train** (median ms/step, peak GB): ckpt off+cl 722 / 11.5; **ckpt off+cl+compile 527 / 9.7** (warm-up 249 s,
  2 graphs); ckpt [0,1]+cl 802 / 6.7. → Kaggle Track A uses `grad_checkpointing=false perf.channels_last=true
  perf.compile=true` (`notebooks/kaggle_train/train.py` EXTRA_SETS).
- **A_infer**: compile cuts 1027 → 740 ms (autocast+cl); half+cl+compile 707 ms at 2.0 GB; 4 studies/forward adds nothing.
- **B_train**: compile 862 → 172 ms (5×); bs4 171 ms/4 studies at 11.1 GB. channels_last alone: no gain.
- **B_infer**: compile halves it (222 → 115 ms); half ≈ autocast once compiled.
- **MAE bs128**: compile 483 → 296 ms; channels_last slower (506 ms) → keep it off.
- fp16 numerics: relative L2 vs fp32 ≤ 8.4e-4 (A, half), all finite.
- Verdict: **keep** — compile on for every T4 workload; channels_last only for Track A.

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
