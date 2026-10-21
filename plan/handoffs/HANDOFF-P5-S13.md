# HANDOFF P5-S13 — 2026-10-21

## Done this session (same Claude Code session as S07–S12)
- D-027 (Dom): P4.4 upload / P4.5 submission skipped; tag `p4-pool`.
- P5.1: `casmi/models/encoder.py` (Encoder 29.0M: peak tokens = MLP[sinusoid m/z, sinusoid neutral loss, z-scored
  peakfeat + sqrt inten, loss_id embedding]; precursor token = MLP[sinusoid neutral mass, top-16 formulas + scores,
  CE, iso] + adduct/instrument/polarity embeddings; pre-LN 8x512; attention pooling over a molecule's spectra;
  fp head (focal BCE) + formula head (w 0.1)). Profile gate: 2,077 spectra/s, data-wait 0.5%, GPU ~95%, 6.7 GB.
- `scripts/train_encoder.py` (state.pt every 1k + auto-resume, `--libs`, `--init`), `scripts/train_retry.sh`.
- P5.2 `p5_pre`: 50k steps, all libs, 2.1 h. P5.3 `p5_ft`: init p5_pre, enveda libs 2,3, lr 1e-4, 20k steps.
- `scripts/val_candidates.py` -> `pool_v1/valcand/` (fixed val windows, 18.1M rows + fp). `scripts/eval_encoder.py`.
- Inference: `casmi/data/batch.py` (raw rows -> loader-identical batch; `tests/test_batch.py`),
  `casmi.retrieval.pool.gather`, `casmi.kaggle.predict.predict_encoder`, notebook switches on bundle model.pt,
  `build_bundle.py [ckpt]` also stages `data/kaggle/pool` (hard links) for dataset `casmi26-pool`.

## Results / numbers (ranking MRR@25, honest val, a = b = 0; priors only hurt)
| ckpt | V-lib | V-pool | V-scaf | V-novel | weighted | val Tanimoto (thr 0.5) |
|---|---|---|---|---|---|---|
| P4.5 no model | 0.877 | 0.0003 | 0.0013 | 0.0000 | 0.292 | — |
| p5_pre 50k | 0.877 | 0.0271 | 0.0257 | 0.0271 | 0.310 | 0.477 (V-pool) |
| p5_ft 20k | 0.877 | 0.0286 | 0.0264 | 0.0279 | 0.311 | 0.505 (V-pool) |
- Kaggle dry run (simulated /kaggle/input, bundle 0c8e6df + p5_ft): 400/400 valid, 92 s total (candidates 67 s).
- Expected LB: ~0.150 library part + ~0.83 x 0.028 ≈ 0.17.

## Traps discovered
- Whole-model `torch.compile` dynamic -> Inductor tiling assert (torch 2.11); compile `enc` only. Ckpt keys carry
  `_orig_mod.`; all loaders strip it.
- Single DataLoader iterator ends at epoch end -> endless generator. WSL CUDA "unknown error" once -> resume/retry.
- Never run a GPU eval next to training: 12 GB fills, WSL spills to shared memory, both crawl.
- `wsl --cd` with ~ paths fails; kill by PID instead of stop.sh when the pattern is in the command line.

## Background jobs
- none

## Next session starts with
- Pending gate (STATE): 7 = create dataset casmi26-pool (6.1 GB) + version casmi26-bundle (model.pt); 6 = submit
  kernel. Commands: `kaggle datasets create -p data/kaggle/pool --dir-mode zip`, `kaggle datasets version -p
  data/kaggle/bundle --dir-mode zip -m "P5 encoder"`, `kaggle kernels push -p data/kaggle/kernel`, then poll with
  scripts/kaggle_wait*.sh and submit. Log LEADERBOARD, tag p5-encoder.
- Then P6 (contrastive retrieval + hard negatives): kind-2 MRR 0.028 is the bottleneck.
