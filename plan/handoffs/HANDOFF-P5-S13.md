# HANDOFF P5-S13 — 2026-10-20

## Done this session (same Claude Code session as S07–S12)
- D-027 (Dom): P4.4 upload / P4.5 submission skipped; tag `p4-pool`.
- P5.1: `casmi/models/encoder.py` (Encoder 29.0M params: peak tokens = MLP[sinusoid m/z, sinusoid neutral
  loss, z-scored peakfeat + sqrt inten, loss_id embedding]; precursor token = MLP[sinusoid neutral mass, top-16
  formulas + scores, CE, iso] + adduct/instrument/polarity embeddings; pre-LN 8x512; attention pooling over a
  molecule's spectra; fp head (focal BCE) + formula head (smooth L1 on log1p counts, w 0.1)). `tests/test_encoder.py`.
- `scripts/train_encoder.py` (bf16, fused AdamW, cosine lr, warmup 2k; only `model.enc` compiled, dynamic=True;
  state.pt every 1k steps + auto-resume; `--libs`, `--init`), `scripts/train_retry.sh` (reruns until exit 0).
- Profile gate: 2,077 spectra/s, data-wait 0.5%, GPU ~95%, 6.7 GB peak; 1 epoch (cap 16 spec/mol) ~ 5.8k steps.
- `scripts/val_candidates.py` -> `pool_v1/valcand/` (per-split windows + 18.1M union rows with fp, 4.6 GB).
- `scripts/eval_encoder.py <ckpt>`: library tier, then soft Tanimoto + a*formula + b*curated (sweep) -> `eval_pool.json`.
- Loader: `libs=` filter (enveda = ingest_lib ids 2,3; 800k train spectra) + test.

## Results / numbers
- Ranking MRR (a=b=0 best so far): step 5k V-pool 0.0092 / V-novel 0.0083; step 10k V-pool 0.0141.
  V-lib stays 0.877 (library tier). Priors (a, b > 0) hurt.

## Traps discovered
- Whole-model `torch.compile` with dynamic shapes -> Inductor tiling AssertionError (torch 2.11); compile `enc` only.
- Checkpoints from compiled `enc` carry `_orig_mod.` keys; loaders strip it.
- One DataLoader iterator ends at epoch end (StopIteration) -> endless generator in train_encoder.
- WSL CUDA "unknown error" killed run 2 at 3.4k steps (async, not reproducible) -> resume + retry wrapper.

## Background jobs
- `p5_pre`: `scripts/train_retry.sh --name p5_pre --steps 50000`, log `logs/p5_pre.log`, ckpts `runs/p5_pre/`.

## Next session starts with
- When p5_pre finishes: `python scripts/eval_encoder.py runs/p5_pre/last.pt`. Then P5.3 fine-tune on enveda:
  `train_retry.sh --name p5_ft --init runs/p5_pre/last.pt --libs 2,3 --lr 1e-4 --warmup 500 --steps 20000`
  (<4 h). Then P5.4: Kaggle inference path for the encoder + pool (pool upload = gate 7, submit = gate 6).
