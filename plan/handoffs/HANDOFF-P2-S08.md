# HANDOFF P2-S08 — 2026-10-19

## Done this session (continuation of S07, same Claude Code session)
- P2.5 DONE: full v1 shard build finished (`logs/shards_v1.log`); `manifest.json['shards']` has config, stats, sha256 per file. `scripts/check_shards.py data/processed/v1 50` passed for all 5 splits (`logs/check_shards_v1.log`). Format: see HANDOFF-P2-S07.
- P2.6 DONE: `casmi/data/loader.py` — `ShardDataset` / `make_loader(root, split, workers=8, spec_per_batch=256, max_spec_per_mol=16, block_mols=2048, shuffle=True, seed=0)`. Yields molecule-grouped batches: per-spectrum tensors (peaks trimmed to L = max npeaks rounded up to 8), `seg` int64[B] (spectrum -> molecule row in batch), `mol_fp` u8[M,256], `mol_formula`, `mol_idx` (row in split's mols.parquet). Contiguous block reads, block + molecule shuffle, length-sorted within block, batch order shuffled. Val: `max_spec_per_mol=None, shuffle=False`.
- `tests/test_loader.py` (2 tests; suite 19/19). `scripts/bench_loader.py [workers] [split]`.
- Env: added conda-forge `gcc_linux-64` 16.2 to `casmi` env (new packages only; python/torch/rdkit untouched) because torch.compile/Triton had no C compiler. `casmi/__init__.py` sets `CC` to it (we run the env python without activation).
- D-021 Windows copy deleted (empty folder may remain); D-022 final picks = 2.

## Results / numbers
- Shards: train 1,948,633 spectra / 265,125 mols (mean npeaks 47.8); v_lib 4,719/1,000; v_pool 9,209/2,000; v_scaf 4,586/1,000; v_novel 4,515/1,000 (val counts == frozen splits). 9.8 GB. Drops: multimer 341,155; keep_row 224,717; prec 77; unused 1,997.
- Build time 5,520 s featurize (14 procs; rg 9–18 heavy, up to 4,300 s) + 150 s merge.
- Loader (W=8, train): 24,050 spectra/s, 96 batches/s, mean B 250, mean L 94. Dummy P5 (d512, 8L, bf16, compiled): 2,454 spectra/s -> loader/model 9.8x (target >= 3x). Coupled 200 steps: data-wait 0.6% (target < 5%).

## Decisions & approvals
- D-021, D-022 (Dom). Agent: conda gcc install (not a Python/CUDA/RDKit change, so not gate 8).

## Traps discovered
- torch.compile fails with "Failed to find C compiler" unless CC is set; fixed via casmi/__init__.py (scripts must import casmi before compiling).
- Claude Code reaps its idle background watcher shells under Windows memory pressure while WSL holds ~8–12 GB. The WSL job itself survives. Check with a one-off command instead.
- Heavy row groups (9–18) are ~6–20x slower to featurize than rg 0–7; a re-run of featurize needs ~95 min.

## Background jobs
- none

## Next session starts with
- P2.7 DONE in-session: tests/test_parity.py (200 spectra: 120 train + 20 per val split; all fields incl. vocab ids) passes in 13 s. P2 closed, tag p2-data. Next: P3.1.

## Open questions for Dom
- none
