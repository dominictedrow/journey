# HANDOFF P2-S07 — 2026-10-19

## Done this session
- P2.5 shard writer `scripts/build_shards.py` (pass 1: 14 workers, one per row group -> unsorted temp memmaps in `data/processed/v1/_tmp/`; pass 2: per split, scatter into molecule order, sha256 into `manifest.json['shards']`). `--smoke N` writes `data/processed/_smoke` instead.
  - Output dirs `data/processed/v1/{train,v_lib,v_pool,v_scaf,v_novel}/`, all `.npy` (load with `np.load(..., mmap_mode='r')`):
    - per spectrum: `mz f32[N,128]`, `inten f16[N,128]`, `peakfeat f16[N,128,15]`, `npeaks u16`, `prec f64`, `neutral_mass f64`, `ce f32[N,2]` (min,max eV; NaN if none), `iso f32[N,6]`, ids `adduct u16 / instrument u8 / polarity u8 / lib u8` (0 = unknown, `vocab.json` in v1), `formulas i16[N,16,10]`, `nformulas u8`, `formula_{score,ppm,iso_score,rdbe} f32[N,16]`, `src_row i32` (train.parquet row), `mol_idx i32`.
    - per molecule: `mol_offsets i64[M+1]`, `mol_fp u8[M,256]` (Morgan r2 2048 packed), `mol_formula i16[M,10]` (-1 row if element outside EL), `mols.parquet` (ik14, smiles, formula).
  - `train/` = spectrum_role 'train' (incl. v_lib non-enveda); val dirs = enveda queries of that split.
- `scripts/check_shards.py [root] [n]`: sort/offset/padding/finite checks + re-featurize n random rows vs shards. Passed on smoke.
- `scripts/bg.sh <log> <script> [args]`: detached background launcher (setsid nohup) -> `logs/<log>.log`, prints PID.

## Results / numbers
- Disk estimate: ~5.3 KB/spectrum padded -> ~12 GB final, ~24 GB peak (temp). No gate.
- Smoke (3k rows x 2 rg): 4,864 train / 51 query spectra kept; mean npeaks ~65. Drops: multimer 988, keep_row 93, prec 1, unused 3.
- Full run: rg 0–7 ~195–250 s each (~93k kept/rg); rg 8,13–15 took 1,150–1,530 s (~115k kept/rg; heavier). RAM ~8 of 12 GB, 3–4 GB available.

## Decisions & approvals
- none new. Agent-chosen implementation details: extra cached per-spectrum arrays (formulas, iso, src_row) beyond PLAN's list; train-only vocab; queries never row-filtered (mirror test); [3M]/[4M] dropped with [2M] (D-020); one representative SMILES per ik14 (first in structures.parquet).

## Traps discovered
- `wsl -e bash -lc "... &"` from PowerShell: the background child dies when the wsl call returns immediately and `$!` gets eaten. Use `scripts/bg.sh` (setsid) — verified it survives.
- PowerShell can't take multi-line `python -c`; write a script file instead.
- Claude Code reaps its own idle background shells under memory pressure (the build itself kept running). Don't babysit long WSL jobs from a watcher.

## Background jobs
- PID 5033: `scripts/build_shards.py` (full v1), log `logs/shards_v1.log`, started 12:35. 9/21 rg done at ~25 min; expect featurize end ~13:20–13:40, then merge (~10–20 min) + sha256. Done when the log ends with a JSON block containing `"seconds"` and `data/processed/v1/_tmp` is gone.
- Check: `wsl -e bash -lc "pgrep -f build_shards.py | wc -l; tail -n 25 ~/casmi/logs/shards_v1.log"`
- If it died: delete `data/processed/v1/{_tmp,train,v_*}` + `vocab.json` (own partial output, not a cache) and relaunch `scripts/bg.sh shards_v1 scripts/build_shards.py`; consider `Pool(12)` if OOM.

## Next session starts with
- Confirm the job finished, then `scripts/check_shards.py data/processed/v1 50` (via bg.sh if slow), record per-split spectra/molecule counts + disk size, then P2.6 loader + benchmark.

## Open questions for Dom
- none (final picks = 2, D-022; Windows copy deleted, D-021)
