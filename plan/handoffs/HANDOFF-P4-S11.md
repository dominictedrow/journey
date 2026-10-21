# HANDOFF P4-S11 — 2026-10-20

## Done this session (same Claude Code session as S07–S10)
- D-025 reroute (Dom yes): PubChem keyed by plain InChIKey14 after the first 467 chunks; rows carry `taut` (True = tautomer ik14). PLAN P4.2 marked.
- Pool build finished (`logs/pool_v1.log`, ALL DONE 00:10). One crash (KekulizeException in MolToInchiKey) -> per-molecule try/except in `scripts/build_pool.py`; resumed from chunks.
- `casmi/features/fp.py` (`fingerprints`, `FP_BITS`): shared Morgan r2/2048 packed; `scripts/build_shards.py` now imports it (inference will need it for PubChem window fps).
- `scripts/pool_index.py`: adds `fp` (binary 256, curated rows only) to each `pool_v1/pool/bin_*.parquet`; writes flat mass-sorted `pool_v1/{mass.npy f64, ik14.npy S14, src.npy u8, taut.npy bool}` + `manifest.json['index']` (bin_offsets, sha256).
- `scripts/pool_recall.py` -> `reports/pool_recall.json` (P4.3). `scripts/stop.sh <pattern>` (safe pkill from wsl -e).

## Results / numbers
- pool_v1: 90,043,950 unique keys. By source flag: train 274,203 · COCONUT 475,892 · LOTUS 150,377 · PubChem 89,957,005. fp rows 724,237. Disk: pool 2.3 GB, raw chunks 5.5 GB, flat arrays ~2.2 GB.
- PubChem: 101.2M SMILES in (after '.' text filter) -> 95.3M kept pre-dedup; ~16k/s plain key vs 1,150/s tautomer.
- Recall @5 ppm (any / external-only), median / p90 / max candidates: V-lib 0.995 / 0.937, 6,687 / 15,513 / 33,080; V-pool 1.000 / 0.972, 6,609 / 14,366 / 43,896; V-scaf 1.000 / 0.962; V-novel 1.000 / 0.976, 6,609 / 15,037 / 43,896. @10 ppm: recall +0–0.3 pt, candidates ~1.6x.

## Decisions & approvals
- D-025 (Dom).

## Traps discovered
- Edit tool rewrites drop the exec bit on .sh files; call `bash scripts/x.sh`.
- `pkill -f <pat>` inside `wsl -e bash -c "..."` kills its own shell (argv contains the pattern) -> `scripts/stop.sh`.
- polars 2.0: `allow_missing_columns` removed (`missing_columns='insert'`) and it only inserts columns the first file has; schema was backfilled instead.
- WSL VM idles out when no wsl.exe is attached, killing background jobs. Keep-alive: hidden `wsl -d Ubuntu-24.04 -e sleep infinity` started from Windows (Dom may set `vmIdleTimeout=-1` in .wslconfig later).
- Claude Code reaps its own watcher shells under Windows memory pressure; check with one-off commands.

## Background jobs
- none (keep-alive wsl process may still be running on Windows; harmless)

## Next session starts with
- P4.5 design: with ~6.6k candidates per window, rank = library cosine where refs exist, else formula filter (featurize top-k formulas; formula top5 0.962 on V-pool) + prior (src flags, COCONUT level/np_likeness, low CID). Measure V-pool/V-novel MRR with pool_v1 (V-novel: exclude only the train-source flag for its keys, keep external rows). Then P4.4 🛑 pool upload, P4.5 🛑 submit.
- Raise with Dom (gate 2 if acted on): V-novel answers are 97.6% present in external sources, so with pool_v1 V-novel measures kind 2, not kind 3.

## Open questions for Dom
- V-novel definition (see above).
