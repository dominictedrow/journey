# HANDOFF P4-S10 — 2026-10-19

## Done this session (same Claude Code session as S07–S09)
- D-024 (Dom: "full"): pool = train + COCONUT 2.0 + LOTUS + PubChem filtered; PLAN P4.2 marked rerouted (fps only for train/COCONUT/LOTUS; PubChem fps per ppm window at inference).
- `scripts/download_pool.sh` -> `data/external/`: coconut/coconut_csv-10-2026.zip (248 MB; 700 MB CSV, cols incl. canonical_smiles, annotation_level, np_likeness, collections), lotus/260413_frozen_metadata.csv.gz (90 MB, Zenodo 19360665 v11), pubchem/CID-SMILES.gz, CID-Mass.gz (1.4 GB; unused so far).
- `scripts/build_pool.py`: `ingest <train|lotus|coconut|pubchem>` -> `data/processed/pool_v1/raw/<src>_<chunk>.parquet` (20k rows/chunk, sorted by mass, resumable: done chunks skipped); `merge` -> `pool_v1/pool/bin_<lo>.parquet` (25 Da bins; dedup on tautomer ik14; SMILES priority train > coconut > lotus > pubchem; cols ik14, smiles, mass (RDKit exact), formula, src bitmask 1/2/4/8, cid min, level max, np_likeness max) + `pool_v1/manifest.json` (counts by source, sha256). Filters: all must parse; PubChem only: single component, net charge 0, no isotopes, elements in EL, 150–1,200 Da.
- `scripts/run_pool.sh`: train -> lotus -> coconut -> pubchem -> merge. `scripts/bg.sh` now also runs .sh scripts.

## Results / numbers
- Tautomer ik14 cost: PubChem mid-file sample 2.1 ms/mol (plain InChIKey 0.17 ms; tautomer != standard ik14 for 3.6%). LOTUS NPs: median 1.6 ms, mean 36 ms, p99 645 ms (top 5% = 65% of time).
- Estimate: LOTUS+COCONUT ~45 min, PubChem ~4–5 h on 14 procs.
- Train ingest: 277,566/277,566 kept, ik14 reused from P2.1 (3 s).

## Decisions & approvals
- D-024 (Dom). Agent: kept full tautomer canonicalization (must match scorer; no enumerator caps).

## Traps discovered
- First draft applied PubChem filters to all sources and dropped ~570 charged/multi-component train structures ([M]+ answers). Fixed: PubChem-only.
- WSL kills a setsid/nohup child if the wsl -e call returns instantly; bg.sh now sleeps 5 s (first pool launch died this way).
- `timeout` on a foreground smoke run killed LOTUS before any 50k chunk finished; NP tautomer tail is heavy, so use small chunks.

## Background jobs
- PID 1018 `scripts/run_pool.sh` -> `logs/pool_v1.log`, started ~17:45 2026-10-19, expect ~6 h. Done when the log ends with `ALL DONE`. Progress lines: `<src>: N in, M kept (Ts)`.
- Check: `wsl -e bash -lc "tail -n 5 ~/casmi/logs/pool_v1.log; pgrep -f build_pool | wc -l"`
- If it died: just relaunch `scripts/bg.sh pool_v1 scripts/run_pool.sh` (resumes from finished chunks).

## Next session starts with
- Confirm ALL DONE, record counts by source from `pool_v1/manifest.json`. Then P4.2 remainder: packed Morgan fp for rows with src & 7 (train/COCONUT/LOTUS), mass index. Then P4.3 recall report (V-pool in-window share + candidate counts at 5/10 ppm).

## Open questions for Dom
- none
