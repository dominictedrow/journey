# HANDOFF P2-S03 — 2026-10-18

## Done this session
- P2.1 done: `scripts/build_structures.py` -> `data/processed/v1/structures.parquet` + `data/processed/v1/manifest.json` (key `structures`; later P2 steps add their own keys).
- Columns: smiles, ik14 (`casmi.metric.ik14`, tautomer-canonical), ik14_raw (train inchikey14), formula (CalcMolFormula), exact_mass, scaffold (Murcko, stereo stripped; '' = acyclic), n_spectra.
- Script skips if output exists (cache reuse; deleting it is a gate).

## Results / numbers
- 277,566 unique SMILES, 0 null; scan 1 s, RDKit 266 s on 14 procs; file 11.9 MB.
- unique ik14 274,195 vs raw ik14 275,810; ik14 != raw on 4,467 SMILES.
- 127,237 unique scaffolds; 14,113 acyclic. exact_mass 31.04–3399.57.
- Local test key: 1,214/1,214 answer SMILES present; 400 molecules -> 401 canonical ik14 (the known 1 molecule with 2 structures).

## Decisions & approvals
- Dom: "go" = boot from the WSL state (pasted START-HERE was the stale Windows copy). No plan change.

## Traps discovered
- The Windows copy `C:\Users\blazi\kaggle\Envenda` has a stale START-HERE (P0, session 0). Always paste `\wsl$\Ubuntu-24.04\home\dom\casmi\plan\START-HERE.md`.
- From Git Bash, `wsl ... bash -l <<'EOF'` works for multi-line commands; `nohup ... &` inside that heredoc never starts (no log). For long jobs use `wsl -d Ubuntu-24.04 -- bash -lc '...' < /dev/null` as a tracked background task. Don't combine a heredoc with `< /dev/null` (stdin overridden -> silent no-op).
- Exact mass is computed on the stereo-stripped neutral SMILES; charged SMILES (if any) keep their formula charge from CalcMolFormula — check in P2.4 when deriving neutral mass.

## Background jobs
- none

## Next session starts with
- P2.2 spectrum cleaning (isotope features first, then ppm/peak filters, top-P=128, sqrt intensity). Thresholds proposed in `reports/data_audit.md`.

## Open questions for Dom
- Confirm final-pick count (assumed 2).
- Delete the stale Windows copy `C:\Users\blazi\kaggle\Envenda`? (gate 4/8)
