# HANDOFF P4-S12 — 2026-10-20

## Done this session (same Claude Code session as S07–S11)
- D-026 (Dom): V-novel kept as defined (option a).
- `scripts/pool_prior.py` -> flat `pool_v1/{fkey.npy u64, cid.npy u32, level.npy i8, npl.npy f16}` (~40 s), manifest['prior'].
  `casmi.features.formula.formula_key` (uint64 equality key of an int16[10] count vector).
- `Library.scores()` split out of `Library.rank()` (same output; tests 20 passed).
- `casmi/retrieval/pool.py`: `Pool` (mmap flat arrays, `window`), `formula_score`, `rank` (tier 1 = library structures
  with cosine >= COS_T, then formula agreement, curated NP source, COCONUT level, lowest CID).
- `scripts/eval_pool.py` -> `reports/eval_pool.json`. Library excludes V-novel; every val structure loses its train
  src bit in the pool (test kind-2 molecules are not train structures).

## Results / numbers
- Weighted 0.292 at COS_T 0 (V-lib 0.877, V-pool 0.0003, V-scaf 0.0013, V-novel 0.0000). COS_T 0.5 -> V-lib 0.828.
- Why kind-2 is ~0: the truth's formula is in the pool row 100% and is the top query formula 66% of the time, but a
  window holds a median 3,841 same-formula isomers (of ~6.7k). V-pool truths are mostly PubChem with high CIDs, so
  the CID/NP prior can't separate them. Only predicted fingerprints (P5) can.
- P3's V-pool 0.039 was optimistic: its library pool held the V-pool structures themselves (~50 candidates per
  window). On the honest test analog, P3 and P4 are equal; a P4 submission should score ~0.150.

## Decisions & approvals
- D-026.

## Traps discovered
- Heredocs inside `wsl -e bash -lc '...'` lose every single quote. Write Python files with the Write tool.
- `bash scripts/stop.sh <pat>` in the same wsl command as other text containing <pat> kills the outer shell; run it alone.

## Background jobs
- none

## Next session starts with
- If pending gate P4-skip is approved: PLAN P4.5 marked (rerouted D-027), tag p4-pool, start P5.1. If rejected: P4.4
  upload (pool flat arrays + bins ~4.5 GB plus SMILES lookup by row) and submit.

## Open questions for Dom
- Gate P4-skip (see STATE pending_gates).
