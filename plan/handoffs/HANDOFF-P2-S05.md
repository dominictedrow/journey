# HANDOFF P2-S05 — 2026-10-18

## Done this session
- P2.3 gate approved (D-019); [2M…] drop approved (D-020, PLAN P2.2 edited).
- `scripts/build_splits.py` -> `data/processed/v1/splits.parquet` (ik14, split, scaffold, n_enveda, n_other; [2M] excluded from counts) + manifest key `splits`. Skips if output exists (frozen).
- `casmi/data/splits.py`: `spectrum_role(split, is_enveda)` -> train/query/None; `in_pool(split)` (False only for v_novel). P2.5 must use these.

## Results / numbers
- 271,029 ik14 (excl. [2M]-only). enveda 180,008; enveda & other libs only 2,509 (V-lib ceiling); other-only 91,021.
- Molecules: train 265,989 · v_pool 2,000 · v_scaf 1,000 (570 scaffolds, each ≤20 mols) · v_lib 1,000 · v_novel 1,000 · held 40.
- Enveda query spectra (pre-clean): v_pool 9,209 · v_scaf 4,586 · v_lib 4,719 · v_novel 4,515. Enveda train spectra 800,516.
- Invariants checked: unique ik14, no scaffold shared between v_scaf/held and other splits, v_lib has both sources, v_pool/v_novel enveda-only.

## Decisions & approvals
- D-019, D-020 (see DECISIONS.md).

## Traps discovered
- Enveda-180 molecules are almost all ring systems (15 acyclic of 180k) and almost all absent from other libs: kind-1 (library match) val is small.
- n_enveda counts are before `keep_row`; a few val molecules may lose all queries after cleaning — P2.5 reports real query counts.

## Background jobs
- none

## Next session starts with
- P2.4 physics feature cache in `casmi/features/` (neutral mass, formula decomposition, per-peak features, loss pairs, isotope-ratio score). ≤10 ms/spectrum.

## Open questions for Dom
- Final-pick count (assumed 2)? Delete stale Windows copy?
