# HANDOFF P7-S20 — 2026-10-25

## Done this session
- P7.4: D-041 hand blend wired into `casmi/kaggle/predict.py` P6 path: `P6_POP = 0.02`, `P7_EXPL = 0.5`. Rank loop split
  into score (src-7 top 100) -> fragment pass (multiprocessing Pool over molecules, `mol_feats` col 0) -> dedupe/fallback.
  Timing key `fragments` is new.
- `mol_feats` + new `spectra()` (first MAX_SPEC = 8 spectra, charge -1 via vocab ionization_mode 'negative') moved from
  scripts/frag_features.py to `casmi/rerank/fragment.py`; frag_features imports them (output bit-identical to S19).
- `tests/test_parity.py::test_fragment_feats_cached_equals_on_the_fly`: mol_feats on shard spectra == on featurize_rows. 2/2 pass.
- Commit 6ba032e. Gates 7+6 (D-042): casmi26-bundle code-only version (sha 6ba032e), kernel v9, Submission #7.

## Results / numbers
- Kaggle path on V-np (scripts/val_as_test.py runs/np_p6_ft/last.pt v_np): **0.7951** (== D-041 diagnostic 0.795).
- Local dry run (p6_ft, test.parquet): 400 rows, known-answer 0.967 (#6: 0.959); fragments 18 s on 4 workers.
- Kaggle v9: total 848 s (fragments 85 s, candidates 391 s, rank 253 s).
- Submission #7 public LB: **0.288** (+0.024 over #6 0.264). V-np +0.027 vs LB +0.024: the fragment term transfers.

## Decisions & approvals
- D-042 (gates 7+6) logged.

## Traps discovered
- Kernel pushed right after `datasets version` binds the OLD dataset version. v8 ran old code (no `fragments` timing,
  1843 s) and was not submitted. `datasets list --mine` row also changes on usabilityRating, so don't diff the row:
  compare `datasets files <slug>` creationDate (or the lastUpdated column) before `kernels push`.
- Always confirm new code ran on Kaggle via a new timing key / log line before submitting.
- Kaggle `read` time varies 72 s .. 440 s run to run (cold /kaggle/input); total 848 .. 1843 s.
- PowerShell mangles nested quotes in `wsl -e bash -lc "..."`; put multi-step commands in logs/*.sh and run `bash -l`.

## Background jobs
- none

## Next session starts with
- Tag `p7-rerank` (P7 exit) if not done, then P8.1: 🛑 gate 1 (architecture: adds a decoder) before any code.
  Write the gate per plan/CLAUDE.md: decoder size, pretrain data (pool structures), GPU-hours, alternatives.

## Open questions for Dom
- none
