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

## Later in S20 (after #7): P7.5 gap analysis + Submission #8
- D-043: P8 deferred, P7.5 gap analysis added. Findings in reports/P7.5-gap-analysis.md (public code pulled to
  data/kaggle/public/, gitignored; read only, never executed).
- D-044 / #8 (015da35, kernel v10): PubChem-only side list interleaved at slots 4/8/12/16/20 unless library cosine
  >= 0.9. tests/test_interleave.py. Health keys pc_mols / pc_lib_blocked in timings. **LB 0.292 (+0.004 = noise).**
- scripts/val_as_test.py v_np now excludes v_np structures from the library (the train shard still holds them; lib
  cosine was 1.0 for 98%). With that: V-np lib_max median 0.40 (1% >= 0.9) vs in-library test 1.0. Old V-np numbers
  were not inflated (truths are curated, so the lib flag added nothing).
- Read: the public lineage's PubChem gain comes with a far stronger PubChem ranker (FPNet A+B + fold-safe ensemble,
  z-scored, 0.15 x pop, top-200-documented shortlist, promotion rule). Our blend can't put answers into 5 slots.

## Decisions & approvals
- D-042 (gates 7+6), D-043 (rule zero: P8 deferred, P7.5), D-044 (gates 7+6, #8) logged.

## Traps discovered
- Kernel pushed right after `datasets version` binds the OLD dataset version. v8 ran old code (no `fragments` timing,
  1843 s) and was not submitted. `datasets list --mine` row also changes on usabilityRating, so don't diff the row:
  compare `datasets files <slug>` creationDate (or the lastUpdated column) before `kernels push`.
- Always confirm new code ran on Kaggle via a new timing key / log line before submitting.
- Kaggle `read` time varies 72 s .. 440 s run to run (cold /kaggle/input); total 848 .. 1843 s.
- A pushed kernel runs on the VISIBLE test (copy of train), so every molecule is a library hit (pc_lib_blocked 400).
  Health checks must accept that; the hidden test only runs on submission.
- PowerShell mangles nested quotes in `wsl -e bash -lc "..."`; put multi-step commands in logs/*.sh and run `bash -l`.

## Background jobs
- none

## Next session starts with
- 🛑 Strategic gate (pending_gates): (a) keep our pipeline and port levers (next: a strong fingerprint ranker for the
  PubChem list / ICEBERG-GLACIER same-formula rerank), or (b) rebase on the public 0.467 notebook (licence check of
  its ~12-16 datasets first) and add our validated pieces as extra signals. Re-ask; nothing else until answered.

## Open questions for Dom
- none
