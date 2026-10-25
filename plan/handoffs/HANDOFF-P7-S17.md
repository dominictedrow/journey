# HANDOFF P7-S17 — 2026-10-25

## Done this session
- np_chain finished (np_p5_pre 50k -> np_p6_pre 30k -> np_p6_ft 10k). Honest V-np eval of runs/np_p6_ft/last.pt.
- `casmi/kaggle/predict.py`: `P6_POP = 0.05` adds `P6_POP * pool.popularity[lo:hi]` to the P6 blend over the whole
  window (loads popularity.npy only when P6_POP > 0). Commit 3ee59a4.
- Gates 7+6 (D-037): casmi26-bundle code-only version (bundle/code rsynced from casmi/, casmi_bundle.json sha
  3ee59a4, model.pt = p6_ft weights, verified equal), casmi26-pool version adds popularity.npy (hardlink). Kernel v7.
- `scripts/kaggle_wait_sub.sh` fixed: the kaggle "outdated version" warning is on stdout, so `sed -n 3p` read the
  separator; now greps the first numeric row.

## Results / numbers
- Honest V-np, np_p6_ft, blend 1,4,0,0.05 (top-200 rerank): truth in top-200 0.918
  | lam | 0 | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.5 |
  |---|---|---|---|---|---|---|---|
  | all cand | 0.464 | 0.617 | 0.695 | 0.720 | 0.707 | 0.650 | 0.587 |
  | curated only | 0.635 | 0.719 | 0.758 | 0.765 | 0.753 | 0.704 | 0.643 |
- Submission #6 (kernel v7, #5 + 0.05 x popularity): public LB **0.264** (+0.016 over #5 0.248). Kaggle run 693 s.
  Local dry run: 400 rows, known-answer 0.959. Parity test passed.
- **V-np predicted +0.13, LB gave +0.016.** V-np (196 curated enveda molecules) still overstates hidden-test gains.

## Decisions & approvals
- D-037 (gates 7+6) logged.

## Traps discovered
- `kaggle datasets files` pages at 200 entries; the pool has 196 shard files, so the npy files after ik14/mass are on page 2 (`--page-token`).
- `datasets status` says "ready" for the OLD version while a new one is processing; wait for `datasets list --mine` lastUpdated to change.
- The Claude Code background waiter can be killed by Windows-side memory pressure; WSL jobs keep running.

## Background jobs
- none

## Next session starts with
- Understand why V-np overstates the LB (+0.13 vs +0.016). Candidates: probe reranks the top-200 only, but Kaggle adds
  popularity over the whole window; hidden test not all curated NPs; PubChem-only rows are masked (P6_SRC 7), so the
  prior only reorders curated rows. Cheap check: re-run np_probe with popularity added before the top-200 cut (needs eval_encoder change).
- Then P7.2/P7.3: fragment features + LambdaRank reranker over (blend, popularity, fragment, formula rank), on a separate fold (gate 2).

## Open questions for Dom
- none
