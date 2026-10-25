# HANDOFF P7-S19 — 2026-10-25

## Done this session
- D-039: Submission #7 at lam 0.02 skipped; straight to reranker.
- `scripts/frag_features.py`: `--splits v_np`; diagnostic now ranks the no-tier `blend` (Kaggle since D-034) and
  prints `lowpop` per weight (D-038). Output runs/np_p6_ft/frag100_v_np_pop0.02_src7.npz.
- `scripts/rerank_cv.py`: LightGBM LambdaRank, 5-fold CV grouped by molecule on V-np (D-040), `--nopop`; writes
  runs/<name>/rerank.txt (not used; D-041).
- PLAN P7.3 marked rerouted D-040/D-041.

## Results / numbers (np_p6_ft, V-np, src 7, top 100, blend 1,4,0,0.05 + 0.02 x pop)
- base 0.768 full / 0.706 low-pop (42 mols). truth in top-100 0.98. Fragment pass on V-np: 7 s (14 workers).
- + w x feature: expl 0.5 -> **0.795 / 0.752**; expl1 1.0 -> 0.774 / 0.748; peak_frac 0.5 -> 0.789 / 0.724;
  log_nfrag and iso hurt at every w.
- LambdaRank OOF (25..400 rounds): with pop features 0.80-0.81 full / 0.65-0.70 low-pop (learns the V-np popularity
  shortcut); --nopop 0.77-0.78 / 0.68-0.76 (noisy). Gain dominated by blend_gap (0.72-0.75).

## Decisions & approvals
- D-039, D-040 (gate 2), D-041 (rule zero: hand blend ships instead of LambdaRank). All logged.

## Traps discovered
- frag_features `truth_minus_mean` prints NaN when a molecule has -1 rows in top-K (mean over empty slice); harmless.
- Any learned reranker trained on V-np inherits its popularity bias; judge on low-pop first.

## Background jobs
- none

## Next session starts with (P7.4, one step)
- Wire D-041 into `casmi/kaggle/predict.py` P6 path: after the src-7 + P6_POP score, take top `4 * MAX_RANK` (100),
  compute `expl` exactly as `scripts/frag_features.mol_feats` col 0 (casmi.rerank.fragment.fragments + explain,
  spectrum-intensity-weighted over up to MAX_SPEC = 8 spectra, charge -1 for negative mode via vocab
  ionization_mode), add `P7_EXPL = 0.5 * expl`, re-sort. Set `P6_POP = 0.02`. Use a multiprocessing Pool (Kaggle 4 cores).
  Move `mol_feats` into casmi/rerank/ so train and Kaggle share it (parity rule); add a parity test row.
- Then the parity test, a dry run (expect +~3 min for fragments), val_as_test sanity check, and 🛑 gates 7+6 for Submission #7.

## Open questions for Dom
- none
