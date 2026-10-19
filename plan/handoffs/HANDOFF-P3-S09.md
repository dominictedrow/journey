# HANDOFF P3-S09 — 2026-10-19

## Done this session (same Claude Code session as S07/S08)
- P2.7 + P2 exit: `tests/test_parity.py` (200 spectra, all fields) passes; tag `p2-data`.
- P3.1 `casmi/retrieval/library.py`: `bin_rows` (0.02 Da bins, L2-normalized CSR), `Library(root, ref_splits, exclude_ik14)` (cached `data/processed/v1/cache/library_<hash>.npz`, 1.2 GB) / `Library.load(npz)`; `rank(q, mass, ppm=5, top=25)`: candidates in ppm window (one SMILES per ik14), score = mean over query spectra of max cosine to candidate's refs, tie -> mass closeness.
- P3.3 `scripts/eval_baseline.py` (truth via structures ik14, same rule as metric).
- P3.2 `casmi/kaggle/`: `predict.py` (`predict`, `validate`, `load_test`; features on the fly via clean_peaks/neutral_mass; fallback windows 5/20/100 ppm then 'C'), `notebook.py` (kernel script; finds files by glob under KAGGLE_INPUT; pip installs bundled rdkit; copies code to /kaggle/working because numba cache needs a writable dir), `build_bundle.py` (`python -m casmi.kaggle.build_bundle` -> `data/kaggle/{bundle,kernel}`; submission library = train + all val dirs, nothing excluded). `scripts/score_dryrun.py` (known-answer check), `scripts/kaggle_wait*.sh` (dataset / kernel / submission pollers).
- P3.4: dataset `dominictedrow/casmi26-bundle` v1 (private), kernel `dominictedrow/casmi26-submit` v1, Submission #1. Tag `p3-baseline`.

## Results / numbers
- Val MRR@25: V-lib 0.878 (top1 0.801, in-window 0.996, median 46 cands) · V-pool 0.039 · V-scaf 0.035 · V-novel 0 · weighted 0.306.
- Local dry run: 400/400 valid, 2.3 s, known-answer MRR 1.000. Kaggle run: Python 3.13.15, 35 s (library load 18 s).
- **Public LB 0.150** (ref 57412603). Prize line 0.474, top 0.495, public notebooks 0.42–0.45. Implies roughly 15–20% of hidden molecules are library hits at our V-lib quality; the rest needs P4/P5.

## Decisions & approvals
- D-023: gates 7 + 6 approved by Dom (dataset upload, Submission #1).

## Traps discovered
- PowerShell eats `$` inside `wsl -e bash -lc "..."` (`$PATH`, `$!`, `$(...)`, sed `$`). Put loops/vars in a .sh script or use the Edit tool.
- `kaggle datasets status` returns 403 for a few minutes after create; `scripts/kaggle_wait.sh` retries.
- Kaggle extracts `--dir-mode zip` directories (code/, wheels/) automatically.
- `casmi.metric` imports RDKit at import time; inference code must not import it.

## Background jobs
- none

## Next session starts with
- P4.1 🛑 gate: candidate-pool sources + size estimate (proposal posted to Dom in-session; re-ask if unanswered).

## Open questions for Dom
- P4.1 source choice (see pending_gates).
