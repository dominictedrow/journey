# HANDOFF P1-S02 — 2026-10-18

## Done this session
- P0 closed: limits recorded, tag `p0-env`, pushed (main 169b886).
- P1.1–P1.3, P1.5 done: `scripts/audit.py`, `scripts/test_in_train.py`, `scripts/audit_plots.py`; `reports/data_audit.md` + 4 PNGs; `plan/COMPETITION.md` claims verified/corrected (✅ P1 marks).
- Added matplotlib to env; lock re-exported.

## Results / numbers
- Full train stream audit: 48 s, 4.8 GB peak RSS (batch 65,536, list columns included).
- Local test = copy of enveda-180 train: 1,213/1,213 spectra, 400/400 molecules (399 single ik14). Answer key at `logs/test_in_train.csv` (gitignored; regenerate in ~1 min).
- M+1 isotope peak inside MS2: train 21.2%, test 49.6%. Peaks above precursor: 33.2% / 58.2%.
- Test peaks/spectrum p50 230 vs train 42. Top-128 peaks hold 99.6% intensity (median).
- Leaderboard 2026-10-18: #1 0.495, #5 0.474.
- Proposals (in report): P=128, noise floor 0.001, |ppm| ≤20 train / ≤10 for windows, 7 test adducts first, drop [2M…] (~13.5% of train) pending ablation.

## Decisions & approvals
- Dom: OK to commit + push P0 and continue the plan (push done for P0 only).

## Traps discovered
- `kaggle competitions list` metadata calls the metric "Mean Average Precision"; scorer code is MRR@25.
- Raw train `inchikey14` ≠ tautomer-canonical ik14; 1 local-test molecule maps to 2 raw ik14s.
- Precursor m/z max is 2.4e6 and ppm error max 4.7e8: garbage rows exist, filter in P2.

## Background jobs
- none

## Next session starts with
- If Dom answered the P1.4 gate: apply the answer (PLAN.md edit + DECISIONS row if rerouted), tag `p1-audit`, then P2.1.

## Open questions for Dom
- 🛑 P1.4 isotope gate (see pending_gates in START-HERE).
- Push P1 commits to origin?
- Confirm final-pick count (assumed 2).
