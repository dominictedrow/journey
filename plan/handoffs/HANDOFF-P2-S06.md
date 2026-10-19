# HANDOFF P2-S06 — 2026-10-18

## Done this session
- P2.4 feature code (all in `casmi/features/`, entry point `casmi.features.featurize(mzs, intens, prec, adduct)` -> dict; P2.5 and Kaggle inference must call only this):
  - `formula.py`: adduct parser (generic `[nM±…]z±`, Na/K/Li/Ca), `neutral_mass`, numba `decompose` (sorted N/P/S/halogen core table + C/O/H loops, 10 ppm, Seven-Golden-Rules ratios, RDBE parity, SENIOR), `iso_score` (D-018), `prior_score`, `rank_formulas` (top 16).
  - `peaks.py`: per-peak `FEATS` (15): md, kmd, loss_md, loss_id (26-entry `LOSSES` table), pair_loss (top-16 peaks), is_prec, subformula of top formula (hit, ppm, C/H/N/O/S+P/X counts, rdbe). Precise masses stay in mz/prec, not peakfeat (float16).
  - `formula_prior.npz` (committed, 2 KB): train-split-only log-priors on N/P/S/F/Cl/Br/I counts, RDBE, H/C, O/C. Rebuild: `scripts/build_formula_prior.py`.
- `tests/test_features.py` (4 tests; suite 17/17). `scripts/eval_formula.py [n] [split]`, `scripts/bench_features.py [n]`.

## Results / numbers
- Formula rank (true formula, enveda spectra): train-split sweep chose sigma 2 ppm, w_iso 0 -> top1 0.707 / top5 0.944 / top16 0.999. V-pool (all 9,209): top1 0.747 / top5 0.962 / top16 0.996 / recall 1.000.
- Priors: top1 0.43 -> 0.63 (sigma 5); sigma 5 -> 2: +0.07–0.11. Isotope ratio score: no gain (-0.01..+0.005); kept as feature `formula_iso_score`, weight 0 in ranking.
- featurize: test 0.66 ms mean (p99 1.5); train first 5k 3.5 ms mean, p50 0.61, p99 14.8, max 2.8 s (huge masses). No-formula 0.4% train / 0% test. Peaks with subformula: test 86%, train 70%.

## Decisions & approvals
- none new. Implementation choices (agent): ranking = ppm Gaussian + element/RDBE/H-C/O-C priors; fragments annotated as [f+H]+ / [f-H]- only.

## Traps discovered
- Git Bash heredocs with nested quoted heredocs/quotes break; write files via `\\wsl$\Ubuntu-24.04\home\dom\casmi\...` with the Write tool, run via `wsl -d Ubuntu-24.04 -- bash -lc '...' < /dev/null`.
- [3M…]/[4M…] adducts (~50 rows) exist besides [2M…]; treated as covered by D-020 (multimers) — drop with them in P2.5.
- [M]+ / [M]- (charged molecules) need `require_even=False`; featurize does this via adduct delta == 0.
- Na/K adduct fragments are not annotated (ponytail note in peaks.py).

## Background jobs
- none

## Next session starts with
- P2.5 shard writer: estimate disk first (🛑 if > 40 GB). Use `featurize`, `keep_row`, `spectrum_role`/`in_pool`, drop multimers (D-020). Parallelize over row groups (~14 procs, watch 12 GB RAM); train mean 3.5 ms/spectrum -> ~10 min.

## Open questions for Dom
- Final-pick count (assumed 2)? Delete stale Windows copy?
