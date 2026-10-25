# LEADERBOARD.md — every Kaggle submission, in order

Local val = MRR@25 on V-lib / V-pool / V-novel / weighted. Δ is the public-LB change vs the previous best. Flag any submission where LB and val move in opposite directions (possible LB noise or overfitting).

| # | Date | Phase | Commit | Kaggle ver | Val lib | Val pool | Val novel | Val wtd | Public LB | Δ best | Runtime | Notes |
|---|------|-------|--------|-----------|---------|----------|-----------|---------|-----------|--------|---------|-------|
| 1 | 2026-10-19 | P3 | ecc658b | 1 | 0.878 | 0.039 | 0.000 | 0.306 | 0.150 | n/a (first) | 35 s (local test) | Library baseline: ±5 ppm mass window, best binned-sqrt cosine; dataset casmi26-bundle v1 |
| 2 | 2026-10-22 | P5 | 0c8e6df | 3 | 0.877 | 0.0286 | 0.0279 | 0.311 | 0.184 | +0.034 | 67 min (Kaggle, 400 mols; candidates 61 min) | Plan Submission #3 (P4 #2 skipped, D-027). Encoder p5_ft + casmi26-pool v1 + casmi26-bundle new version; V-scaf 0.0264 |
| 3 | 2026-10-24 | P7 | 45f6a16 | 4 | 0.841 | 0.595 | 0.580 | 0.672 | 0.184 | +0.000 | 575 s (Kaggle, 400 mols) | D-034 #4a: p5_ft, library forced only if cosine >= 0.7. ⚠ val +0.36 vs #2 but LB unchanged: val and LB disagree (no-tier gain does not transfer). V-scaf 0.565 |
| 4 | 2026-10-24 | P7 | a886d1e | 5 | 0.829 | 0.700 | 0.701 | 0.743 | 0.207 | +0.023 | 429 s (Kaggle, 400 mols) | D-034 #4b: p6_ft contrastive, no forced tier, blend 1,4,0,0.05; casmi26-bundle + casmi26-pool (adds fkey.npy) new versions. ⚠ val 0.743 vs LB 0.207. V-scaf 0.663 |
| 5 | 2026-10-24 | P7 | 0e75d1c | 6 | n/a | n/a | n/a | n/a | 0.248 | +0.041 | 1863 s (Kaggle, 400 mols; candidates 1109 s) | D-035 (a) probe: #4b with P6 ranking only train/COCONUT/LOTUS rows (P6_SRC = 7). V-sets are PubChem-only so val does not apply; confirms the NP-heavy hidden test |
| 6 | 2026-10-25 | P7 | 3ee59a4 | 7 | n/a | n/a | n/a | n/a | 0.264 | +0.016 | 693 s (Kaggle, 400 mols) | D-037: #5 + 0.05 x PubChem popularity (P6_POP); casmi26-pool adds popularity.npy. Honest V-np (np_p6_ft) curated-only 0.635 -> 0.765 |
