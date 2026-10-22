# LEADERBOARD.md — every Kaggle submission, in order

Local val = MRR@25 on V-lib / V-pool / V-novel / weighted. Δ is the public-LB change vs the previous best. Flag any submission where LB and val move in opposite directions (possible LB noise or overfitting).

| # | Date | Phase | Commit | Kaggle ver | Val lib | Val pool | Val novel | Val wtd | Public LB | Δ best | Runtime | Notes |
|---|------|-------|--------|-----------|---------|----------|-----------|---------|-----------|--------|---------|-------|
| 1 | 2026-10-19 | P3 | ecc658b | 1 | 0.878 | 0.039 | 0.000 | 0.306 | 0.150 | n/a (first) | 35 s (local test) | Library baseline: ±5 ppm mass window, best binned-sqrt cosine; dataset casmi26-bundle v1 |
| 2 | 2026-10-22 | P5 | 0c8e6df | 3 | 0.877 | 0.0286 | 0.0279 | 0.311 | 0.184 | +0.034 | 67 min (Kaggle, 400 mols; candidates 61 min) | Plan Submission #3 (P4 #2 skipped, D-027). Encoder p5_ft + casmi26-pool v1 + casmi26-bundle new version; V-scaf 0.0264 |
