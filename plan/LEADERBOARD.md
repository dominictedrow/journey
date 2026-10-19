# LEADERBOARD.md — every Kaggle submission, in order

Local val = MRR@25 on V-lib / V-pool / V-novel / weighted. Δ is the public-LB change vs the previous best. Flag any submission where LB and val move in opposite directions (possible LB noise or overfitting).

| # | Date | Phase | Commit | Kaggle ver | Val lib | Val pool | Val novel | Val wtd | Public LB | Δ best | Runtime | Notes |
|---|------|-------|--------|-----------|---------|----------|-----------|---------|-----------|--------|---------|-------|
| 1 | 2026-10-19 | P3 | ecc658b | 1 | 0.878 | 0.039 | 0.000 | 0.306 | 0.150 | n/a (first) | 35 s (local test) | Library baseline: ±5 ppm mass window, best binned-sqrt cosine; dataset casmi26-bundle v1 |
