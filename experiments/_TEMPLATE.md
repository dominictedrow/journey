# EXP-###: <title>

> Copy this file to `experiments/EXP-###_<slug>/design.md`. Fill in **Design** and commit it before running anything. Then write `results.md` from the **Results** section.

## Design (pre-registered)
- **Date / phase / session:**
- **Hypothesis:** (one falsifiable sentence)
- **Why it matters:** (which decision this result changes)
- **Data:**
  - universe snapshot(s)
  - date range
  - feature version
  - `news_available` coverage
- **Splits:**
  - walk-forward scheme (expanding; 3-month test blocks)
  - purge = label horizon
  - embargo = 5 days
- **Baseline to beat:** (e.g. EXP-001 baseline IC = …)
- **Model / config:** (path to the YAML in this folder; seeds)
- **Costs:** equities __ bps/side, crypto __ bps/side (measured or assumed)
- **Metrics:**
  - primary
  - secondary (log loss, Brier, rank IC + t-stat, Sharpe, Sortino, MDD, turnover)
- **Decision rule:** RETAIN if …; REJECT if …
- **Leak checks:**
  - no-lookahead test passes
  - out-of-fold only for stacking
  - point-in-time universe

## Results
- **Run:** git SHA, data hash, wall time, hardware
- **Numbers:** (table per fold + aggregate)
- **Plots/artifacts:** (paths under `reports/`)
- **Verdict:** RETAIN / REJECT / INCONCLUSIVE
- **What changes because of this:** (DECISIONS.md entry, config change)
- **Caveats:**
