# Experiments

Each experiment is **pre-registered**:
1. Write `experiments/EXP-###_<slug>/design.md` from [`experiments/_TEMPLATE.md`](experiments/_TEMPLATE.md) and commit it **before** running anything.
2. Write `results.md` after the run.
3. Add a row to this index.

Verdicts: `RETAIN` · `REJECT` · `INCONCLUSIVE` · `RUNNING` · `PLANNED`

| ID | Date | Phase | Hypothesis (one line) | Key metric(s) | Result | Verdict | Folder |
|---|---|---|---|---|---|---|---|
| EXP-001 | — | 5 | The baseline scorer (s = momentum rank, c = 0.5 + 0.3 × rank) through the rules engine sets the bar | mean rank IC, t-stat, Sharpe after costs, MDD | — | PLANNED | — |
| EXP-002 | — | 7 | Ablation: dropping each base model in turn. Keep a model only if it lowers the meta learner's out-of-sample loss | meta OOS log loss | — | PLANNED | — |
| EXP-003 | — | 7 | Meta learner on base outputs + regime only, vs the full input set | OOS log loss, IC | — | PLANNED | — |
| EXP-004 | — | 7/8 | Shared LLM-adjusted inputs vs raw inputs for MLP / CNN #1 (cost of correlated errors) | meta OOS loss, inter-model correlation | — | PLANNED | — |
| EXP-005 | — | 7 | CNN #1: 2D convolutions vs 1D-over-time fallback; feature-family row order vs correlation clustering | OOS IC | — | PLANNED | — |
| EXP-006 | — | 8 | KV-cache manipulation experiments on the homemade LLM | tokens/s, memory, quality | — | PLANNED | — |
| EXP-007 | — | 8 | LLM size sweep inside 50–125M params | extraction F1, Δ usefulness | — | PLANNED | — |
| EXP-008 | — | 3 | HY OAS proxy: `BAA10Y` vs ICE series over the overlap window (D-001) | correlation, HMM state agreement | — | PLANNED | — |
| EXP-009 | — | 10 | Price-transformer limit head vs the q70 rule | fill rate, slippage bps | — | PLANNED | — |
