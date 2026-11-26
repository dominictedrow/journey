# Questioning: Phase 5 · Portfolio rules engine + baseline backtest

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/05_rules_engine.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Portfolio rules engine

1. **Starting NAV:** paper defaults to $100k. Should the paper account and backtests use your real planned capital?
2. **D-002:** keep "never sell same session" now that PDT is retired? (Recommended: keep it as a flag, on by default.)
3. **Anchors:** at most 2 companies have a market cap of $200B or more and pass every criterion. If fewer than 2 qualify, the budget stays in cash per the spec. OK?
4. **Supports switching:** does the "2 weeks" hold count weekly screens or trading days?
5. **Opportunity mode:** how should the 10-day restore be enforced (trim the lowest-c active names first)?
6. **Order priority ties:** within the same priority, break ties by c, or by size?
7. **Backtest fills:** use next-day open, VWAP proxy or the q70 limit simulation? Which fill model do you want for the baseline?
8. **Reason codes:** do you have a preferred taxonomy, or should I propose one?
9. Should every threshold in the spec live in `configs/rules.yaml` with exactly the spec's defaults?
