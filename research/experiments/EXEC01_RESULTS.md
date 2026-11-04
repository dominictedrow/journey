# EXEC01 — passive limit-order execution of REV01 EUR/USD (2026-11-03)

**Verdict: FAIL on the pre-registered criteria, in both variants, in every scenario.** Shelved for good per the design. Holdout untouched.

Design and pass criteria: `research/experiments/EXEC01_DESIGN.md` (b1dcdfe). Code committed before the run: `src/trader/backtest/passive.py`, `scripts/exec01_run.py` (d28280f). Data: Dukascopy EUR/USD 1-minute bid/ask, dev window 2004-01 → 2025-09. Ledgers: `artifacts/exec01/`.

## Results (rev01_fx_5m signals, $25k unless noted)
| Variant | Scenario | Fill rate | Trades | E[R]/trade | Return | Fwd R filled / unfilled |
|---|---|---|---|---|---|---|
| E1 | retail (≥1.0 pip) | 79% | 16,007 | −0.18 | −99% | −0.15 / **+0.60** |
| E1 | ecn_floor | 85% | 22,683 | −0.10 | −99% | −0.10 / +0.68 |
| E1 | retail 2× | 58% | 8,332 | −0.38 | −99% | |
| E1 | retail, $300k | 78% | 22,914 | −0.18 | −100% | |
| E1 | retail, through-fill (optimistic) | 95% | 40,083 | −0.04 | −96% | |
| E2 | retail | 79% | 16,886 | −0.17 | −99% | −0.14 / **+0.61** |
| E2 | ecn_floor | 85% | 21,735 | −0.11 | −99% | |
| E2 | retail, through-fill (optimistic) | 95% | 40,293 | −0.04 | −97% | |

**Pass criteria (retail, cross fill):** E[R] > 0.05 in 2012–19 and 2020+: **no** (E1 −0.23 / −0.41, E2 −0.24 / −0.42). Net > 0 at 2×: **no**. Fill rate ≥ 30% and ≥ 300 trades/yr: yes. Max DD < 25%: **no** (99%).

## Why it fails: adverse selection
"Fwd R" is the 60-minute move after each signal, in units of stop distance, in the trade's direction. It doesn't depend on fills or equity.
- **Filled** signals average −0.15R: the limit fills because price keeps going against the trade.
- **Unfilled** signals average +0.60R: the winners revert straight away and never come back to the limit.

So the limit order filters out the good trades and keeps the bad ones. Even the optimistic trade-through fill rule (95% fills) loses in every era (−0.02 to −0.07R). At ECN spreads, passive entry (−0.04 to −0.08R after 2012) does **worse** than market entry did in D-026 (+0.01R).

## Caveat
Era rows after 2011 are path-dependent: equity collapses in 2004–11, so later trades are fewer (minimum-size rejections). The adverse-selection numbers and the optimistic-fill rows, which keep ~10k trades in 2020+, aren't affected and point the same way.

## Conclusion
The EUR/USD short-horizon reversion line (REV01 → EXEC01) is closed. The gross signal is real (D-026), but neither market nor passive execution captures it at any realistic cost.
