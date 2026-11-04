# J-T01..J-T03 (Joovier INTERPRETATION modes): first competition result (2026-10-30)

**Verdict on development data: all six variants fail after costs.** None has a reliable gross edge either. Holdout not touched.

## Setup
- Modes `jt01`, `jt02`, `jt03` v0.1.0. Rules: `research/experiments/JOOVIER_MODE_RULES.md`. Variants pre-registered and committed before the first P&L run (4f198e6).
- Reports: `reports/competition_20261031T000419Z.md` ($25k), `reports/sensitivity_300k/competition_20261031T000805Z.md` ($300k).
- Dev period: EURUSD 2004-01 → 2025-09; S&P CFD 2012-01 → 2025-09. Assumed costs: 1.4 EURUSD pips and ≈1 MES point per round trip.
- **This tests our reading of a partial transcript, not the creator's method.**

## Results
| Variant | Trades/yr (1×, $25k) | Exp. R (1×) | 1× $25k | 1× $300k | Gross bp/trade (0×) | Gross R (0×) | Median cost / stop |
|---|---|---|---|---|---|---|---|
| jt01_fx | 633 | −0.28 | −$24.9k (−100%) | −100% | +0.10 | +0.014 | 0.22R |
| jt02_fx | 26 | −0.08 | −$2.6k (−10%) | −10% | +0.27 | +0.007 | 0.09R |
| jt03_fx | 453 | −0.31 | −$24.9k (−99%) | −100% | −0.03 | −0.003 | 0.32R |
| jt01_spx | 202 | −0.53 | −$21.0k (−84%) | −99% | −0.07 | −0.004 | 0.33R |
| jt02_spx | 10 | −0.25 | −$1.3k (−5%) | −7% | +0.87 | −0.052 | 0.17R |
| jt03_spx | 159 | −0.69 | −$20.9k (−84%) | −98% | +0.20 | +0.030 | 0.47R |

Gross columns come from a zero-cost run at $1M equity, so the min-size veto doesn't bind. Losses also hold at 2× and 3× costs.

- **J-T01 / J-T03 (1-minute):** gross edge rounds to zero, while costs take 0.2–0.5R per trade. That is the same picture as REV01: at 1-minute scale with tight stops, costs decide the outcome. J-T01 hits its stop 76% (FX) and 70% (S&P) of the time.
- **J-T02 (squeeze breakout):** the least bad, and the lowest-frequency (10–26 trades a year). Gross bp/trade is positive on both instruments, but gross R is ≈0 (FX) or negative (S&P). FX by era is −1.6 / +0.65 / +2.3 bp (2004–11 / 2012–19 / 2020+). With only 149 trades in 2020+, that rise is not evidence. 1× $25k: FX −10%, S&P −5%.
- In the ranking, `jt02` comes third on both instruments, behind `no_trade` and `always_long` and ahead of `random`. Every other J-T variant ranks below `random`.

## Caveats
- The rules are ours (INTERPRETATION). A failure doesn't refute the creator. It refutes this formalisation on this data.
- The S&P series is an index CFD and EURUSD is single-sided (D-020). Volume cues were dropped, and VWAP uses tick volume.
- Multiple-testing log: 6 variants, plus 6 REV01, 4 TREND02 and 3 IMOM01 before them. That's 19 failed variants so far.
- Before pre-registration, a timing smoke test on 2019 printed trade counts only. No rule changed after it.

## What could change the verdict (not pre-registered; each needs owner approval)
- **Real chart examples** from the course. They could correct our zone, pattern and HA definitions (01-02a is still open).
- **Real broker costs + bid/ask data (D-020).** For the 1-minute modes, costs are 10–30× the gross edge, so cheaper costs alone won't rescue them.
- **J-T02 at slower settings** (for example a 4-hour pattern) would be a newly registered hypothesis, not a retune of this one.

**Recommendation:** shelve J-T01, J-T02 and J-T03 in their current form. Keep the modes registered and tested, and keep the holdout sealed. That completes the B3 queue: every candidate mode fails after costs on this data. Before any further B4 work, the next step with the most value is real costs and bid/ask or per-contract data (D-020).
