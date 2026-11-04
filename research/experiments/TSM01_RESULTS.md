# TSM01 — monthly time-series momentum (2026-11-03)

**Verdict: FAIL on the pre-registered criteria for both instruments (L = 12).** Shelved. Holdout untouched.

Design: `research/experiments/TSM01_DESIGN.md` (3b3bf52). Code committed before the run (b8573dd): `src/trader/modes/tsm.py`, `scripts/tsm01_run.py`. Carry comes from FRED `DFF`/`ECBDFR`, with a 1%/yr retail FX markup and a 1.7% S&P dividend yield (both ASSUMPTIONS). Outputs: `artifacts/tsm01/`.

## A. Scale-free benchmark (10% vol target, carry + costs)
| Instrument | L | Costs | Sharpe | Ann. return | Eras |
|---|---|---|---|---|---|
| EUR/USD | **12** | 1× | **−0.09** | −0.9% | 2004–11 −0.32 · 2012–19 +0.19 · 2020+ −0.17 |
| EUR/USD | 12 | 3× | −0.34 | −3.5% | |
| EUR/USD | 3 | 1× | +0.30 | +3.1% | +0.44 · +0.30 · +0.10 |
| EUR/USD | 3 | 3× | +0.04 | +0.4% | +0.23 · +0.02 · −0.20 |
| S&P CFD | **12** | 1× | **+0.52** | +5.0% | 2013–19 +0.57 · 2020+ +0.45 |
| S&P CFD | 12 | 3× | +0.51 | +4.9% | |
| S&P CFD | 3 | 1× | +0.23 | +2.3% | +0.08 · +0.44 |

Buy-and-hold excess-return Sharpe over the same months: EUR/USD −0.09, **S&P +0.90**.

## Pass criteria (L = 12)
- **EUR/USD: FAIL.** Sharpe −0.09 (needs > 0.30), negative in 2 of 3 eras, negative at 3×. There's no 12-month trend edge.
- **S&P: FAIL on criterion 4 only.** Sharpe 0.52, positive in both eras, insensitive to costs, but well below buy-and-hold's 0.90. The signal was long in **86% of months** (12 flips in 13 years), so in a 2013–2025 bull market it's a worse version of holding the index: it gives up return when it goes short after dips that recover.

L = 3 on EUR/USD reaches exactly 0.30 at 1× but drops to 0.04 at 3× and 0.10 after 2020. It's a pre-registered secondary that can't pass on its own, and it doesn't hold up at higher costs either.

## B. Bot run (EUR/USD, PROPOSED 0.25% risk at a 3σ monthly stop)
At $25k, L=12 makes 10 trades and +$231 over 21 years. At $300k it makes 38 trades and −$1,367, with carry −$1,817. A 0.25% risk budget at a monthly-scale stop gives only about 7% notional exposure, so P&L is negligible either way, and the $25k/$300k sign difference is minimum-size and small-sample noise. **Takeaway:** under the proposed intraday risk rules, a monthly strategy can't make meaningful money even when the signal works. A long-horizon mode would need its own risk budget, which is an owner decision.

## What this tells us
1. Monthly trend on a single instrument is weak, as the papers warn: their Sharpe comes from diversifying across ~58 markets.
2. The S&P "edge" is mostly the equity risk premium (being long), not trend timing.
3. Carry matters for FX holds (−$1.8k of −$1.4k net at $300k) and is now modelled.
