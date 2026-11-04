# PORT01 — core SPY + diversified trend sleeve (2026-11-03)

**Verdict: FAIL (4 of 5 criteria missed).** It halves SPY's drawdown, but its Sharpe is below both SPY and a plain 60/40. Holdout untouched.

Design: `research/PORT01_DESIGN.md` (cde5247). Script committed before the run: `scripts/port01_run.py` (1d80ddf). Dev window 2005-11 → 2025-09. Outputs: `artifacts/port01/`.

## Results
| Portfolio | Sharpe | CAGR | Vol | Max DD | start–2012 | 2013–19 | 2020–25 |
|---|---|---|---|---|---|---|---|
| **PORT01** (60% SPY + sleeve, ≤2×) | **0.57** | 7.2% | 10.2% | **23.6%** | 0.31 | 0.94 | 0.59 |
| PORT01, 3× costs | 0.36 | 5.0% | 10.2% | 25.6% | 0.12 | 0.68 | 0.39 |
| B1 100% SPY | 0.66 | 11.0% | 15.1% | 50.8% | 0.24 | 1.21 | 0.74 |
| B2 60% SPY / 40% cash | 0.66 | 7.5% | 9.1% | 33.2% | 0.24 | 1.21 | 0.74 |
| **B3 60% SPY / 40% IEF** | **0.72** | 8.3% | 9.3% | 29.5% | 0.49 | 1.37 | 0.57 |

Gross exposure averaged 1.94×, and the 2× cap bound in 77% of months.

## Pass criteria
1. Sharpe ≥ SPY + 0.10: **no** (0.57 vs 0.76 needed).
2. Max DD ≤ 0.75 × SPY: **yes** (23.6% vs 38.1% allowed).
3. Sharpe ≥ 60/40: **no** (0.57 vs 0.72).
4. 3× Sharpe ≥ SPY: **no** (0.36).
5. Every era ≥ SPY − 0.10: **no** (2013–19: 0.94 vs 1.11 needed).

## Reading
- The sleeve does diversify: max drawdown fell from 51% to 24% at a similar CAGR per unit of vol to B2. But it **costs** return in strong equity years (2013–19), and retail leverage frictions add up: margin interest at DFF + 1.5%, no rebate on short proceeds, a 0.5% borrow fee, and turnover.
- With the 2× cap binding most of the time, the sleeve runs at about 70% of its designed size, so most of its benefit is spent on financing.
- **A plain 60/40 SPY/IEF beats everything we built,** with no leverage, no shorting and almost no trading.

## Conclusion
The strategy search ends here, with a consistent finding across nine pre-registered tests: on the data available to this project, no active strategy beats a simple, low-cost allocation after realistic retail costs.
