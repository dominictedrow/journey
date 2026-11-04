# TSM02 — diversified trend on 24 ETFs (2026-11-03)

**Verdict: FAIL on the pre-registered criteria (2 of 4 missed), but the weakest failure so far.** The diversified trend edge exists in free data at about Sharpe 0.3, with a negative correlation to the S&P. It's below our bar and fragile to costs. Holdout untouched.

Design: `research/experiments/TSM02_DESIGN.md` (dc7f575). Script: `scripts/tsm02_run.py` (3b9d3be). **Disclosure:** the script was run once just before it was committed, because a lint failure stopped the commit step. It's committed byte-for-byte as run. Data: Yahoo daily adjusted closes, 24 ETFs, dev window 2005-11 → 2025-09 (12–24 assets live, median 24). Monthly returns: `artifacts/tsm02/monthly_returns.csv`.

## Results (after 5 bp costs and 0.5% borrow)
| Variant | Sharpe | Return at 10% vol | Corr. with SPY | start–2012 | 2013–19 | 2020–25 |
|---|---|---|---|---|---|---|
| **Primary L=12 long/short** | **+0.29** | +2.9%/yr | **−0.17** | +0.43 | +0.16 | +0.23 |
| L=12, 3× costs | +0.11 | +1.1%/yr | −0.17 | +0.31 | −0.06 | +0.05 |
| L=3 long/short | +0.24 | +2.4%/yr | −0.23 | +0.70 | −0.10 | +0.05 |
| L=12 long/flat | +0.35 | +3.5%/yr | +0.43 | +0.69 | +0.13 | +0.13 |
| L=12 equities only | +0.35 | | −0.02 | +0.49 | +0.49 | +0.03 |
| L=12 bonds only | +0.00 | | −0.23 | +0.24 | −0.30 | +0.16 |
| L=12 commodities only | +0.40 | | −0.02 | +0.55 | +0.24 | +0.37 |
| L=12 currencies only | −0.04 | | −0.21 | −0.15 | −0.05 | +0.10 |

Context: SPY excess-return Sharpe +0.66; equal-weight buy-and-hold of the universe +0.20.

## Pass criteria (primary)
1. Sharpe > 0.40: **no** (0.29).
2. Positive in every era: yes (+56% / +10% / +15%).
3. Sharpe > 0.20 at 3×: **no** (0.11).
4. Correlation with SPY < 0.30: yes (−0.17).

## Reading
- **The edge is real but modest.** It's positive in every era and negatively correlated with equities, which is the diversifying behaviour the papers describe. Its size (≈0.3 Sharpe after 2005) matches what's publicly known about trend following's weak 2009–2019 decade.
- **Costs bite through leverage.** Scaling each asset to 40% vol puts low-vol ETFs (bonds, currencies) at 3–5× notional, so 5 bp of turnover cost becomes significant. At 15 bp the 2013–19 era turns negative.
- **Commodities carry the result** (0.40). Currencies and bonds add nothing in this window. That's a post-hoc observation and isn't a basis for a new variant without fresh pre-registration.
- **Long/flat** (holdable in a cash account) has a higher Sharpe (0.35) but loses the diversification (correlation +0.43): it's mostly long risk assets.

## Conclusion
The trend-following line is closed against our pre-registered bar. The evidence says a diversified trend sleeve is a ~0.3-Sharpe diversifier, not a standalone edge. Using it as a diversifier next to a core long position is a portfolio-construction question, i.e. option (b) from the 2026-11-03 discussion, and needs owner direction.
