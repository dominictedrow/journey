# REV01 — first competition result (2026-10-30)

**Verdict on development data: fails after costs.** EURUSD shows a small *gross* reversion edge that has grown since 2012. Costs are ~5× larger than that edge. Holdout not touched.

## Setup
- Mode `rev01` v0.1.0 (`src/trader/modes/reversion.py`), ADAPTATION of source #18 (monthly stock reversal → intraday). Fade |z| ≥ 2 vs a 20-bar SMA or VWAP proxy. Exit on reversion to the mean, after 12 bars, or at a stop 1.5σ beyond entry. Kaufman ER > 0.3 skips trending markets. Sessions: S&P 09:45–15:30 ET (flat 15:55); EURUSD 03:00–12:00 ET (flat 16:00).
- Six variants (5m SMA, 15m SMA, 15m VWAP per instrument) were **committed before the first real-data run** (07db1ec). REV01 was queued after TREND02 lost gross of costs, so it counts as an informed test (multiple-testing log).

## Results (dev period: EURUSD 2004-01 → 2025-09; S&P 2012-01 → 2025-09)
| Variant | Trades (0×) | Gross bp/trade | Gross R/trade | 0× $1M net | 1× $25k | 1× $300k |
|---|---|---|---|---|---|---|
| rev01_fx_5m | 40,799 | +0.08 | +0.029 | +$15.2M (compounded at the leverage cap) | −99% | −100% |
| rev01_fx_15m | 19,745 | −0.06 | — | +$1.46M | −98% | −99% |
| rev01_fx_15m_vwap | 17,656 | −0.05 | — | +$1.37M | −97% | −98% |
| rev01_spx_5m | 14,759 | +0.06 | — | +$0.49M | −83% | −98% |
| rev01_spx_15m | 7,154 | −0.63 | — | −$0.38M | −72% | −94% |
| rev01_spx_15m_vwap | 5,575 | −0.47 | — | −$0.11M | −65% | −90% |

- **Costs vs edge (fx_5m):** round trip 1.4 pips (ASSUMED) against a median stop of 8 pips, so costs are ≈0.17R per trade (≈0.35R in the tightest quartile). Gross edge is +0.03R overall, +0.07R in the tightest-stop quartile.
- **Why the zero-cost dollars look big:** sizing gives tight-stop trades large positions, up to the 20× leverage cap, and gains compound. Per-trade edge is close to zero. Max drawdown is 83% even at zero cost.
- **Over time (fx_5m gross bp/trade):** 2004–11 −0.30, 2012–19 +0.26, 2020+ +0.37. Gross P&L is negative in every year 2008–2011.
- Over half of all trades exit at the stop: a 2σ stretch usually keeps going.
- The S&P has no consistent gross edge (5m +0.26 bp in 2012–19, −0.20 bp in 2020+).

## Caveat that matters here
EURUSD is a single-sided price series (D-020). Quote noise in single-sided 1-minute data can create **spurious** short-horizon reversion. Without bid/ask data we cannot tell whether the post-2012 gross edge is tradable.

## What could change the verdict (none pre-registered; each needs owner approval)
- **Passive execution (EXEC01):** a reversion strategy that *earns* the spread with limit orders instead of paying it could flip the cost sign. Testing that honestly needs **real bid/ask data** for fill modelling (D-020).
- **Real broker costs:** 1.4 pips is a guess. Even 0.3 pips round trip would still exceed the +0.06-pip gross edge.

**Recommendation:** shelve REV01 as a market-order strategy. Keep the holdout sealed. Note EURUSD short-horizon reversion as the one gross signal found so far; it's worth revisiting only with bid/ask data and a passive-execution model.
