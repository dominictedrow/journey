# IMOM01 — first competition result (2026-10-30)

**Verdict on development data: fails its kill criterion ("no edge net of spread/commissions").** Holdout not touched.

## Setup
- Mode `imom01` v0.1.0 (`src/trader/modes/imom.py`), rules per `PAPER_19_IMOM_METHODS_REVIEW.md`: r_ROD = P(15:30 ET)/P(prev 16:00 ET) − 1; long if > 0, short if < 0; enter at the first bar after 15:30 (+ optional lag); exit at the 16:00 ET open; early-close days skipped; stop = 2σ of the expected 30-min move (our adaptation; the paper has no stop).
- Data: `USA500IDXUSD` index CFD (D-020 caveat), 2012-01-16 → 2025-09-30. Holdout from 2025-10-01 sealed.
- Costs (ASSUMED): MES 1-tick spread, 1-tick slippage, $0.62/side ≈ 1.0 index point per round trip.
- Pre-registered variants: base, `entry_lag_min=5`, `require_onfh_agree=true`. All three reported.

## Raw signal, independent of the engine (scratch calculation from 1-min closes)
| Sample | Days | β×100 (paper ES: 6.18) | corr(r_ROD, r_LH) | Gross bp/day, sign(r_ROD)·r_LH | Hit rate |
|---|---|---|---|---|---|
| All 2012-01 → 2025-09 | 3,078 | 1.35 | +0.047 | +0.72 (executable opens) | 50.6% |
| Paper sample, to 2020-05 | 1,754 | 3.09 | +0.101 | +1.03 | 50.0% |
| Post-paper, 2020-06 → | 1,324 | −0.71 | −0.027 | +0.30 | 51.3% |

The effect is present in-sample but at roughly 1/3–1/2 of the paper's strength, and the regression slope is gone after publication.

## Engine reconciliation (equity $10M so sizing never binds)
| Run | Mean bp/trade | Stopped out | Median stop | Expectancy (R) |
|---|---|---|---|---|
| Zero cost, stop 10σ | +0.58 | 0% | 39 pts | −0.001 |
| 1× cost, stop 10σ | −2.24 | 0% | 39 pts | −0.033 |
| Zero cost, stop 2σ | +0.83 | 12.3% | 8 pts | +0.004 |
| 1× cost, stop 2σ | −1.98 | 12.3% | 8 pts | −0.144 |

The engine's gross matches the independent calculation. Costs of ≈2.8 bp per round trip are 3–5× the gross edge. The default 2σ stop does not create the loss, but it makes the loss look large **in R**, because a tight stop means more contracts, so costs per R rise.

## Competition (1× costs, dev period)
- $25k (proposed): imom01 −$10.5k, Sharpe −3.05; most trades vetoed as `below_min_size` (D-021).
- $300k (sensitivity only): imom01 −$180k, lag5 −$177k, onfh −$159k. Always-long baseline +$22.7k, Sharpe 0.63.
- Reports: `reports/competition_20261030T231548Z.md`, `reports/sensitivity_300k/`.

## What could change the verdict (each needs owner approval; none is pre-registered)
1. **Real costs:** with costs ≤ ~0.5 bp round trip (well below 1 tick + fees), gross +0.7 bp could survive. Unlikely for retail MES market orders.
2. **Real ES futures data:** the CFD's 15:30/16:00 prints may differ from ES (D-020).
3. **IMOM01-G (dealer gamma filter):** the paper puts the effect in negative-gamma days (β 6.63 vs 0.82). Needs paid options data.

**Recommendation:** shelve IMOM01 as a stand-alone mode. Keep it in the competition as a logged, failed hypothesis. Do not unseal the holdout for it.
