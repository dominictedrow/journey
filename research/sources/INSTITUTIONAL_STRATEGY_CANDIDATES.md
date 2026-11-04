# Research-Driven Strategy Candidate Register (v8)

**Status:** Research hypotheses ONLY; none approved for implementation, none validated, none can trigger an order. Owner authorized strategy research but material strategy additions/implementation scope must be approved at the Stage 01 interpretation gate. Parent document: `INSTITUTIONAL_PAPER_INVENTORY.md`.

## Competing alpha families and orthogonal overlays

| ID | Testable direction and rule sketch (research proposal, not author's exact published rule) | Universe / resolution | Source IDs | Main falsification |
|---|---|---|---|---|
| IMOM01 | Session-to-final-30m continuation: use return since prior regular-session close up to cutoff, enter aligned for final 30m, flat by close; strict cutoff/no leakage. | MES 1m/5m (EURUSD not supported by source: EC β_ROD t=−0.17, Table B4; see `PAPER_19_IMOM_METHODS_REVIEW.md`) | 19 | No edge net spread/commissions, day-hour effect or stale signals; exit 16:00 ET not 16:15 |
| TREND02 | Multi-speed trend ensemble: EMA fast-minus-slow direction at 15m, 60m, 4h; separate speeds rather than optimally tuned blend. | MES and EURUSD | 6, 1, 2, 10 | Noise, whipsaw, turnover after cost |
| REV01 | Short-horizon mean-reversion: standardized return or price distance from rolling VWAP/mean exceeds threshold, fade with volatility stop and time stop; avoid known trend regimes. **Adaptation, not literal Robeco algorithm**. | MES/EURUSD 5m/15m | 18 | Continued trend, bid/ask bounce, selection bias |
| XMARKET01 | Cross-market lead-lag: lagged ES/NQ or rates/USD features predict MES/EURUSD; all features must be observable before decision; controls include own-return baseline. | MES with MNQ proxies initially offline | 20, 19 | Spurious synchronization, data costs, multiple testing |
| FLOW01 | Signed order-flow / imbalance impact on FX; must acquire authentic bid/ask or trade-direction data, otherwise classify as proxy-only and not test claimed mechanism. | EURUSD conditional | 22 corrected | Provider supplies only midquotes, no signed aggregated interdealer flow |
| FXFACTOR01 | FX factor ensemble (carry, momentum, value) as slow positioning filter, never infer 5m direction from monthly valuation factors. | EURUSD plus expanded universe | 3,12,13 | No cross-section, financing vs price-only P&L confusion |
| TSM01 | 1-12 month own-return sign with realized-vol targeting as long-horizon **research benchmark**; test monthly first; optionally investigate shorter horizon as new hypothesis. | futures/FX continuous histories | 1,2,10 | Time-scale extrapolation and rollover distortion |
| XSECTION01 | Rank multiple comparable markets and long top/short bottom with risk neutrality; **blocked** until sufficiently broad liquid instrument universe authorized. | Expanded universe | 3,15,21 | Two symbols cannot support stable ranking |
| REGIME01 | Conditional volatility-based strategy selection: compare fixed allocation vs pre-registered volatility bins/market state; avoid future-volatility features. | MES/EURUSD | 4,7,8 | Overfit regime gate, high transition churn |
| TAIL01 | Examine convex response in sustained market dislocations and beta-timing contribution, **not** guarantee of crash protection. | MES/FX portfolio | 11,16 | Short abrupt crash, protection cost, historical-tail sample scarcity |
| EXEC01 | Execution experiment: market vs passive limits, fill probability, bid/ask slippage and time-outs with adverse-selection penalties. Support layer, **not alpha**. | MES/EURUSD | 23,6 | Unrealistic OHLC fills and ignored unfilled orders |

## Source-to-strategy distinction
- AQR 1,2, CFM 10, and Man 6 are largely **related trend hypotheses**, not four unrelated sources of alpha.
- Momentum and reversal hypotheses can conflict; test separately first and quantify conditional correlation before ensemble design.
- CFM 11 and Fidelity 16 assess tail exposures / beta timing; don't label them independent directional signals.
- MSCI 12/13 evaluate **hedge ratios** for existing currency exposures; turning these into standalone spot-FX alpha is a new, separately falsifiable adaptation.
- Research sources 5 and 17 are directories/datasets, not trading strategies; BIS 23 is execution microstructure.

## Candidate priority queue
1. `IMOM01`, `TREND02`, `REV01`: distinct and economically grounded, feasible after intraday feed audit.
2. `REGIME01`, `EXEC01`: overlays to all alpha candidates and safety/realistic measurement.
3. `XMARKET01`, `FLOW01`: conditional on data availability and synchronized timestamp review.
4. `FXFACTOR01`, `TSM01`, `XSECTION01`, `TAIL01`: longer horizons, expanded universe, or risk-focused investigations; keep off critical path.

## Strict experiment protocol
1. Register null/alternative hypothesis, economic mechanism, information timestamp, signal pseudocode, timezone, session convention, bars, exits and maximum holding window.
2. Freeze contract specifications and continuous MES roll logic; preserve unadjusted executable prices alongside research-adjusted continuous series. FX uses actual bid/ask and financing/rollover assumptions.
3. Use chronology-preserving splits, purging and embargo for overlapping labels, separate final holdout, pre-specified parameter grid, realistic warmup and delays. Do not use contemporaneous close as an executable entry at that same close.
4. Include a no-trade and buy/hold or sensible directional baseline, matched risk budget and identical costs. Report separately futures and forex before pooling.
5. Measure after-cost returns, expectancy, profit factor, drawdown, Sharpe/Sortino (with honest sampling), turnover, trade count, exposure, tail losses, worst sessions, variance across folds and cost sensitivity (1x/2x/3x). No cherry-picked positive windows.
6. Evaluate strategies for correlation and diversification *only after* individual holdout evaluation. Preserve all rejected hypotheses and multiple-comparison count.
7. SVM vs deterministic baseline is an independent ablation. RNN only after data volume, stable SVM uplift and complexity gate justify it. All positions require independent protective stop and hard no-trade condition if minimum risk sizing is impossible.
8. Research experiment does not authorize order activation, brokerage connections, purchases or repository operations.

## Stage 01 acceptance requirements
- Each selected signal has primary source, data feasibility, precise decision boundary and 'kill criteria' documented.
- Source reviewer signs off on distinctions between reproduced paper formula vs engineering adaptation.
- Owner must approve Stage 01 hypotheses and Stage 02 change in scope before writing executable strategy implementation.
