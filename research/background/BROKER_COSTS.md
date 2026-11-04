# Broker cost check (D-020 part 1) — 2026-10-31

Public numbers only. No accounts were opened and nothing was bought. Third-party figures are marked as such. Re-check against the broker's own fee schedule before paper trading (Stage 06).

## MES (Micro E-mini S&P 500), per contract per side
| Component | Public figure | Source |
|---|---|---|
| Broker commission (Tradovate) | $0.39 Free plan / $0.29 Monthly ($99/mo) / $0.09 Lifetime ($1,499) | tradovate.com/pricing (official; no effective date shown) |
| CME exchange fee, non-member | ≈ $0.35–0.37 | TradeStation fee table; Stage 5 Trading PDF (Jan 2026). Third-party; CME schedule not retrieved |
| NFA assessment | $0.01–0.02 | Third-party; NFA page not retrieved |
| Clearing | not published by Tradovate; third-party all-in estimates imply ≈ $0.10–0.20 | third-party guides |
| **All-in** | **≈ $0.66 (Lifetime) to $0.96 (Free)**; third-party estimate | damnpropfirms / chartmini guides |

**Our assumption:** $0.62/side commission, plus a 1-tick spread (half-spread 0.125 pt) and 1 tick of slippage on market/stop fills. That gives ≈ $5.00 per round trip (≈ 1 point).
- The commission is at the **low end** of the realistic range. On the Free plan it is ≈ $0.34/side too low, ≈ $0.68 per round trip.
- The spread and slippage dominate (≈ $3.75 of the $5.00). One tick of spread is normal for MES in regular hours. One tick of slippage on every market and stop fill is a deliberate conservative choice, and only tick/quote data can calibrate it.
- **Verdict:** the MES cost model is in the right range, about −15% to +5% of a realistic all-in. It is not the reason the modes failed.

## EUR/USD (US retail, OANDA)
| Pricing | Spread | Commission | Round trip (pips) | Source |
|---|---|---|---|---|
| Spread-only (standard) | ≈ 1.0 pip quoted; ≈ 1.6–1.7 avg in third-party tests | none | ≈ 1.0–1.7 | fxbrokertrust, forexbrokers.com (third-party) |
| Core + commission | ≈ 0.1–0.4 pip | ≈ $5/side per 100k (≈ 0.5 pip/side; OANDA SG page says $3) | ≈ 1.1–1.4 | oanda.com/sg-en pricing; compareforexbrokers (third-party) |

OANDA's US page shows no EUR/USD spread figures. Spreads widen around the daily roll and news.

**Our assumption:** 1.0 pip spread + 0.2 pip slippage per side = **1.4 pips per round trip**. That sits inside the realistic 1.0–1.7 range. The best realistic case is ≈ 1.0–1.1 pips.

## What this means for the shelved modes
The zero-cost runs already gave the gross edge, so the question is whether *any* realistic cost could rescue a mode. None can:

| Mode | Gross edge per trade | Best-case realistic cost | Gap |
|---|---|---|---|
| REV01 fx_5m | ≈ +0.06–0.09 pip (+0.03R) | ≈ 1.0 pip (≈ 0.12R) | costs ≈ 4× the edge |
| J-T01 fx | +0.10 bp ≈ 0.1 pip (+0.014R) | ≈ 0.16R | > 10× |
| J-T03 spx | +0.03R | ≈ 0.45R (MES all-in is already ≈ our assumption) | ≈ 15× |
| IMOM01 | +0.7 bp/day | ≈ 2.5–2.8 bp | ≈ 4× |

**Conclusion:** the assumed costs are realistic, not inflated. Lower broker costs cannot rescue any shelved mode. The remaining data question is different: **is the gross signal itself real?** Two data gaps could be distorting it:
1. **EUR/USD is single-sided (bid or mid, unknown).** Bid-only bars can create fake short-horizon reversion and mis-time stops. A matching **ask** series measures the true spread by hour and fills stops on the correct side.
2. **The S&P series is an index CFD, not ES/MES.** Its overnight coverage, roll-free path and tick volume differ from the traded futures.

## Recommended data route (owner action; nothing downloaded)
1. **EUR/USD ask side (free).** The supplied files look like Dukascopy exports (the `USA500IDXUSD` symbol name and the header format match). If so, Dom can export the **ask** side for the same symbols and dates from Dukascopy's free Historical Data Export. Then `bid + ask` gives observed spreads for EURUSD and the CFD. Check Dukascopy's terms in the export dialog first.
2. **ES per-contract 1-minute OHLCV (paid, needs approval).** Databento (CME-licensed, pay-per-use) is the cleanest option and covers both the futures-vs-CFD question and real volume. It needs an account and a purchase, so it needs Dom's explicit OK.
3. **Optional:** a few months of MES top-of-book quotes, to calibrate the 1-tick slippage assumption.

Engine work once data arrives: the loader takes an optional ask file, and the engine fills buys at the ask and sells at the bid (observed spread replaces `half_spread`). Then rerun the shelved modes unchanged as a data-quality check, not a new hypothesis.

## Sources
- Tradovate pricing: https://tradovate.com/pricing/
- OANDA SG pricing: https://www.oanda.com/sg-en/trading/our-pricing/
- OANDA core pricing sheet: https://www.oanda.com/assets/documents/566/OANDA-CC-Pricing.pdf
- TradeStation exchange fees: https://www.tradestation.com/pricing/exchange-execution-and-clearing-fees/
- Stage 5 Trading fee PDF (Jan 2026): https://stage5trading.com/wp-content/uploads/2026/01/stage-5-membership-pricing-updated-1.2026.pdf
- Optimus Futures on CME micro fees: https://community.optimusfutures.com/t/how-to-verify-official-cme-exchange-fees-for-micro-futures-mes-mnq-mgc/11570
- Third-party fee guides: https://damnpropfirms.com/trading-guides/tradovate-fees-explained-commissions-data-plans/ , https://chartmini.com/blog/futures-brokers-compared-fees-features
- OANDA third-party reviews: https://www.fxbrokertrust.com/insights/oanda-review-regulation-costs-and-platforms , https://www.forexbrokers.com/compare/forex-com-vs-oanda , https://www.compareforexbrokers.com/reviews/oanda/
- Dukascopy historical export: https://www.dukascopy.com/swiss/english/marketwatch/historical/
