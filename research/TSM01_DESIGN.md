# TSM01: time-series momentum, monthly (design + pre-registration)

**Status:** pre-registered 2026-11-03 before any TSM01 code or P&L run. Owner approved building it (chat, 2026-11-03).
**Sources:** #1 AQR *Time Series Momentum* (Moskowitz, Ooi, Pedersen 2012), #2 *A Century of Evidence on Trend-Following*, #10 CFM *Two Centuries of Trend Following*. **Provenance: ADAPTATION.** The papers trade 58 futures; we have two instruments, so the diversification that drives most of the published Sharpe isn't available. Single-asset TSM is weaker, and the papers report roughly 0.2–0.5 Sharpe per asset.

## Rules (fixed now)
- **Bars:** daily, 17:00 ET roll, built from the 1-minute files. Decisions on the **last daily bar of each calendar month**; fills at the next bar's open (engine convention).
- **Signal:** the sign of the past L-month return, `close[i] / close[i − 21·L] − 1`. L = **12** (primary, the paper's headline) and L = **3** (secondary). Zero or undefined → flat.
- **Volatility:** EWMA std of daily log returns with centre of mass 60 days (the paper's estimator), scaled to monthly: σ_m = σ_d·√21.
- **Rebalance:** monthly. If the signal flips, exit and enter the other way at the same fill. If it's unchanged, keep the position and move the stop *tighter only* to close ∓ 3σ_m. If flat (e.g. after a stop-out), enter on the current signal.
- **Protective stop (D-015, binding):** at entry ∓ **3σ_m** × price, stop-market. It isn't in the papers and is rarely hit at 3σ monthly; it exists only to satisfy D-015.

## Two tests
**A. Research benchmark (scale-free, tests the hypothesis).** Vectorised and monthly, not through the engine. Weight `w = sign × (10% / σ_annual)`; monthly P&L = `w × next-month return + carry − costs`. Cost per change in weight = `|Δw| × one-way cost`, in return terms. Sharpe and eras don't depend on the 10% scale. Both instruments.

**B. Bot run (can our framework trade it under the PROPOSED risk rules?).** EUR/USD through `Backtester` at $25k and $300k with 0.25% risk at the 3σ_m stop and 1×/2×/3× costs. **The S&P can't be run here:** one MES contract at a 3σ_m stop risks ≈ $3,400, which needs ≈ $1.4M of equity at 0.25% (extends D-021). That's reported as a structural finding, not tested.

## Carry (positions are held for months)
Rates are free from FRED, daily: USD `DFF` (effective fed funds) and EUR `ECBDFR` (ECB deposit rate).
- **EUR/USD:** a long accrues `(r_EUR − r_USD)/365` per calendar day on notional, a short the opposite. A **retail markup of 1.0%/yr is charged against both sides** (ASSUMPTION).
- **S&P (benchmark only):** a long accrues `(q − r_USD)/365`, with dividend yield **q = 1.7% constant** (ASSUMPTION), as a futures-equivalent excess return; a short the opposite. No markup.
- **Bot run:** carry is added to each trade's P&L after the run, from its entry/exit times and notional. Sizing isn't re-run, which is second-order at ≈ 7% notional exposure.

## Costs
EUR/USD 1.4 pips per round trip (`instruments.py`); S&P 1 MES point per round trip (≈ 1.3 bp at 7,700). The 2× and 3× cost scenarios scale trading costs **and** the retail carry markup.

## Data and windows
- **EUR/USD:** `../data/cache/EURUSD_1m.parquet` (= Dukascopy bid, D-026), from 2004-01-01, with lookback history from 2003-05.
- **S&P:** `../data/cache/USA500IDXUSD_1m.parquet` from 2012-01-16 (data hole before), so the L=12 signal starts 2013-01.
- Holdout sealed from 2025-10-01 for both.
- **Eras:** EUR/USD 2004–11, 2012–19, 2020+; S&P 2013–19, 2020+.

## Pass criteria (L = 12 decides; L = 3 is reported only and can't pass on its own)
Per instrument, benchmark A at 1× costs with carry:
1. Sharpe > **0.30** over the dev window.
2. Positive in **every** era (EUR/USD 3 eras; S&P 2 eras).
3. Sharpe > 0 at **3×** costs and markup.
4. S&P also: Sharpe ≥ buy-and-hold Sharpe − 0.15 over the same window (i.e. it isn't just a worse long-only).

EUR/USD also needs bot run B at 1× to have net P&L > 0 and E[R] > 0 at $25k.

Passing makes an instrument a holdout candidate (one shot, owner approval). Failing shelves TSM01 for that instrument. No parameter tuning in either case.

## Multiple testing
This is the 6th hypothesis family. L=12 is the source parameter. L=3 is a pre-registered secondary.
