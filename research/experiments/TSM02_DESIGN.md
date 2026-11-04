# TSM02: diversified time-series momentum on 24 ETFs (design + pre-registration)

**Status:** pre-registered 2026-11-03 before any TSM02 return was computed. **Research-only:** it tests whether the published diversified trend edge exists in data we can get free. It isn't a bot mode, and trading it would be a scope change (D-029).
**Sources:** #1 Moskowitz, Ooi & Pedersen (2012), #2 AQR century of trend, #10 CFM two centuries. **Provenance: ADAPTATION.** Their portfolio is 58 futures; ours is 24 ETFs.
**Why:** TSM01 (D-028) found single-market monthly trend weak, which matches the papers: their Sharpe comes from diversification.

## Universe (fixed; Yahoo daily adjusted close, total return including dividends and fund costs)
- **Equities (7):** SPY, IWM, EFA, EWJ, EWG, EWU, EEM
- **Bonds (5):** IEF, TLT, LQD, TIP, BWX
- **Commodities (6):** GLD, SLV, USO, UNG, DBA, DBB
- **Currencies (6):** FXE, FXY, FXB, FXA, FXC, FXF

Data QA (2026-11-03): no duplicates; the largest daily moves all match real events; gaps ≤ 7 days (2001-09). ETFs give investable returns with the roll and fees already in them. Yahoo's continuous futures were rejected because of unadjusted roll jumps.

## Rules (MOP 2012, adapted)
- **Monthly**, on the last trading day of each calendar month. Each asset joins once it has 12 months of history plus a 60-day vol warm-up.
- **Excess return** = daily total return − cash, where cash = FRED `DFF`/360 per calendar day.
- **Signal:** the sign of the asset's past 12-month cumulative excess return (primary, L = 12).
- **Vol:** EWMA std of daily excess returns, centre of mass 60, annualised ×√261.
- **Position:** `w_i = sign_i × 40% / σ_i`. Portfolio = **equal-weighted average** of `w_i × next-month excess return` over the assets available that month, then scaled to **10% annual vol ex ante** by a constant factor. The scale changes no Sharpe ratio.
- **Costs:** 5 bp one-way per unit of turnover (`|Δw|` at the asset level) (ASSUMPTION; liquid ETFs). **Short borrow:** 0.5%/yr on short notional (ASSUMPTION).

## Variants
- **Primary (decides):** L = 12, long/short.
- **Secondary (reported only):** L = 3 long/short; L = 12 **long/flat** (shorts replaced by cash, which is what a cash account could actually hold); and by asset class (four sub-portfolios).

## Windows
- **Dev:** first full month with ≥ 12 assets live → 2025-09-30. **Holdout sealed from 2025-10-01**, not loaded.
- **Eras:** start–2012, 2013–2019, 2020–2025-09.

## Pass criteria (primary, after costs and borrow)
1. Sharpe > **0.40** over the dev window.
2. Positive total return in **every** era.
3. Sharpe > **0.20** at **3×** costs and borrow.
4. Monthly correlation with SPY excess returns < **0.30** (a real diversifier, not equity beta).

Passing means the edge exists in free data. It would justify a separate owner decision about scope (instruments, account type, a risk budget for long-horizon modes) and a one-shot holdout check. Failing closes the trend-following line for this project.

## Multiple testing
This is the 7th hypothesis family. The primary uses source parameters; three secondaries are pre-registered and labelled.
