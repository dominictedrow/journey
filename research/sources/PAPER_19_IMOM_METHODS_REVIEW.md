# Paper #19 methods review — Hedging Demand and Market Intraday Momentum (task 01-02b-3, IMOM01)

**Review date:** 2026-10-19. **Status:** full text read; methods, equations and result tables extracted with page references. This is a source review, **not** a replication, backtest or approval to implement.

## Citation and access
- Baltussen, G., Da, Z., Lammers, S., Martens, M. (2021). *Hedging demand and market intraday momentum.* Journal of Financial Economics 142, 377–403. DOI https://doi.org/10.1016/j.jfineco.2021.04.029
- Accessed copy: co-author-hosted PDF of the published journal version, https://www3.nd.edu/~zda/intramom.pdf (supersedes the failed SSRN delivery link in inventory #19). All 27 journal pages (377–403) present.
- **Extraction caveat:** text was machine-extracted (pypdf) because page rendering was unavailable. Prose and equations extracted cleanly; multi-column table layouts were partly scrambled. Table values quoted below were matched to row/column by reading order and should be **visually re-verified against the PDF** before use in any decision document. Page numbers below are journal page numbers.

## Exact authors' method
- **Data (p. 381):** Tick Data LLC 1-minute bars (open, close, volume; volume from 2003) for 62 futures: 17 equity index, 16 government bond, 21 commodity, 8 currency; Dec 1974–May 2020. Most liquid contract, rolled when the next contract's daily tick volume exceeds the current one. Filters: drop non-positive prices, non-business days, **early-close days**, days with < 100 contracts volume.
- **Session definition (p. 381):** "common" trading hours of the underlying cash market, not the futures' extended session. S&P 500 E-mini: **09:30–16:00 ET**. Previous-day close = price at the session's close time (16:00 ET), not the futures settlement.
- **Return definitions (pp. 381–382, Eqs. 1–4):**
  - r_ONFH = P(open+30m, t) / P(close, t−1) − 1
  - r_ROD = P(close−30m, t) / P(close, t−1) − 1 ← primary signal
  - r_M = P(close−60m, t) / P(open+30m, t) − 1
  - r_SLH = P(close−30m, t) / P(close−60m, t) − 1
  - r_LH = return over the last 30 minutes before close ← target
- **Regressions (p. 382, Eqs. 5–7, 11):** r_LH on r_ONFH; on r_ONFH + r_M + r_SLH; on r_ROD; horse race on r_ONFH + r_ROD. Pooled t-stats are clustered by time and market (Cameron et al. 2011); individual-contract t-stats are Newey–West.
- **Out-of-sample test (p. 382, Eqs. 8–10):** expanding-window forecasts, ≥ 500 observations warm-up, benchmark = recursive historical mean of r_LH; Clark–West (2007) test.
- **Trading rule (p. 385, Eq. 12):** η(r) = +r_LH if r_ROD > 0, else −r_LH. In other words, go long for the last 30 minutes if the session-to-date return is positive and short otherwise, then exit at the close. A variant η(r_ONFH, r_ROD) trades only when both signs agree (Table 6, p. 387).
- **Costs (p. 386):** results are **gross of transaction costs**. The authors state, without a reported table, that S&P 500 futures still give a positive net Sharpe ratio at a cost of **one tick**.

## Reported results most relevant to our pilot
| Item | Value (page/table) | Implication |
|---|---|---|
| ES individual, r_ROD | β = 6.18 (×100), NW t = 4.97, R² 3.41%, R²_OOS 2.29% (Table B1, p. 397) | Strongest direct support for an MES variant |
| Equity pooled, η(r_ROD) | ann. return 6.86%, vol 3.96%, SR 1.73, hit rate 0.55 (Table 6, p. 387) | Gross; ≈ 2.7 bp per trading day on unlevered notional (6.86%/252, our arithmetic, pooled 1/N portfolio, not ES alone) |
| Equity subsample 2000–2020 | β_ROD 3.98, t 6.65 (Table 5, p. 386) | Effect not confined to early sample |
| **Euro FX futures (EC)** | β_ROD −0.04, t −0.17, R²_OOS −0.07% (Table B4, p. 399) | **No support for EUR/USD** |
| Currency pooled 2000–2020 | β_ROD 0.33, t 1.45, not significant (Table 5, p. 386) | Currency effect has faded in recent decades |
| 16:00 vs 16:15 close (S&P) | r_ROD predicts r_LH to 16:00; no predictive power for 16:00–16:15 futures return (Table 12, pp. 394–395) | Exit must be at **16:00 ET**, not the futures 16:15 close |
| Reversal | Equity next 2–3 day returns load negatively on r_LH (Table 11, p. 393) | Do not hold IMOM positions overnight; possible interaction with REV01 |
| Option gamma conditioning | NGE < 0: β 6.63, t 4.78; NGE ≥ 0: β 0.82, t 1.03 (Table 7, p. 389) | Effect concentrated on negative dealer-gamma days; the NGE input needs OptionMetrics/SqueezeMetrics-type data (**paid / licensing gate**) |
| Seasonality control | Effect survives adding lagged r_LH (Table 4, p. 386) | Distinct from same-half-hour seasonality |

## Mechanism and null hypothesis
- Mechanism (pp. 377–379, 388–392): short-gamma hedgers (option market makers, leveraged-ETF rebalancers, volatility-targeting strategies) trade in the direction of the day's move, concentrated in the last 30 minutes for liquidity, overnight-risk and capital reasons. Transitory price pressure explains the later reversal.
- H0 for our test: on MES, sign(r_ROD) gives no positive after-cost expectancy for the 15:30–16:00 ET return, relative to always-long and no-trade baselines.

## Differences between the paper and our planned IMOM01 adaptation
1. **Instrument:** paper uses ES (standard contract before mini). MES launched in 2019, so pre-2019 tests would use ES prices scaled to MES costs. Contract-specific spreads and fills differ.
2. **Signal timing / leakage:** the paper earns r_LH from the same P(close−30m) used to compute the signal. Our protocol forbids entering at the same price that produced the signal, so we must add an execution lag (e.g., enter at the next bar's open or a limit fill after 15:30:00) and measure the decay.
3. **Costs:** the paper reports gross returns. At MES scale, one tick = 0.25 pt = $1.25. Round-trip spread plus commissions and exchange fees are a material fraction of a ≈ 2.7 bp gross daily edge. **Cost sensitivity at 1×/2×/3× is the decisive test.**
4. **Sample period:** the paper ends May 2020. The period from June 2020 onward has not been seen by the authors and should serve as our untouched holdout, with a check for post-publication decay (published May 2021).
5. **Calendar:** exclude early-close days as the paper does; use the 16:00 ET cash close, adjust for DST, and use the ET exchange calendar.
6. **EUR/USD:** the paper's evidence does not support applying IMOM01 to EUR/USD. Its currency session (Table 1/A4: 07:20–14:00; the text on p. 381 cites 8:21/15:00 volume spikes, implying a Central vs Eastern labeling inconsistency) is a futures pit-era convention with no natural spot-FX analogue.

## Recommended register changes (pending owner approval at Stage 01 interpretation gate)
- Restrict IMOM01 to **MES only**. Drop EUR/USD from IMOM01 scope, or keep it only as a pre-registered negative control.
- Fix the exit at 16:00 ET. Signal = r_ROD from the prior 16:00 ET close to 15:30 ET. Signal uses only completed bars, and entry is lagged.
- Track the NGE-conditioned variant as `IMOM01-G`, blocked on a data-licensing decision.
- Protective stop still required on every position (D-015), even for a 30-minute hold.

## Remaining open items for 01-02b-3
- Visually verify the table values above against the rendered PDF.
- Gao, Han, Li, Zhou (2018), *Market Intraday Momentum*, JFE, is the predecessor study (r_ONFH signal on ETFs). It has not been reviewed yet and is the natural next source for IMOM01 cross-checking.
- Reviews of the other priority papers (TREND02: #6, #1, #2, #10; REV01: #18) are not started in this session.
