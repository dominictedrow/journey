# TREND02 — first competition result (2026-10-30)

**Verdict on development data: fails.** No variant has a positive edge **even before costs**, in any era, on either instrument. Holdout not touched.

## Setup
- Mode `trend02` v0.1.0 (`src/trader/modes/trend.py`), ADAPTATION. Votes: sign(EMA8 − EMA32) per speed (15m, 60m, 4h, as EMAs on 15-min bars with spans ×1/×4/×16). Ensemble enters on a unanimous vote and exits when the net vote stops favouring the position (reversing if the opposite side qualifies). Catastrophe stop 3 × ATR(15m) × √(slowest speed). Flat from Friday 15:45 ET.
- Pre-registered variants (fixed before the first run): unanimous ensemble, plus each speed alone.
- Data: EURUSD 2004-01 → 2025-09; S&P CFD 2012-01 → 2025-09 (D-020 caveats). Costs ASSUMED (`instruments.py`).
- Sources #6/#1/#2/#10 are not yet methods-reviewed. Their evidence is for **daily-to-monthly** trend, so this tests an intraday extrapolation, not the published claim.

## Results (net P&L, dev period)
| Variant | EURUSD 0× cost, $1M | EURUSD 1×, $300k | S&P 0× cost, $1M | S&P 1×, $300k |
|---|---|---|---|---|
| trend02 (ensemble) | −70k (Sharpe −0.15) | −56k (−0.46) | −19k (−0.07) | −37k (−0.60) |
| trend02_15m | −354k (−0.14) | −292k (−1.68) | −211k (−0.13) | −276k (−2.53) |
| trend02_60m | −93k (−0.07) | −128k (−0.53) | −85k (−0.14) | −101k (−0.82) |
| trend02_4h | −68k (−0.12) | −45k (−0.30) | +0.8k (0.01) | −25k (−0.36) |
| always_long (control) | +5.5k (0.03) | −0.6k (0.00) | +81k (0.57) | +23k (0.63) |

- **By era (ensemble, 1×, $300k):** EURUSD 2004–11 −0.021R, 2012–19 −0.024R, 2020+ −0.020R. S&P 2012–19 −0.048R, 2020+ −0.009R. No regime where it works.
- At $25k (proposed) every variant also loses. On the S&P most entries are vetoed for minimum size (D-021).

## Interpretation
- With zero costs, intraday EMA trend signals are slightly **negative** (≈ −0.006R per trade for 15m). That points to mild intraday mean reversion in both markets, and REV01 tests it directly.
- Costs make it worse in proportion to turnover: 15m makes ~1,000 trades a year and is wiped out.
- This does **not** test the published trend-following claim (multi-week to multi-month horizons). That belongs to `TSM01`, which is daily/monthly and needs a broader universe to be meaningful.

**Recommendation:** shelve TREND02 (intraday). Do not spend the holdout on it. Count it as a falsified hypothesis in the multiple-testing log. If slower trend matters, register `TSM01` separately (daily bars, weeks-to-months lookbacks).
