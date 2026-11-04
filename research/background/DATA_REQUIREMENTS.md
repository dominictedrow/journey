# Data requirements for Phases B2–B4 (derived from plan v10, 2026-10-19)

Owner is sourcing the data. This document lists what every planned mode needs so one purchase/download covers the whole competition. Verify each source's license allows personal research/backtesting use before downloading.

## What each mode needs
| Mode | Instrument | Bar resolution | Volume needed? | Bid/ask needed? | Notes |
|---|---|---|---|---|---|
| IMOM01 | MES (ES history) | 1-min | No (helpful) | Helps cost model | Needs exact prices at 16:00 ET prior day, 15:30 and 16:00 ET today |
| TREND02 | MES + EUR/USD | 15m / 60m / 4h | No | FX yes | All built from 1-min |
| REV01 | MES + EUR/USD | 5m / 15m | **Yes** (VWAP) | FX yes | Spot FX has only tick volume, so EUR/USD VWAP is a proxy |
| J-T01 | MES + EUR/USD | 1h/4h zones, 1-min entries | Discussed | FX yes | |
| J-T02 | MES + EUR/USD | 1h pattern, 1m/5m entries | Discussed | FX yes | |
| J-T03 | MES + EUR/USD | 1-min | **Yes** (VWAP) | FX yes | Heikin-Ashi is derived, not downloaded |
| Baselines | both | daily/1-min | No | FX yes | |

**Result:** a single **1-minute** dataset per instrument covers every mode. Higher timeframes are resampled from it.

## Must-have (blocks B2)
### 1. S&P 500 futures, 1-minute OHLCV
- **Contract:** ES and/or MES. MES only exists from May 2019, so use **ES for long history** (same index; prices are identical in points) and MES contract specs for costs/sizing.
- **Fields:** timestamp, open, high, low, close, **volume**.
- **Per-contract files** (e.g. ESZ2024) are strongly preferred over a pre-stitched continuous series. The bot does its own rolling and keeps unadjusted executable prices. If only continuous data is available, the roll method (back-adjusted? ratio? roll date rule?) must be stated.
- **Session:** full electronic (Globex ~23h) session. Regular hours can be cut from it, not the other way round.
- **History:** minimum **2015 → present**; ideal **2010 → present**. Must include 2018 Q4, the 2020 crash, and the 2022 bear market.

### 2. EUR/USD spot, 1-minute bid **and** ask
- **Fields:** timestamp, bid OHLC, ask OHLC (or mid OHLC + spread), tick volume.
- Bid-only or mid-only data overstates profits for 1-minute modes. Separate bid and ask are the most important FX requirement.
- **History:** same span as futures (2015+ minimum).

### Required for both
- **Timezone stated explicitly** (UTC ideal). Watch out for "EST without DST" files.
- **Bar timestamp convention stated:** does 09:30 mean the bar *starting* or *ending* at 09:30? An off-by-one-bar error here leaks future data.
- Format: CSV or Parquet; one file per contract/year is fine.

## Nice-to-have (improves realism; not blocking)
- **ES/MES tick or top-of-book quote sample** (even a few months) to calibrate slippage and spread instead of assuming 1 tick.
- **Actual broker cost numbers:** MES commission + exchange/NFA fees per side; OANDA EUR/USD typical spread by hour. These are numbers to note, not files.
- **EUR/USD financing/swap rates**, only if any mode holds FX overnight.

## Handled by the bot (no download needed)
- CME and FX trading calendars, holidays, early closes and DST, via an open-source calendar library (`exchange_calendars` / `pandas_market_calendars`; to be pinned in B1).
- Contract specs: MES $5/pt, 0.25-pt tick = $1.25; ES $50/pt.
- Resampled bars, VWAP, Heikin-Ashi and all indicators.

## Later / blocked (not needed now)
- S&P 500 option dealer net-gamma (IMOM01-G): OptionMetrics/SqueezeMetrics, paid.
- MNQ/NQ, GBP/USD, rates/DXY for XMARKET01: only if that mode is approved.
- Real-time streaming feed: only before Stage 06 paper trading.

## Candidate sources to evaluate (owner checks current pricing and terms; none endorsed or purchased)
- **Futures:** Databento (CME-licensed, per-contract 1-min OHLCV, pay-per-use); FirstRate Data; Kibot; Portara/CQG.
- **FX:** Dukascopy historical data (bid and ask, tick/1-min); HistData.com (free 1-min, but **bid-only and fixed EST without DST**, so it needs a spread model and timezone fix); OANDA candles API (bid/ask/mid, needs an account, which is not yet authorized).

## Holdout policy
The most recent **~12 months** of whatever is delivered is sealed as the final untouched holdout before any mode is tuned. For IMOM01, June 2020 onward is also post-sample relative to the source paper.
