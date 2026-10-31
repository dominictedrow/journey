# Joovier modes J-T01..J-T03: implemented rules (INTERPRETATION)

**Status:** pre-registered 2026-10-30, before any P&L run on real data. Code: `src/trader/modes/joovier.py`, tests: `tests/test_joovier.py`.

**Provenance: `INTERPRETATION` (D-018).** The source is a partial third-party transcript of one Joovier course (`research/JOOVIER_TRANSCRIPT_FINDINGS.md`). Nobody has watched the video or checked the charts. Every number below is ours. A loss here tests **our reading** of the rules, not the creator's method. Do not describe these modes as creator-faithful.

## Shared conventions
| Item | Rule | Why |
|---|---|---|
| Swing high / low | Bar j's high is above the k highs before it (strictly) and at least as high as the k after it. It becomes known only at bar j+k. | Avoids the usual look-ahead in swing detection. |
| Higher timeframe (HTF) | Resampled from the mode's own bars. A value is used only after its HTF bar closes. | Causal multi-timeframe. |
| Session | S&P: entries 09:45–15:30 ET, flat 15:55. EURUSD: entries 03:00–12:00 ET, flat 16:00. Weekdays only. | Same windows as REV01; the creator day-trades. |
| Fills | Next 1-minute real open (engine). Stops and targets are checked on 1-minute bars. | Heikin-Ashi prices are never used as fills. |
| Volume | Ignored. | The creator's volume cues have no threshold, and the files carry tick volume only (D-020). |
| Sizing, costs | Unchanged risk layer and assumed costs (1 MES pt; 1.4 EURUSD pips per round trip). | Same as every mode. |

## J-T01: S/R retest + swing break (`jt01`, 1-minute execution, 1-hour zones)
Transcript ≈33:35–38:33.
- **Zones:** the last 3 confirmed 1-hour swing lows (support) and highs (resistance), k = 3. Each zone is the level ± 0.25 × the 1-hour ATR(14) at confirmation.
- **Retest:** a 1-minute bar trades into a support zone without closing below it. That arms the long side for 60 bars. A close below the zone's lower edge disarms it. Entry on the touch itself is not allowed; this is our reading of "wait for the revisit, don't enter on the touch".
- **Entry:** while armed, a later 1-minute close above the last confirmed 1-minute swing high (k = 3).
- **Stop:** the lowest low since the touch (the retest swing low).
- **Target:** the nearest resistance level above the entry. Skip the trade if there isn't one, or if reward < 1.0 × risk.
- **Exit:** stop, target or day flat. Shorts mirror.

## J-T02: consolidation squeeze breakout (`jt02`, 5-minute execution, 1-hour pattern)
Transcript ≈49:44–53:21.
- **Pattern:** the last two confirmed 1-hour swing highs are falling and the last two swing lows are rising (k = 3). That gives two rejections on each line. Each line runs through its pair of pivots in bar units and is extrapolated to the hour now forming. The lines must not have crossed, and the oldest pivot must be no more than 72 hourly bars back.
- **Entry:** a 5-minute close above the upper line **and** above the last confirmed 5-minute swing high (k = 3). Our reading of "a trendline break alone is not enough". One trade per pattern.
- **Stop:** the last confirmed 5-minute swing low.
- **Target:** the older (outer) pattern swing high, the "prior swing". Skip the trade if reward < 1.0 × risk.
- **Exit:** stop, target or day flat. Shorts mirror.

## J-T03: VWAP + Heikin-Ashi pullback scalp (`jt03`, 1-minute)
Transcript ≈04:39–08:43 (scalping chapter).
- **VWAP:** typical price, reset daily at 09:30 ET for the S&P (cash open) and at 17:00 ET for EURUSD (FX trading-day roll). On tick volume it is only a proxy.
- **Bias:** long only when the last 15 closes are all above VWAP, and the last two confirmed swing highs and swing lows are both rising (k = 3). The 15-bar wait is our reading of "avoid the first pullback after crossing VWAP".
- **Pullback:** at least 2 clean bearish HA bars (no upper wick) among the previous 10 bars. They need not be consecutive.
- **Entry:** the first bullish HA bar after a bearish one, and it must be clean (no lower wick).
- **Stop:** the lowest real low of the last 11 bars.
- **Exit:** a clean bearish HA bar, a close back below VWAP, 30 bars, the stop, or day flat. No target, because the transcript gives none. Shorts mirror.

## What the transcript leaves open (all chosen by us)
Zone width, swing k, arm window, the 1R minimum, trendline fitting, pattern age, VWAP anchor, HA wick tolerance (exact zero), the reversal-candle rule, the J-T03 stop and exits. J-T03 is a 1-minute scalp. That conflicts with the owner's low-frequency preference, so it is tested here and **not** prioritised.

## Pre-registered variants (6)
One per mode per instrument, at the source timeframes: `jt01_spx`, `jt01_fx`, `jt02_spx`, `jt02_fx`, `jt03_spx`, `jt03_fx` (`configs/competition.toml`, mirrored in `sensitivity_300k.toml`). No parameter search. Holdout stays sealed.

**Disclosure:** before this commit, a timing smoke test ran each mode on 2019 dev data and printed trade counts and risk-layer rejections only (no P&L). No rule or parameter was changed after it.
