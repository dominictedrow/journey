# Joovier transcript audit — partial, source-linked (2026-10-18 PDT)

## Honest coverage statement
This is **not a completed video watch**. YouTube playback and complete frame-by-frame chart inspection were not available. A public third-party, timestamped transcript of the Jooviers Gems six-hour course (`gxnfl2AoVJA`) was accessible and **selected strategy sections** were inspected. Timestamp accuracy, automated transcription, visual examples, implementation conditions, and other course versions require direct verification. Do not label this entire course watched or audited. No performance outcomes are established.

Source: https://www.youtube.com/watch?v=gxnfl2AoVJA
Transcript: https://my.infocaptor.com/hub/summaries/jooviers-gems/how-to-start-day-trading-for-beginners-in-2025-6-hours-gxnfl2AoVJA

## Extracted hypotheses — NOT approved to implement as creator-faithful indicators

### J-T01: Higher-timeframe support/resistance retest with swing confirmation
- Transcript approximately **33:35–38:33** (timestamps shown within a strategy chapter; validate chapter offsets before deep links).
- Mark support/resistance zones on **1-hour or 4-hour** chart; wait for price to revisit a zone, not an immediate entry on first touch.
- On **1-minute** chart for buys, wait for a break above recent swing high after support retest; for shorts at resistance, break below recent swing low.
- Long stop below recent swing low; short stop above recent swing high. Targets at recent swing highs/lows, with target choice discussed subjectively in relation to volume.
- **Missing formalization:** zone boundaries; swing lookback/confirmation; break defined wick vs close; price source/bid-ask; volume cutoff; invalidations; acceptable spreads.
- Transcript evidence: section around page lines 3708–3829 of the third-party transcript.

### J-T02: Consolidation squeeze breakout
- Transcript approximately **49:44–53:21** in strategy chapter; timestamps reset across concatenated sections.
- Look for converging trendlines, **higher lows and lower highs**; at least **two rejections at each** of upper and lower lines on **1-hour** chart.
- Do not buy merely on trendline break; require confirmation by break of recent swing high (or swing low for shorts), observing **1-minute or 5-minute** execution timeframe.
- Stops beyond recent opposite swing; targets at prior swings. Commentary suggests strong volume, but no numerical threshold defined.
- **Missing formalization:** touch tolerance, how to fit lines without future information, breakout timing and volume rules, false-break handling.
- Transcript evidence: lines 4108–4183.

### J-T03: VWAP + Heikin-Ashi continuation scalping
- Transcript approximately **04:39–08:43** in scalping chapter; chapter offset not known from transcript alone.
- **1-minute** timeframe. Direction filter: long above VWAP, short below VWAP. Demand market structure on that side before taking a pullback (first crossing pullback often avoided).
- Look for **at least two clean Heikin-Ashi pullback bars** with specified absent wick on one side; speaker says these need not be consecutive. Wait for an entry/reversal candle (exact entry condition not fully extracted).
- **Missing formalization:** VWAP anchor/session, precise HA wick tolerance, structure definition, reversal-candle rule, hard stop, execution on real quotes versus synthetic HA values, impact of spreads at 1-minute scale.
- Transcript evidence: lines 4446–4557.

## Engineering implications
- These three approaches should be separate baseline hypotheses with independent after-cost backtests before feeding labels/features to SVM. Original creator rules remain unverified without chart playback.
- A 1-minute scalp may conflict with the owner's minute-to-hour low-frequency selective preference and must not be prioritized without approval.
- Avoid forward-looking swing detection, in-sample tuning, synthetic-candle fills, and creator-reported profit claims. No assertion of profitability.

## Next steps
1. Obtain primary video playback and manually annotate representative chart examples and failures, with full timestamp offset and visual proof.
2. Complete transcript segment review including exact entry triggers and exits; document contradictions.
3. Find legitimate complete public Swaggy C instruction and audit separately; do not treat sales pages as strategy lessons.
4. Present formal definitions and alternatives for owner approval before Stage 04 code.
