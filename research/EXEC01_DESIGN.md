# EXEC01: passive limit-order execution for REV01 EUR/USD (design + pre-registration)

**Status:** DESIGN, pre-registered 2026-11-03 before any EXEC01 code or P&L run. Implementing and running it changes order types, so it needs owner approval (D-027).
**Source:** candidate EXEC01 (`INSTITUTIONAL_STRATEGY_CANDIDATES.md`, BIS #23). It's an execution layer, **not new alpha**: the REV01 signal is unchanged.
**Why now:** D-026 showed REV01 fx_5m has a real gross edge (+0.26 bp/trade in 2012–19, +0.37 in 2020+, the same on mid and bid bars) that costs about 0.6–1.4 pips per round trip destroy. Paying the spread on both legs is the problem, so the test is whether entering and exiting with limit orders keeps the edge.

## Constraint found while designing: commission accounts can't work
The gross edge is about **0.3–0.4 pip per trade**.
- **OANDA core + commission:** about 0.5 pip per side in commission is charged on limit fills too, so the round trip costs at least **1.0 pip before any spread**. That's 2–3× the edge however good the execution is. **Ruled out**, and not tested.
- **Spread-only retail (OANDA standard):** a limit buy at L fills when the broker's **ask** falls to L, at L. With no commission, passive execution can avoid paying the spread, at the cost of missed fills and adverse selection. **This is the scenario to test.**
- **ECN floor:** Dukascopy's own spread (≈0.5 pip). Shown as an upper bound on what's achievable.

## Rules (fixed now)
Signals are exactly REV01 `rev01_fx_5m` as committed in 07db1ec: 5-minute bars, z-score of 20-bar SMA, entry at |z| ≥ 2, ER ≤ 0.3, session 03:00–12:00 ET, flat at 16:00 ET, stop 1.5σ, max 12 bars. Prices are mid bars; the broker quote is `mid ± h`, where `h` is the scenario's half-spread on each 1-minute bar.

**Passive entry.** At the signal bar close, place a buy limit at the broker **bid** (short: a sell limit at the **ask**). It's good for **5 minutes** (one mode bar), then cancelled with no trade. It can fill only from the next 1-minute bar onward (no same-bar fill).

**Fill rule (primary, conservative):** a buy limit at L fills in minute j if the broker **ask low** ≤ L, i.e. the offer crosses down to our bid. Fill price L, no slippage. A sell limit fills if the broker **bid high** ≥ L.
**Fill rule (sensitivity only, optimistic):** a buy fills if the broker **bid low** < L by at least 0.1 pip (trade-through). Reported, never used for the verdict.

**Stop** is set from the actual fill price, 1.5σ away. It's a stop-market order triggered on the correct side (long: bid low ≤ stop; short: ask high ≥ stop), fills at the bid (ask) with the scenario's slippage, and a gap fills at the open.

**Exits, two pre-registered variants:**
- **E1 (entry only):** the REV01 exits (reverted, time, flat) stay market orders at the broker quote. This isolates what passive entry is worth.
- **E2 (both legs):** at entry, also rest a take-profit limit at the 20-bar SMA from the signal bar, fixed, filled under the same cross rule. Time, flat and stop exits stay market. The "reverted" exit is replaced by the limit.

**Missed fills:** a signal whose limit expires unfilled is logged as `unfilled` and takes no trade. If the risk layer rejects at fill time, the order is never placed. Report the fill rate and the gross R of filled vs **unfilled** signals (measured as if entered at market). That's the adverse-selection check: passive orders tend to fill on the losers and miss the winners.

## Scenarios
| Name | Half-spread h per minute | Slippage on market/stop | Commission |
|---|---|---|---|
| retail | max(observed, 0.5 pip), i.e. ≥ 1.0 pip spread | 0.2 pip | 0 |
| ecn_floor | observed Dukascopy | 0 | 0 |
| retail_2x / 3x | retail h and slippage × 2 / × 3 | | 0 |

## Data and windows
`data/dukascopy/EURUSD_1m_bidask.parquet`, dev 2004-01-01 → 2025-10-01 (holdout sealed). Report the full period plus 2004–11 / 2012–19 / 2020+. $25k and $300k, PROPOSED risk config, unchanged.

## Pass criteria (all on the dev window, retail, primary fill rule)
1. Net expectancy > **+0.05R** per trade in **both** 2012–19 and 2020+.
2. Net P&L > 0 at **retail_2x**.
3. Fill rate ≥ 30% and at least 300 filled trades a year.
4. Max drawdown < 25% at $25k.

The 2004–11 period is reported but not gated: REV01's gross edge was already negative there, and D-026 records that as a regime fact.
Failing any criterion shelves EXEC01-on-REV01 for good. Passing would allow, with owner approval, a single holdout evaluation. It would not allow parameter tuning.

## Multiple testing
This is the 5th hypothesis family, and the variant comes from inspecting D-026's results. Count E1 and E2 as 2 new variants informed by prior results. The optimistic fill rule is sensitivity only.

## Engine work required (≈ 1 day, tests first)
1. Bars can carry the broker quote: `bid_*`/`ask_*` columns, or mid + per-minute `h`. The old mid-only path stays the default.
2. A new `Enter(order="limit", limit_price=..., ttl=...)` intent with a pending-order book. Fills only from the next minute, cancels at TTL, and logs unfilled signals.
3. Stops trigger on bid (longs) / ask (shorts). Take-profit limits use the cross rule.
4. Tests: no same-minute fill; TTL expiry; cross vs trade-through rules; stop side; a stop and a target in the same minute (stop wins); no-lookahead for the order book; the unfilled-signal ledger.

The live order path is untouched. This is backtest-only.
