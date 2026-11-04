# Draft quantitative risk policy (simulation only; owner approval pending)

**Status: PROPOSED, NOT APPROVED.** These are conservative *engineering test parameters*, not empirically optimized strategy limits or a suggestion for live sizing. The assumed paper-equity base is a hypothetical $25,000; adjust after the owner selects paper-account equity, broker min size, margin and costs. Initial simulator must reject sizes below a venue's minimum, rather than rounding them upward.

| Control | Proposed initial paper limit | Enforcement |
|---|---|---|
| Sizing | Target at most **0.25% paper equity** loss at stop per new position, *including a modeled slippage/fees buffer* | Risk engine before order |
| Portfolio open-stop exposure | At most **0.75% paper equity** total worst-case modeled stop loss across positions | Cross-market engine |
| Daily realized + unrealized loss breaker | **1%** from day-start equity; block new entries and apply approved flatten policy | Independent supervisor |
| High-water equity drawdown breaker | **5%** from latest approved paper high-water mark; disable entries until explicit reset | Persistent supervisor |
| Max new entries | **3** per rolling 24 hours across pilot symbols (adjust only via experimental approval) | Order gateway |
| Concurrent positions | **2**, no more than one per instrument | Position manager |
| Overnight exposure | Allowed only when exchange/FX session permits, carrying cost and stop protection are accounted for; pre-close policy tested | Session-aware risk engine |
| Stops | Server/broker acknowledged protective stop required for each filled position; if attachment fails, invoke pre-approved emergency flatten, freeze further entries and alert | Execution state machine |
| Gaps, stale feeds, disconnects | Fail closed on new entries, audit existing orders, attempt reconciliation, escalate if exits unverified | Watchdog and reconciliation |

**Important:** Stops do not guarantee fill at the stop price. Leverage, gap risk, margin changes, trading halts and FX counterparty behavior can exceed modeled losses. Paper trading does not establish live execution quality. Broker-native stop/OCO functionality must be verified and tested, including partial fills. Agent may not switch stops off or increase limits without specific approval.

Sizing sketch: `risk_budget = paper_equity * 0.0025`; `raw_units = floor(risk_budget / (stop_distance_in_price * dollars_per_price_unit + estimated_round_trip_cost_and_slippage_per_unit))`; `units = min(raw_units, broker/portfolio/margin constrained units)`. If `units < minimum_legal_order_size`, **NO TRADE**. For spot FX, contract unit and quote-currency conversions must be correct. The target stop distance can be ATR-based but must also be checked against tick size, venue stop constraints and gaps. Backtests should include stricter and looser risk sensitivity scenarios.

### Before any demo orders
Owner explicitly approves final numeric limits and paper equity; provider eligibility and stop semantics tested; separate operator kill switch functional; exchange calendars and trading hours validated; simulated failure-injection passes; account isolation from live endpoints proven. No paper orders before Stage 06 authorization. Live trading requires a separate extensive authorization.
