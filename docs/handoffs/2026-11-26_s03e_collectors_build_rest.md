# Handoff: S03e · 2026-11-26 · Phase 2 build, step 5 (all remaining collectors)

**Phase:** 2 (Collectors), step (b) build
**Session goal:** Push steps 1–4, then build step 5 (yfinance, corporate actions, EDGAR facts, Congress, earnings, crypto structure), backfill, report.
**Outcome:** 🟢 build complete (5 of 5); exit gate pending (Congress backfill finishing, 5-day unattended run)

## What was done
- Dom confirmed the Fed Board roles (Warsh Chair) and D-029; steps 1–4 pushed to `vibe/quant_trader`.
- **New collectors** (all registered in the scheduler, freshness rule each):
  | Collector | Job | Table / source | Live backfill |
  |---|---|---|---|
  | `yfinance` | stock_daily | raw_prices `yfinance` | 106k bars, 16 tickers from 2000 |
  | `corp_actions` | stock_daily | raw_events `alpaca_ca` | 21.8k (dividends, splits, mergers, renames…) since 2016 |
  | `funding` | crypto_daily | raw_crypto_derivs | OKX 281/coin (~3 mo), Hyperliquid 30k/coin from 2023-06 |
  | `crypto_oi` | crypto_daily | raw_crypto_derivs | 12 per run (4 venues × BTC/ETH/SOL) |
  | `coingecko` | crypto_daily | raw_metrics | 365 d × 33 coins (volume, cap, price) + global dominance |
  | `defillama` | crypto_daily | raw_metrics | stablecoin supply from 2017-11 |
  | `earnings` | earnings_daily | raw_events `nasdaq` | ~25k reports 2016→, + next 4 weeks |
  | `edgar_facts` | weekly | raw_fundamentals | 1.52M facts, 596 companies (600 assets) |
  | `congress` | congress_daily | raw_events `congress` | **in progress**: 114th congress offset 6,750 after 1 slice |
- **PIT rule added:** `pit.corp_action` = min(fetch, ex date 00:00 ET). EDGAR `acceptanceDateTime` verified true UTC against filing index pages.
- **Membership:** unchanged weekly snapshot no longer re-stored (S03d issue fixed).
- **Bugs caught live and fixed:** share classes sharing a CIK (GOOGL/GOOG, FOX/FOXA, NWS/NWSA, UA/UAA) were skipped on the second class; now fetched once and stored per asset (re-backfilled). Hyperliquid 429s at 2 req/s → 0.5. Congress backfill could wedge on a page larger than its budget or a 404 bill → fixed + tested.
- `fed_rss` had no freshness rule (fell back to 48 h default): now 2 h; a test asserts every collector has an explicit rule.
- Service restarted on step 5 code (paper, caffeinate). `stock_daily_no_session` on Thanksgiving, as expected.

## Files touched
| File | Change |
|---|---|
| `src/qt/collect/{yfin,corp_actions,edgar,congress,earnings,crypto_structure,crypto_derivs}.py` | New |
| `tests/test_crypto_collectors.py`, `tests/test_event_collectors.py` | New (19 tests) |
| `src/qt/collect/alpaca.py` | `corporate_actions()` |
| `src/qt/collect/pit.py` | `corp_action` rule |
| `src/qt/collect/membership.py`, `tests/test_membership.py` | Skip unchanged snapshot |
| `src/qt/collect/backfill.py` | New collectors in `--all` order; report covers metrics/derivs/fundamentals |
| `src/qt/common/config.py`, `configs/collect.yaml` | 7 new config sections, freshness rules, host rate limits |
| `src/qt/orchestrator/jobs.py`, `tests/test_scheduler.py` | Register collectors |
| `pyproject.toml`, `uv.lock` | `yfinance` |
| `.env.example`, `DECISIONS.md` | Key notes; D-031 |

## Tests
- Command: `uv run pytest -q` and `uv run ruff check .`
- Result: 168 passed; ruff clean.

## Decisions made (also in DECISIONS.md)
- D-031: EDGAR per-company API (not the 1.4 GB zip); corp-action release rule; Congress candidate filter + budgeted backfill; Nasdaq earnings backfill; **SoSoValue dropped** (no key, 403); Finnhub fallback not built; CoinGecko 365-day limit and dominance only from 2026-11-26; funding interval from settlement spacing; yfinance close semantics; membership dedupe; Fed roles confirmed.

## Experiments
None.

## State of the system
- DB: all Phase 2 raw tables populated; 0 null `released_at`; nothing stamped in the future except announced corporate actions/earnings (released at fetch). Report: `reports/backfill_20261126.md` (gitignored).
- **Congress backfill still running** as a background driver from the session scratchpad (up to 25 slices, ~70 min each; ~14 more needed). If it dies (reboot), the 06:00 ET `congress_daily` job continues it at 4,000 requests/day. Both can run at once safely (ON CONFLICT; worst case a slice is redone).
- Scheduler running? Yes, launchd `com.qt.scheduler` (paper) under caffeinate, on `32c5117`.
- Git: `32c5117` (+ this handoff). Steps 1–4 pushed; step 5 **not pushed**.

## Blockers / surprises
- Nasdaq times out about once per 600-day slice; the slice keeps its progress and the next run resumes (8 runs to finish).
- No fundamentals for delisted members (SEC's ticker map lists only current companies). Survivorship gap for Phase 3's factor history, alongside the 3 bar-less delisted names (S03d).
- CoinGecko Demo allows only 365 days, so the D-021 ADV filter has one year of history; BTC dominance history starts today.
- Earnings backfill release times are approximate (report − 7 days) for everything before today.

## Next exact task
1. Let the Congress backfill finish; check `collector_state.cursor->backfill->done`.
2. Phase 2 exit gate (MASTER_PLAN / component doc §13 item 5): 5 consecutive days unattended. Check the daily summary each day for freshness of every collector, and news gaps < 1 h. The first full weekday cycle is Fri 2026-11-27 (early close 13:00, stock_daily 13:30 ET).
3. While the gate runs: Phase 3 questioning (`docs/interrogation/03_*.md`).

## Questions to ask Dom next session
- OK to push step 5 (`32c5117` + handoff)?
- SoSoValue: want to sign up for a free key so ETF flows come back? Otherwise they stay dropped.
- Accept the survivorship gaps (no EDGAR facts for delisted members; 3 delisted names without bars), or look for a source in Phase 3?

## Addendum (same session, after Dom's answers)
- Dom: push step 5, get a SoSoValue key, close the survivorship gaps in Phase 3. Step 5 pushed (`e147c04`).
- **`sosovalue` collector built** (D-032): skipped until `SOSOVALUE_API_KEY` is set; 2 tests; 170 passed.
- Survivorship gaps added as Phase 3 question 9, with unverified leads (SEC `cik-lookup-data.txt` for delisted CIKs; Stooq or successor tickers for the 3 bar-less names).
- Congress backfill: slice 2 done (115th congress, offset 2,250); the driver continues.
