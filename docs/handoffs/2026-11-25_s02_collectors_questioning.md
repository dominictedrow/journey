# Handoff: S02 · 2026-11-25 · Phase 1 close-out + Phase 2 questioning

**Phase:** 1 closed; 2 (Collectors), step (a) questioning
**Session goal:** Close the Phase 1 exit gate, commit the build, run the Phase 2 questioning and write the build spec.
**Outcome:** ✅ done. Phase 1 is ✅. All 12 Phase 2 questions are answered and `docs/components/02_collectors.md` is written.

## What was done
- **Phase 1 close-out:**
  - Re-sent test pushes. Dom's subscription first failed because the `qt-` prefix was left off the topic; after he fixed it, all pushes arrived and a fresh priority-5 push was sent to test banners.
  - D-020 accepted. Phase 1 marked ✅ in MASTER_PLAN.
  - Phase 1 build committed as `9788e9c` (44 files staged by name, secret scan clean, 55 tests green, ruff clean).
- **Phase 2 questioning:** Q1–Q12 answered inline in `docs/interrogation/02_collectors.md`.
- **Probes run during questioning:**
  - Alpaca 30-day crypto ADV on its own venue: BTC ~$203k, ETH ~$57k, SOL ~$35k, XRP ~$24k, the rest < $7k. So the spec's $50M filter must use CoinGecko global volume instead (D-021).
  - Alpaca crypto 1Day bars start at `T00:00:00Z`, which supports the 00:00 UTC daily close (D-024).
  - Alpaca crypto bars answered without API keys.
- **Build spec** `docs/components/02_collectors.md`:
  - §3 point-in-time `released_at` rule per source
  - migration 005 tables
  - collectors and the scheduler
  - config
  - tests
  - exit gate
- **`.env`:** `SEC_EDGAR_USER_AGENT` set (Dom's chosen contact email; not committed anywhere).

## Files touched
| File | Change |
|---|---|
| `docs/interrogation/02_collectors.md` | Answers to Q1–Q12 |
| `docs/components/02_collectors.md` | New: Phase 2 build spec |
| `DECISIONS.md` | D-020 accepted; D-021 to D-026 added |
| `docs/plan/MASTER_PLAN.md` | Phase 1 ✅; Phase 2 status |
| `START_HERE.md`, this handoff | Status |
| `.env` (gitignored) | `SEC_EDGAR_USER_AGENT` |

## Tests
- No code changed after the Phase 1 commit. `uv run pytest -q`: 55 passed; ruff clean (run before commit `9788e9c`).

## Decisions made (also in DECISIONS.md)
- D-020: accepted by Dom
- D-021: crypto pool from Alpaca; ADV ≥ $50M measured on CoinGecko global volume
- D-022: news every 15 min, store everything, `released_at` = first-seen
- D-023: all ALFRED vintages, as-of reads
- D-024: stock collect = close + 30; crypto close = 00:00 UTC
- D-025: S&P 500, ~80 candidates, Alpaca 2016+, yfinance pre-2016 index-level only
- D-026: migration 005 table layout; derivs for BTC/ETH/SOL

## Experiments
None.

## State of the system
- DB: migrated to 004; no market data.
- Scheduler running? No (built in Phase 2).
- Git: branch `vibe/quant_trader`, last commit `9788e9c`; this session's doc changes are uncommitted. Nothing is pushed.

## Blockers / surprises
- **API keys still empty in `.env`:** Alpaca paper, FRED, Congress (api.data.gov), CoinGecko Demo, optionally SoSoValue and Finnhub. The build can start with mocked tests, but backfills need the keys.
- Several **(default)** choices in the spec were not asked:
  - Wikipedia as the weekly membership source
  - futures `released_at` at 17:00 ET
  - a FRED default release time of 17:00 ET
  - the FOMC schedule known from Dec 1 of the prior year
  - earnings backfill at report date − 7 days
  - storing Alpaca `adjustment=all` alongside raw
  - Congress at 06:00 ET

  Dom accepted all of them at the end of S02.

## Next exact task
1. Dom adds the API keys to `.env` (the defaults are accepted; the S02 docs are committed).
2. Phase 2 build, in this order:
   1. migration 005 + `collect/http.py`/`base.py`/`pit.py` + PIT tests
   2. news loop + Fed RSS + scheduler + launchd, started ASAP so news history begins accumulating (D-018)
   3. FRED/ALFRED + HY archive
   4. Alpaca + membership + backfill
   5. everything else

## Questions to ask Dom next session
- Are the keys added?
