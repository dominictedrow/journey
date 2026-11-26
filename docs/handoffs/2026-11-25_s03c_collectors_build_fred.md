# Handoff: S03c · 2026-11-25 · Phase 2 build, step 3 (FRED/ALFRED + HY archive)

**Phase:** 2 (Collectors), step (b) build
**Session goal:** Confirm the launchd service, commit step 2, then build FRED/ALFRED (all vintages) and the HY OAS archive.
**Outcome:** 🟡 partial (build step 3 of 5 done)

## What was done
- Dom granted the removable-volume permission; the launchd service runs under `paper`. Step 2 committed as `434e551`.
- **`collect/fred.py`:** `FredCollector` (`fred`) stores every ALFRED vintage of the 23 configured series (`source='alfred'`); `HyArchiveCollector` (`hy_archive`) snapshots BAMLH0A0HYM2 daily (`source='fred_archive'`, released_at = fetch time). Both run in `stock_daily`. Backfill and daily runs share one code path (resume from the newest stored vintage).
- **Live backfill done:** 186,193 ALFRED rows over 23 series and 785 archive rows; 0 null `released_at`. A second run inserts nothing new. Spot checks: PAYEMS 2020-04 first print 131,072 → now 130,426 over 10 vintages; CPI 2020-03 as of release day = the first print.
- **Migration 006:** `raw_macro` PK includes `source`.
- **PIT fixes from live probes (D-029):** default release 18:00 ET; HY OAS and VIXCLS released at the next session 12:00 ET (ALFRED backdates them).
- **Session-paced freshness:** session-only feeds no longer read as stale over weekends and holidays (would have alerted on Thanksgiving).
- `install_scheduler.py` waits for the old service to unload before bootstrapping (fixes "Bootstrap failed: 5").

## Files touched
| File | Change |
|---|---|
| `src/qt/collect/fred.py`, `tests/test_fred.py` | New |
| `src/qt/db/migrations/006_raw_macro_source_key.sql` | New |
| `src/qt/collect/pit.py`, `tests/test_pit_rules.py` | Release lag in sessions; 18:00 default |
| `src/qt/collect/base.py`, `tests/test_scheduler.py` | `stale_after`, `session_paced` rules |
| `src/qt/common/config.py`, `configs/collect.yaml` | `FredCfg`, `fred` section, lag config, freshness rules |
| `src/qt/orchestrator/jobs.py` | `stock_daily` runs `fred` + `hy_archive` |
| `scripts/install_scheduler.py` | Wait for unload |
| `DECISIONS.md` | D-029 |

## Tests
- Command: `uv run pytest -q` and `uv run ruff check .`
- Result: 134 passed; ruff clean.

## Decisions made (also in DECISIONS.md)
- D-029: 18:00 ET default, HY/VIX next-session release, clipping-aware vintage diff, ≤ 1900 vintages per range with a split on 400, migration 006, session-paced freshness.

## Experiments
None.

## State of the system
- DB: migrated to 006; ALFRED history loaded; news and Fed accumulating.
- Scheduler running? Yes, launchd `com.qt.scheduler` (paper), restarted on the step 3 code. The first scheduled `fred`/`hy_archive` run is Fri 2026-11-27 13:30 ET (early close).
- Git: step 3 committed (see START_HERE). Nothing pushed.

## Blockers / surprises
- ALFRED vintages for daily Treasury/FX series start in 2005–2016, so pre-vintage history is stamped with the first vintage date (conservative: it only becomes visible then). Matters for any pre-2005 HMM features in Phase 3.
- FRED's vintage counting is off by one vs `vintagedates`; handled.

## Next exact task
1. Build step 4: Alpaca daily bars (equities/ETFs from 2016, crypto) + Alpaca assets + fja05680 membership + EDGAR ticker map → aliases + backfill. Register in `stock_daily` / `crypto_daily` / `weekly`.
2. Then step 5 (everything else) and the backfill report.

## Questions to ask Dom next session
- Confirm the Fed Board roles in `configs/collect.yaml`.
- `caffeinate` for the 5-day gate?
- OK with the release-time changes in D-029 (18:00 ET default; HY/VIX next session)?
