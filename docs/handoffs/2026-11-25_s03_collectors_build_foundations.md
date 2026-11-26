# Handoff: S03 · 2026-11-25 · Phase 2 build, step 1 (foundations)

**Phase:** 2 (Collectors), step (b) build
**Session goal:** Start the Phase 2 build: migration 005, shared collector plumbing, point-in-time rules and readers, with tests.
**Outcome:** 🟡 partial (build step 1 of 5 done)

## What was done
- Confirmed Dom's keys load: Alpaca paper (passes the `PK` guard), FRED, Congress, CoinGecko Demo (`CG-`), SEC UA. SoSoValue and Finnhub are still empty (optional).
- **Migration `005_collectors.sql`** (applied to the real DB): `index_membership`, `aliases`, `raw_events`, `raw_metrics`, `raw_crypto_derivs`, `collector_state`; the new `articles` columns and the `released_at` comment; append-only triggers on the 4 new raw tables.
- **`configs/collect.yaml`** + a strict `CollectCfg` (sections `pit`, `http`, `freshness`). The remaining sections from component doc §9 are added as each collector is built.
- **Secrets:** `CONGRESS_API_KEY`, `COINGECKO_DEMO_API_KEY`, `SOSOVALUE_API_KEY`, `FINNHUB_API_KEY`, `SEC_EDGAR_USER_AGENT` added to `Secrets`, with `Secrets.has()`.
- **`qt.collect.pit`:** one pure function per §3 rule. Rejects naive datetimes; `UntimeableRow` when a row can't be timed.
- **`qt.collect.http`:** shared httpx client: per-host rate limits, retries on transport/429/5xx with backoff + jitter, `Retry-After`, 4xx fail fast, no query strings in logs.
- **`qt.collect.base`:** `Collector` protocol, `RunCtx`, `CollectResult`, `insert_rows` (chunked `ON CONFLICT DO NOTHING`, rejects untimed rows), `run_collector`/`run_all` (missing-key skip, `run_context`, `collector_state`, failure isolation), `freshness` + `alert_stale`.
- **`qt.db.pit`:** `macro_asof`, `latest_rows` (table whitelist).

## Files touched
| File | Change |
|---|---|
| `src/qt/db/migrations/005_collectors.sql` | New |
| `configs/collect.yaml` | New |
| `src/qt/common/config.py` | `CollectCfg` and the new secrets |
| `src/qt/collect/{pit,http,base}.py`, `src/qt/db/pit.py` | New |
| `tests/test_pit_rules.py`, `test_no_lookahead.py`, `test_append_only_collectors.py`, `test_http.py`, `test_collect_base.py` | New |
| `tests/test_migrate.py` | Expects the 005 tables |
| `DECISIONS.md` | D-027 |

## Tests
- Command: `uv run pytest -q` and `uv run ruff check src tests`
- Result: 102 passed (55 before + 47 new); ruff clean.

## Decisions made (also in DECISIONS.md)
- D-027: collector names ≤ 13 chars, `collector_state.last_status`, `collect.pit` config, URL redaction in logs.

## Experiments
None.

## State of the system
- DB: migrated to 005; no market data yet.
- Scheduler running? No.
- Git: branch `vibe/quant_trader`, last commit `fc919dc`; this session's work is uncommitted. Nothing pushed.

## Blockers / surprises
- None. Note: `ruff format --check` reports 4 files that would be reformatted. Phase 1 only enforced `ruff check`, so nothing was reformatted.

## Next exact task
1. Build step 2: news loop (`collect/news/{feeds,extract,dedup,tag}.py`) + Fed RSS + scheduler (`orchestrator/{scheduler,jobs}.py`) + `scripts/install_scheduler.py`. Add deps `feedparser`, `trafilatura`, `apscheduler>=3.10,<4`. Ticker tagging needs `aliases`, which is built in step 4, so news tagging stores `tickers='{}'` until then, or builds a minimal alias set first.
2. Then step 3 (FRED/ALFRED + HY archive), step 4 (Alpaca + membership + backfill), step 5 (everything else).

## Questions to ask Dom next session
- OK to install the launchd agent (`~/Library/LaunchAgents/com.qt.scheduler.plist`) when the scheduler is ready?
