# Handoff: S01 · 2026-11-25 · infra questioning + build

**Phase:** 1 (Infrastructure), step (a) questioning **and** step (b) build
**Session goal:** Run the Phase 1 questioning with Dom, write the build spec, then build Phase 1.
**Outcome:** 🟡 almost done. All of the build is finished and tested. The exit gate is met except for one item: Dom confirming the test push on his phone.

## What was done
### Part 1: questioning (commit `5e6985a`)
- Removed 239 macOS `._*` AppleDouble files (102 in the tree, 137 in `.git`; none tracked).
- Asked all 10 questions in `docs/interrogation/01_infrastructure.md`, recorded the answers, and wrote `docs/components/01_infrastructure.md`.
- Logged D-012 to D-018. Updated API_REGISTRY (ntfy), MASTER_PLAN and `.env.example`.

### Part 2: build (not yet committed)
- **uv project:** `pyproject.toml` and `uv.lock`, Python 3.13. The venv is at `~/.venvs/quant_trader`. Runtime deps: psycopg[binary,pool], pydantic, pyyaml, python-dotenv, exchange_calendars, pandas, pyarrow, httpx. Dev deps: pytest, pytest-cov, ruff.
- **`docker-compose.yml`:** `qt_db` (port 54320) and `qt_db_backup` (port 54321), both `timescale/timescaledb:2.32.1-pg17`, bound to localhost, each with its own named volume, both with healthchecks. Both are running and healthy.
- **Migrations:** `src/qt/db/migrations/001` to `004`:
  - all 20 spec tables, plus `alerts_log`, `backup_log`, `article_scores` and the `model_input_latest` view
  - hypertables with compression policies
  - append-only triggers that block UPDATE, DELETE and TRUNCATE on the raw tables
- **Migration runner:** `qt.db.migrate`, with sha256 tamper detection, an advisory lock and one transaction per file. The real `quant_trader` DB is **migrated to 004**.
- **Config:** `qt.common.config`: base YAML plus the `dev`/`paper`/`backtest` overlays, pydantic validation with unknown keys rejected, `config_hash`, the paper-key guard, and the hard-coded paper URL.
- **Runs:** `qt.common.runs`: `new_run_id` and `run_context` (runs row, contextvars, critical alert on failure).
- **Logging:** `qt.common.logging`: text console, daily UTC JSONL files with 30-day pruning, and the root alert handler.
- **Alerts:** `qt.common.alerts`: ntfy JSON publish, coalescing with "xN" follow-ups, flush at exit, OS-notification fallback, the `alerts_log` sink, and a guarantee that alerting never raises.
- **Calendar:** `qt.common.calendar.TradingCalendar` (XNYS, early closes, DST, naive datetimes rejected).
- **Reports:** `qt.common.reports.write_daily_summary`, a scaffold whose Runs, Alerts and Backups sections are filled from the DB.
- **Backups:** `qt.db.backup` + `scripts/db_backup.py`: dump → guard against the backup pointing at the primary cluster → drop and recreate → pre_restore → pg_restore → post_restore → exact row-count compare → prune → `backup_log` → info push. It has run for real twice, and both runs were OK with 24 tables matched.
- **`scripts/alert_test.py`:** test pushes. Four pushes are confirmed on the ntfy server: two backup OKs, the deliberate critical error and one info test.

## Files touched
| File | Change |
|---|---|
| `pyproject.toml`, `uv.lock` | New |
| `docker-compose.yml` | New |
| `configs/{db,app,backup,calendar}.yaml`, `configs/profiles/*.yaml` | New |
| `src/qt/common/{paths,config,logging,alerts,runs,calendar,reports}.py` | New |
| `src/qt/db/{conn,migrate,backup}.py`, `src/qt/db/migrations/00{1..4}_*.sql` | New |
| `scripts/db_backup.py`, `scripts/alert_test.py` | New |
| `tests/*` (11 files) | New |
| `configs/.gitkeep`, `scripts/.gitkeep`, `tests/.gitkeep` | Deleted (folders now have content) |
| `docs/components/01_infrastructure.md` | Pinned the tag, ET dump naming, §10a build notes |
| `DECISIONS.md` | D-019, D-020 |
| `docs/plan/MASTER_PLAN.md`, `START_HERE.md`, this handoff | Status |
| `.env` (gitignored, **not committed**) | Created with a generated DB password and `NTFY_TOPIC` |

## Tests
- Commands: `uv run pytest -q` and `uv run ruff check src tests scripts`.
- Result: **55 passed, 0 skipped, 0 failed**, and ruff is clean. The only warnings are Pandas4 deprecation warnings raised inside exchange_calendars.

## Decisions made (also in DECISIONS.md)
- D-012 to D-018: see part 1
- D-019: image pinned to `2.32.1-pg17`, ports bound to localhost only
- D-020 (**proposed**, Dom to confirm): pushes only in the `paper` profile, and dumps named by ET date

## Experiments
None.

## State of the system
- DB: `qt_db` is migrated to 004 and has no market data yet. `qt_db_backup` holds a restored copy as of 2026-11-25 ~20:55 ET.
- Dumps: `data/backups/qt_20261125.dump`
- Scheduler running? No, that's Phase 2. The evening backup has to be run by hand until then.
- Git: branch `vibe/quant_trader`. Part 1 is committed as `5e6985a`; the build is **uncommitted**. Nothing is pushed.

## Blockers / surprises
- **AppleDouble files come back immediately.** macOS writes `._*` next to every file on exFAT, and they broke the YAML glob (UnicodeDecodeError). Every glob now skips `._*`. They are gitignored, so they never reach git.
- ntfy's free-tier message cap is still unverified, but volume is tiny.
- The exit-gate phone check is pending: Dom must subscribe to the topic in the ntfy app. Messages stay cached on ntfy for about 12 h, so the four test pushes should show up once he subscribes.

## Next exact task
1. Dom subscribes to `NTFY_TOPIC` (from `.env`) in the ntfy app and confirms the pushes arrived. Then mark Phase 1 ✅ in MASTER_PLAN.
2. Confirm or reject D-020.
3. Start the Phase 2 questioning (`docs/interrogation/02_*.md`), getting to the **news collector + HY OAS archive** first (urgent, D-001/D-018).

## Questions to ask Dom next session
- Did the pushes arrive? Is D-020 OK?
- Phase 2 questioning.
