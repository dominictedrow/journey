# Handoff: S01 · 2026-11-25 · infra questioning

**Phase:** 1 (Infrastructure), step (a) questioning
**Session goal:** Run the Phase 1 questioning with Dom and write the build spec.
**Outcome:** ✅ done

## What was done
- Removed all 239 macOS `._*` AppleDouble files: 102 in the tree and 137 inside `.git`. None were tracked, and `git fsck` was clean afterwards. exFAT keeps creating them, so the cleanup needs repeating now and then (they're gitignored).
- Asked all 10 questions in `docs/interrogation/01_infrastructure.md` and recorded the answers inline.
- Wrote `docs/components/01_infrastructure.md`, the full build spec: Docker, schema, migrations, config, `run_id`, logging, alerts, backups, calendar, tests and the exit gate.
- Logged D-012 to D-018. Added ntfy to API_REGISTRY. Updated MASTER_PLAN (Phase 1 status; `db_dump.py` renamed to `db_backup.py`) and `.env.example` (new DB, backup and ntfy keys).

## Files touched
| File | Change |
|---|---|
| `docs/interrogation/01_infrastructure.md` | Answers recorded |
| `docs/components/01_infrastructure.md` | New: build spec |
| `DECISIONS.md` | D-012 to D-018 appended |
| `docs/plan/API_REGISTRY.md` | ntfy row added (limit unverified) |
| `docs/plan/MASTER_PLAN.md` | Phase 1 status; backup script name |
| `.env.example` | POSTGRES_*, ports 54320/54321, BACKUP_DATABASE_URL, NTFY_TOPIC, QT_PROFILE |
| `START_HERE.md`, this handoff | Session end |

## Tests
None. There is still no code.

## Decisions made (also in DECISIONS.md)
- D-012: hypertables (`raw_prices`, `features`, `predictions`) from the start; long-format features in the DB plus a Parquet cache
- D-013: plain SQL migrations plus a runner, no Alembic
- D-014: per-component YAML plus profile overlays (`dev`/`paper`/`backtest`), validated with pydantic
- D-015: text console plus JSONL logs, a Markdown daily summary, and the readable UTC `run_id`
- D-016: ntfy push alerts; every failure and every trade alerts; no email
- D-017: a separate backup DB container refreshed each evening, plus 7 dump files
- D-018: only collectors run on the Mac; models and trading run only as smoke tests until Windows

## Experiments
None.

## State of the system
- DB: none yet (no compose file)
- Scheduler running? no
- Git: branch `vibe/quant_trader`, last commit `30e794a`. This session's doc changes are **uncommitted**.

## Blockers / surprises
- Dom first asked for a backup "page", then clarified that he meant a separate backup **database**. The spec reflects the clarified version.
- The free ntfy.sh message cap is unverified. Check it during the build (alert coalescing keeps volume low anyway).

## Next exact task
1. Phase 1, step (b) build, from `docs/components/01_infrastructure.md`:
   1. `pyproject.toml` and `uv sync`
   2. `docker-compose.yml` (pin the TimescaleDB tag and record it in DECISIONS)
   3. migrations 001 to 004 and the runner
   4. config loader and the `configs/` files
   5. `run_id` and `run_context`
   6. logging and alerts
   7. `scripts/db_backup.py`
   8. calendar
   9. the tests in §13
2. Meet the §14 exit gate. Dom must receive one test push on his phone.

## Questions to ask Dom next session
- Has he installed the ntfy app? Generate a random `NTFY_TOPIC` for him to subscribe to.
- Is Docker Desktop installed and running on the Mac?
