# Handoff: S03b · 2026-11-25 · Phase 2 build, step 2 (news, Fed RSS, scheduler, launchd)

**Phase:** 2 (Collectors), step (b) build
**Session goal:** Commit step 1, then build the news loop, Fed RSS, the scheduler and the launchd service so news history starts accumulating (D-018).
**Outcome:** 🟡 partial (build step 2 of 5 done; the service is running unattended)

## What was done
- Step 1 committed as `34fbe52`.
- **News** (`collect/news/{feeds,extract,dedup,tag,collector}.py`): 5 feeds every 15 min; new URLs only (via `article_seen`); page fetch → raw HTML archived to `data/news_archive/` (gzip jsonl per source per day) → trafilatura extract (`ok`/`short`/`failed`, all stored) → dedup (canonical URL sha256, title hash, 64-bit SimHash, Hamming ≤ 3 within 30 days → `near_dup_of`) → alias tagging → `articles`. `released_at` = first-seen. Per-feed failure counting in the cursor.
- **Fed RSS** (`collect/fed.py`): press_monetary, speeches, press_all → `raw_events` (source `fed`), kind classified by config rules, speaker + role from config.
- **Scheduler** (`orchestrator/{jobs,scheduler}.py`): APScheduler 3.x, ET, coalesce, max_instances 1, misfire grace 900 s. Jobs: news_loop, heartbeat, freshness, crypto_daily, congress_daily, earnings_daily, weekly, backup, daily_summary, stock_daily_replan, stock_daily (re-planned daily; 13:30 ET early closes, none on holidays; catches up if started late). Job error/missed/overlap → ERROR → critical alert. Heartbeat to `data/run/`.
- Daily summary gained rows-collected and heartbeat-gap sections; backup is callable as a job.
- **launchd** (`scripts/install_scheduler.py`): installed and running as `com.qt.scheduler`, profile `paper`.
- Live: first dev run collected 156 articles (0 failed, 6 short) and 44 Fed events. The paper-profile service run at 05:36 UTC added 5 articles.

## Files touched
| File | Change |
|---|---|
| `configs/collect.yaml` | `news`, `fed`, `schedule` sections |
| `configs/aliases.yaml` | New (finance cues, suffixes, manual aliases) |
| `src/qt/common/config.py` | News/Fed/Schedule/Aliases config models |
| `src/qt/collect/news/*.py`, `src/qt/collect/fed.py` | New |
| `src/qt/orchestrator/{jobs,scheduler}.py`, `scripts/install_scheduler.py` | New |
| `src/qt/common/{reports,logging}.py`, `src/qt/db/backup.py` | Summary sections, quiet noisy libs, `backup_job` |
| `pyproject.toml`, `uv.lock` | apscheduler 3.x, feedparser, trafilatura |
| `tests/conftest.py`, `tests/fixtures/news/*`, `tests/test_{news_dedup,news_collector,scheduler}.py` | New tests, synthetic fixtures |
| `DECISIONS.md` | D-028 |

## Tests
- Command: `uv run pytest -q` and `uv run ruff check .`
- Result: 124 passed; ruff clean.

## Decisions made (also in DECISIONS.md)
- D-028: launchd log paths off the drive, TCC requirement, Fed first-run backfill timing, stale alert fires once, feed alert after 4 consecutive failures.

## Experiments
None.

## State of the system
- DB: migrated to 005; articles and Fed events accumulating.
- Scheduler running? **Yes**, launchd `com.qt.scheduler` (paper). Check with `uv run python scripts/install_scheduler.py --status` and `data/run/heartbeat.json`.
- Git: this step committed (see START_HERE). Nothing pushed.

## Blockers / surprises
- launchd EX_CONFIG (78) when its log files were on exFAT → moved to `~/Library/Logs/quant_trader/`.
- TCC blocked the service from reading the removable volume until Dom granted access to the uv python binary. If the python is upgraded (new uv-managed path), access must be granted again.
- Mac sleep stops the service: news gaps and missed-job alerts. Matters for the 5-day exit gate.
- Ticker tagging finds nothing until `aliases` is populated (step 4). It emits current tickers; step 4 must handle historical tickers (e.g. FB → META).

## Next exact task
1. Build step 3: FRED/ALFRED all vintages (D-023) + the `BAMLH0A0HYM2` archive.
2. Then step 4 (Alpaca bars + assets + membership + aliases + backfill), step 5 (everything else).

## Questions to ask Dom next session
- Confirm the Fed Board roles in `configs/collect.yaml` (Warsh Chair, Jefferson VC, Bowman VCS, Barr/Cook/Waller/Miran Governors).
- Wrap the service in `caffeinate` to keep the Mac awake on power during the 5-day gate?
