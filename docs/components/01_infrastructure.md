# Component 01: Infrastructure

> Build spec for Phase 1. Written from Dom's answers in [`docs/interrogation/01_infrastructure.md`](../interrogation/01_infrastructure.md) (S01, 2026-11-25).
> Where this file and `docs/spec/system_spec_v2.md` disagree, this file wins. Decisions: D-012 to D-018.

## 1. Purpose
This component is the plumbing every later phase stands on:
- a reproducible database, plus a backup copy
- a migrated schema for every table in the spec
- validated config
- `run_id`s
- structured logs
- push alerts
- XNYS calendar helpers

It contains no data collection, no features and no models.

## 2. Scope
| In Phase 1 | Deferred |
|---|---|
| `pyproject.toml` + uv project, dev tooling (pytest, ruff) | APScheduler service and launchd/Task Scheduler registration → Phase 2 |
| `docker-compose.yml`: `db` + `db_backup` | Daily summary *content* (needs collectors) → Phase 2. Phase 1 ships only the writer skeleton |
| Numbered SQL migrations + runner, all spec tables | Parquet feature cache *builder* → Phase 3 (Phase 1 fixes only the layout contract) |
| Raw-table append-only enforcement (DB triggers) | Dashboard → Phase 10/11 |
| Config loader (base YAML + profile overlay + `.env`, pydantic) | |
| `run_id` + `runs` table helpers | |
| Logging (text console + JSONL file) + ntfy alert handler | |
| `scripts/db_backup.py` (dump → backup DB → verify) | Its evening scheduling → Phase 2 scheduler |
| XNYS calendar helpers | |

## 3. Docker
`docker-compose.yml` at the repo root defines two services. Both use the same pinned image, **`timescale/timescaledb:<exact-tag>-pg17`**. Pin the exact tag during the build and record it in DECISIONS. The two services must use the same version, because restoring a TimescaleDB dump requires matching extension versions.

| Service | Container | Named volume | Host port (env, default) | DB name |
|---|---|---|---|---|
| `db` | `qt_db` | `qt_pgdata` | `QT_DB_PORT` = 54320 | `quant_trader` |
| `db_backup` | `qt_db_backup` | `qt_pgdata_backup` | `QT_DB_BACKUP_PORT` = 54321 | `quant_trader` |

- Both volumes are Docker named volumes and never sit on exFAT (D-010).
- Ports 54320/54321 avoid clashing with any other local Postgres on 5432.
- Credentials come from `.env` (`POSTGRES_USER`, `POSTGRES_PASSWORD`). `DATABASE_URL` and `BACKUP_DATABASE_URL` are built from them.
- Both services have a healthcheck (`pg_isready`), and both have `restart: unless-stopped`.

## 4. Schema and migrations
### 4.1 Runner
- Migrations live in `src/qt/db/migrations/NNN_<slug>.sql` (3-digit, zero-padded).
- `uv run python -m qt.db.migrate [--target backup]` does the following:
  1. creates `schema_migrations(version int pk, filename text, sha256 text, applied_at timestamptz)` if it's missing
  2. applies each unapplied file in order, **each inside its own transaction**
  3. records each file's sha256
- If an applied file's sha256 has changed, the runner **fails loudly**. Applied migrations are immutable; fixes are new forward migrations. There are no down-migrations (D-013).
- The runner is pure Python (psycopg 3), with no shell.

### 4.2 Initial migrations
- **`001_extensions.sql`:** `CREATE EXTENSION IF NOT EXISTS timescaledb;`
- **`002_core_tables.sql`:** creates every table in the spec's schema section, with the changes below.
- **`003_hypertables.sql`:** sets up hypertables and compression.
- **`004_append_only.sql`:** adds the append-only triggers.

Changes from the spec's table list. Each one is needed for the hard rules or for the hypertables.

| Table | Change | Why |
|---|---|---|
| `raw_prices` | add `released_at timestamptz not null` (bar close time for bars) | Point-in-time rule: every raw row has `released_at` |
| `raw_macro` | already has `released_at` and `vintage` | — |
| `raw_fundamentals` | add `released_at` (= `filed_at` acceptance time) and `ingested_at` | Point-in-time |
| `articles` | add `ingested_at`; `released_at` = `published_at` (or first-seen if missing) | Point-in-time for news |
| `features` | add `feature_version int not null`, `computed_at`, `run_id` | Versioned features (adjusted data goes into its own version) |
| `predictions` | add `asof timestamptz not null` | A hypertable needs a time column |
| `runs` | add `stage`, `profile`, `config_hash`, `git_sha`, `error` | Reproducibility |
| new `alerts_log` | `ts, severity, event, title, body, delivered bool, run_id` | Audit trail of every alert sent or attempted |
| new `backup_log` | `ts, dump_file, dump_bytes, status, row_counts jsonb, mismatches jsonb` | Backup health record |

Other schema rules:
- **Timestamps:** every timestamp column is `timestamptz` (stored as UTC).
- **Enums:** enums are `text` with `CHECK` constraints, which are easier to migrate than PG enums.
- **Keys:** every table has a primary key or a unique key that makes collector inserts idempotent, e.g. `raw_prices (asset_id, ts, source)` and `raw_macro (series_id, period, vintage)`. Collectors use `INSERT ... ON CONFLICT DO NOTHING`.

### 4.3 Hypertables (D-012)
| Table | Time column | Chunk interval (config) | Compression after (config) | Segment by |
|---|---|---|---|---|
| `raw_prices` | `ts` | 30 days | 90 days | `asset_id` |
| `features` | `asof` | 30 days | 90 days | `feature_name, asset_id` |
| `predictions` | `asof` | 90 days | 180 days | `model_name` |

All other tables are plain tables. The chunk and compression numbers come from `configs/db.yaml`. The migration hard-codes the initial values, and later changes go through a new migration generated from config.

### 4.4 Append-only enforcement
The tables `raw_prices`, `raw_macro`, `raw_fundamentals`, `articles` and `article_seen` each get a `BEFORE UPDATE OR DELETE` trigger that raises an exception.

**Exception for `articles`:** the news pipeline must later write `svm_prob` and `kept`. These columns move to a separate `article_scores(article_id, run_id, svm_prob, kept, scored_at)` table, so `articles` itself stays immutable. Phase 8 will confirm this.

Compression and retention jobs run as the TimescaleDB background worker and are not blocked by row triggers.

### 4.5 Parquet feature cache: layout contract only (D-012)
- **Path:** `data/cache/features/v{feature_version}/{asof_start}_{asof_end}.parquet`. The format is wide: one row per (asof, asset_id) and one column per feature.
- **Metadata:** a sidecar `.json` records `feature_version`, the asof range, the source query, the row count and `built_at`.
- **Not the source of truth:** the cache can be deleted at any time and is rebuilt from the DB. The builder is written in Phase 3.

## 5. Config (D-014)
```
configs/
  db.yaml          app.yaml (logging, alerts, paths)   calendar.yaml
  backup.yaml      collect.yaml, features.yaml, ... (added by later phases)
  profiles/
    dev.yaml       paper.yaml       backtest.yaml
```
- **Loading:** `qt.common.config.load(profile: str | None = None) -> Config`:
  1. reads every `configs/*.yaml` into a dict keyed by file stem
  2. deep-merges `configs/profiles/{profile}.yaml` on top, where the profile comes from the `QT_PROFILE` env var or defaults to `dev`
  3. validates against pydantic models (`extra="forbid"`, so unknown keys fail)
  4. loads `.env` with python-dotenv
- **Secrets:** secrets are exposed only as `SecretStr`, and no secret ever appears in YAML.
- **`config_hash`:** the sha256 of the canonical JSON of the resolved config, with secrets excluded. It is stored on `runs`.
- **Paper guard:** this lives in config validation from day one. If an Alpaca key id is present and doesn't start with `PK`, startup fails. The base URL constant is the paper URL and can't be overridden from config.

Initial keys (defaults):
```yaml
# db.yaml
host: localhost
port: 54320
backup_port: 54321
name: quant_trader
pool: {min_size: 1, max_size: 5, max_idle_s: 300}
hypertables:
  raw_prices:  {chunk_days: 30, compress_after_days: 90}
  features:    {chunk_days: 30, compress_after_days: 90}
  predictions: {chunk_days: 90, compress_after_days: 180}

# app.yaml
logging:
  level: INFO
  dir: logs
  retention_days: 30
alerts:
  enabled: true
  provider: ntfy
  server: https://ntfy.sh
  coalesce_window_s: 300
  timeout_s: 5
  priority: {info: 3, warn: 4, critical: 5}
  os_fallback: true
reports:
  daily_dir: reports/daily

# backup.yaml
evening_time_et: "21:00"
dump_dir: data/backups
keep_dumps: 7

# calendar.yaml
exchange: XNYS
tz: America/New_York
```

## 6. `run_id` and the runs table
- **Format:** `qt.common.runs.new_run_id(stage)` → `YYYYMMDDTHHMMZ-<stage>-<4hex>` in UTC, e.g. `20261125T2130Z-collect-a3f9`. The stage must match `^[a-z][a-z0-9_]{1,20}$`.
- **Context manager:** `with run_context(stage) as run:` does the following:
  1. inserts a `runs` row with status `running`, the profile, `config_hash` and `git_sha` (or `unknown`)
  2. sets the `run_id`/`stage` contextvars the logger reads
  3. on exit, sets `finished_at` and the status (`ok` or `failed`)
  4. on an exception, stores the error, **raises a critical alert** and re-raises

## 7. Logging (D-015)
- **Setup:** `qt.common.logging.setup(config)` is called once per process.
- **Console handler:** human-readable text: `HH:MM:SS LEVEL stage run_id message key=val`.
- **File handler:** writes `logs/qt_YYYYMMDD.jsonl`, one JSON object per line: `ts` (UTC ISO), `level`, `logger`, `run_id`, `stage`, `event`, `msg`, plus any `extra=` fields. Files rotate at UTC midnight and are pruned after `retention_days`.
- **Helper:** `log.info("collected", extra={"feed": "fred", "rows": 412})`. Event names are short snake_case, because the summary and breakers parse them.
- **Daily summary writer:** `qt.common.reports.write_daily_summary(date)` writes `reports/daily/YYYY-MM-DD.md`. Phase 1 builds only the section scaffolding:
  - feed freshness
  - rows collected
  - runs
  - alerts
  - backups
  - predictions and orders, which stay empty until later phases

  Phase 2 fills in the content.

## 8. Alerts (D-016)
- **Transport:** push via **ntfy** (`POST {server}/{NTFY_TOPIC}`). `NTFY_TOPIC` is a long random string in `.env`, treated as a secret because anyone who knows the topic can read the messages. No email.
- **Every failure alerts** (Dom). The alert handler is attached to the root logger at `ERROR` and above, so any `log.error`, `log.exception` or failed `run_context` sends an alert without the caller doing anything extra. Explicit calls are also available: `alert(severity, event, title, body)`.

| Severity | ntfy priority | Triggers |
|---|---|---|
| `info` | 3 | **Every trade event:** order submitted, filled, partially filled, cancelled (Phase 6). The daily backup succeeding (one line) |
| `warn` | 4 | A retry that eventually succeeded, a feed nearing its staleness limit, slippage outliers |
| `critical` | 5 | Any `ERROR`/exception, failed run, stale critical feed, order rejected, breaker tripped, backup failure, backup row-count mismatch, scheduler not heartbeating |

- **No suppression:** failures are never suppressed, and there are no quiet hours for warn or critical.
- **Storm protection:** an identical `(event, title)` alert repeated inside `coalesce_window_s` is folded into one follow-up message, e.g. "×37 in last 5 min". So every failure is still reported, but a crash loop can't send hundreds of pushes.
- **Delivery failure:** if the ntfy POST fails (timeout, network down), the alert falls back to an OS notification (`osascript` on macOS, a PowerShell toast on Windows, behind a platform check). It is recorded in `alerts_log` with `delivered=false` and appears in the daily summary. Alert delivery must **never** raise into the caller.

## 9. Backups (D-017)
`uv run python scripts/db_backup.py`. Phase 1 builds it and the Phase 2 scheduler runs it daily at `backup.evening_time_et`, after the post-close pipeline.

1. Run `pg_dump -Fc` on `quant_trader` inside `qt_db` (via `docker compose exec -T`), writing to `data/backups/qt_YYYYMMDD.dump`. Dump files are fine on exFAT.
2. Refresh the backup DB in `qt_db_backup`:
   1. drop and recreate the database
   2. `CREATE EXTENSION timescaledb`
   3. `SELECT timescaledb_pre_restore()`
   4. `pg_restore` the dump
   5. `SELECT timescaledb_post_restore()`
3. Compare exact row counts per table (`count(*)` on hypertables, so not estimates) between primary and backup. Any mismatch is a **critical alert**.
4. Prune dump files beyond `keep_dumps` (newest 7 kept).
5. Write a `backup_log` row and send an `info` alert on success. Any failed step is a `critical` alert, and the old backup DB contents are lost only if the restore step itself fails mid-way. That's acceptable because the dump files remain.

The backup DB is a full, queryable copy as of last night, and each refresh doubles as a test restore. The dump files give 7 nights of point-in-time history. The Phase 12 Windows migration restores from the latest dump, so `scripts/db_backup.py` replaces the `db_dump.py` name in MASTER_PLAN.

## 10. Calendar (`qt.common.calendar`)
This module wraps `exchange_calendars` XNYS. All inputs and outputs are tz-aware, and naive datetimes raise.
- `is_session(date) -> bool`
- `sessions(start, end) -> list[date]`
- `session_open(date)` and `session_close(date)` return UTC datetimes and handle early closes.
- `next_session(date)` and `previous_session(date)`.
- `is_early_close(date) -> bool`.
- `to_et(dt)` and `to_utc(dt)`.
- `decision_time(session) -> datetime`: the UTC instant of the post-close cycle. It is close + 30 min, set in config, which handles early closes.

## 11. Mac runtime policy (D-018)
- **On the Mac:** only lightweight collectors run continuously, meaning news plus the daily `BAMLH0A0HYM2` archive and other cheap feeds. Features, models and trading run only as smoke tests until the Windows box is set up.
- **Phase 1 itself starts nothing continuously.** Phase 2 adds the scheduler and the launchd agent.

## 12. Dependencies (add with `uv add`, then pin in `uv.lock`)
- **Runtime:** `psycopg[binary,pool]`, `pydantic`, `pyyaml`, `python-dotenv`, `exchange_calendars`, `pandas`, `pyarrow`, `httpx`.
- **Dev:** `pytest`, `pytest-cov`, `ruff`.

Torch, sklearn and the rest wait for the phases that need them.

## 13. Tests (`uv run pytest`)
DB tests are marked `@pytest.mark.db`. They use a throwaway database `qt_test_<random>` created on `qt_db` and dropped afterwards, and they're skipped with a clear message if Docker isn't up.

| Test | Asserts |
|---|---|
| `test_migrate_empty_db` | All migrations apply on an empty DB; every spec table plus `alerts_log`, `backup_log`, `article_scores` and `schema_migrations` exists |
| `test_migrate_idempotent` | A second run applies nothing |
| `test_migrate_checksum_tamper` | Editing an applied file makes the runner fail |
| `test_hypertables` | `raw_prices`, `features` and `predictions` appear in `timescaledb_information.hypertables` with the configured chunk intervals |
| `test_raw_append_only` | UPDATE and DELETE on each raw table raise; INSERT works; a duplicate INSERT with `ON CONFLICT DO NOTHING` is a no-op |
| `test_config_merge` | The profile overrides only its keys; an unknown key fails; a missing required key fails |
| `test_config_hash_stable` | The same config gives the same hash, a changed value gives a different hash, and secrets don't affect the hash |
| `test_paper_guard` | A non-`PK` key id fails validation |
| `test_run_id_format` | Matches the regex, the time is UTC, two IDs in the same minute differ |
| `test_run_context` | Success → `ok` row; exception → `failed` row with the error, an alert fires (ntfy mocked), the exception is re-raised |
| `test_logging_jsonl` | Each line parses as JSON and carries `run_id`/`stage` from the context |
| `test_alert_on_error` | `log.error` triggers exactly one ntfy POST (mocked); identical errors within the window coalesce; ntfy down → OS fallback + `delivered=false`, never raises |
| `test_calendar` | Known holidays (2026-11-26 Thanksgiving closed), an early close (2026-11-27 closes 13:00 ET), the DST boundary, naive datetimes raise |
| `test_backup_roundtrip` (db, slow) | Seed rows → run the backup → the backup DB row counts match → the dump file exists → pruning keeps N |
| `test_paths_crossplatform` | No hard-coded `/` paths in `src/` (a grep test); all paths are built with `pathlib` from the repo root |

There is no no-lookahead test in Phase 1 because there are no features yet. The schema does provide `released_at` on every raw table, which the Phase 2 and 3 tests rely on.

## 14. Acceptance criteria (exit gate)
1. On a clean machine with Docker running, `docker compose up -d` brings up `qt_db` and `qt_db_backup`, and both are healthy.
2. `uv sync && uv run python -m qt.db.migrate` migrates an empty DB to head. Re-running it is a no-op.
3. `uv run python scripts/db_backup.py` produces a dump, refreshes the backup DB, reports matching row counts and sends the success alert.
4. A deliberate error sends a push to Dom's phone via ntfy, which Dom confirms once by hand.
5. `uv run pytest` is green, including the DB tests.
6. `.env.example` lists every new key (`POSTGRES_USER`, `POSTGRES_PASSWORD`, `QT_DB_PORT`, `QT_DB_BACKUP_PORT`, `BACKUP_DATABASE_URL`, `NTFY_TOPIC`, `QT_PROFILE`), and no secret is in git.
