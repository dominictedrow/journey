# Questioning: Phase 1 · Infrastructure

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/01_infrastructure.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Database schema, Flow and schedule

1. Should Docker Desktop run on the Mac now, or do you want Postgres running natively during development? (Recommended: Docker, so the Mac and Windows setups match.)
   **A:** Docker. The TimescaleDB image is the same on Mac and Windows, and the data lives in a named volume, off exFAT.
2. Do we use TimescaleDB hypertables for `raw_prices`, `features` and `predictions` from the start, or plain Postgres first?
   **A:** Hypertables from the start for `raw_prices`, `features` and `predictions`. All other tables (assets, runs, orders, news) are plain tables.
3. Should the `features` table be long format (asof, asset_id, feature_name, value) as in the spec, or wide Parquet snapshots for training speed? Or both: the DB as the record and Parquet as a cache?
   **A:** Both, option (c). The DB keeps the long-format table and is the record. Wide Parquet snapshots, tagged with feature version and `asof` range, are a training cache that can be rebuilt from the DB, so they are never the source of truth.
4. Which migration tool do we use: plain numbered `.sql` files with a tiny runner, or Alembic?
   **A:** Plain numbered `.sql` files plus a small Python runner, which records applied files in `schema_migrations`. No Alembic: there's no ORM, so autogenerate gives us nothing, and the TimescaleDB DDL is raw SQL anyway. Rollback means writing a new forward migration.
5. Config layout: one `configs/*.yaml` per component. Is that OK? Do you want named profiles (dev, paper, backtest)?
   **A:** One base YAML per component, plus thin profile overlays in `configs/profiles/{dev,paper,backtest}.yaml` that hold only the keys that change. `QT_PROFILE` picks the profile, with `dev` as the default. The loader deep-merges base and profile, then validates with pydantic, so a typo fails at startup. Each run records a hash of the resolved config with its `run_id`. Secrets stay in `.env` only.
6. What are the logging requirements: plain files in `logs/`, JSON lines, and do you want a daily summary report (Markdown, email, or a dashboard)?
   **A:** Both formats. Human-readable text goes to the console. `logs/qt_YYYYMMDD.jsonl` gets JSON lines with `run_id`, `stage`, `level` and the event fields, so the summary and breakers can parse the logs without regex. Files rotate daily and retention is set in config (default 30 days). A Markdown daily summary goes to `reports/daily/YYYY-MM-DD.md` and covers feed freshness, rows collected, predictions, orders, fills, P&L and breakers. A dashboard is deferred to Phase 10/11, and alerts are covered in Q10.
7. Which `run_id` format do you want (UUID, or `YYYYMMDD-HHMM-<stage>`)?
   **A:** No preference, so we use the recommendation: `YYYYMMDDTHHMMZ-<stage>-<4hex>` in UTC, e.g. `20261125T2130Z-collect-a3f9`. It sorts by time, is readable, and the suffix prevents collisions.
8. How should DB backups work (nightly `pg_dump` into `data/backups/`, and how many to keep)?
   **A (revised):** Backups go into a **separate database**, not a status page. It is refreshed **once each evening**, after the post-close pipeline finishes (default ~21:00 America/New_York, set in config), so it never competes with the trading or collection runtime.
   - A second Postgres/TimescaleDB container, `qt_db_backup`, with its own named volume, so the backup doesn't share the primary's volume.
   - Evening job: `pg_dump -Fc` the primary, then `pg_restore --clean` into `qt_db_backup`. Each refresh doubles as a test restore. The job compares per-table row counts between primary and backup, and a mismatch is logged as an error.
   - It also keeps the nightly `.dump` files in `data/backups/qt_YYYYMMDD.dump`, the last 7 by default, set in config. They give point-in-time history beyond one night, in case bad data gets in.
9. Should the Mac run the daily pipeline 24/7 until the Windows move? Do we need it to survive sleep, and wake-from-sleep handling?
   **A:** The full pipeline doesn't run 24/7 on the Mac. Only **lightweight collectors** run continuously there: news, plus the daily `BAMLH0A0HYM2` archive (D-001) and other cheap feeds. Features, models and trading wait for the Windows box; the Mac only runs smoke tests for those. Collectors run under an APScheduler process supervised by a launchd agent (Task Scheduler on Windows; the Python code is the same). On wake, collectors backfill missed windows, and no trading decision is ever replayed. DB connections pre-ping and reconnect (news_bot `db.py` pattern).
10. Do you need an alert channel when a cycle fails or a breaker trips (macOS notification, email, push)?
   **A:** Push notifications to the **ntfy app**. No email, because Dom's inbox is too noisy. **Every failure anywhere in the system raises an alert**, since any failure means system health has degraded. That covers collector errors, stale critical feeds, backup or row-count failures, breaker trips, order rejects and unhandled exceptions. Dom also wants an **alert on every trade**: order submitted, filled, cancelled or rejected. Severity levels (info for trades, warn and critical for failures) set the ntfy priority. Failures are never suppressed by quiet hours. A local OS notification and a line in the daily summary serve as the fallback if the push fails.
