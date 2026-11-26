# Questioning: Phase 1 · Infrastructure

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/01_infrastructure.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Database schema, Flow and schedule

1. Should Docker Desktop run on the Mac now, or do you want Postgres running natively during development? (Recommended: Docker, so the Mac and Windows setups match.)
2. Do we use TimescaleDB hypertables for `raw_prices`, `features` and `predictions` from the start, or plain Postgres first?
3. Should the `features` table be long format (asof, asset_id, feature_name, value) as in the spec, or wide Parquet snapshots for training speed? Or both: the DB as the record and Parquet as a cache?
4. Which migration tool do we use: plain numbered `.sql` files with a tiny runner, or Alembic?
5. Config layout: one `configs/*.yaml` per component. Is that OK? Do you want named profiles (dev, paper, backtest)?
6. What are the logging requirements: plain files in `logs/`, JSON lines, and do you want a daily summary report (Markdown, email, or a dashboard)?
7. Which `run_id` format do you want (UUID, or `YYYYMMDD-HHMM-<stage>`)?
8. How should DB backups work (nightly `pg_dump` into `data/backups/`, and how many to keep)?
9. Should the Mac run the daily pipeline 24/7 until the Windows move? Do we need it to survive sleep, and wake-from-sleep handling?
10. Do you need an alert channel when a cycle fails or a breaker trips (macOS notification, email, push)?
