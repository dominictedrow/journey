# Architecture

High-level architecture of the Copytrader platform. The authoritative module-by-module design is
in [`plan.md`](plan.md); this document tracks what is actually built.

## Services (Docker Compose)

| Service    | Image source                | Role                                            |
| ---------- | --------------------------- | ----------------------------------------------- |
| `postgres` | `postgres:16`               | System of record (markets, wallets, trades, …)  |
| `redis`    | `redis:7`                   | Celery broker/result backend, kill-switch flag  |
| `api`      | `docker/backend.Dockerfile` | FastAPI app (REST + WebSocket)                  |
| `worker`   | `docker/worker.Dockerfile`  | Celery worker (ingest, analytics, copy engine)  |
| `beat`     | `docker/worker.Dockerfile`  | Celery beat (periodic jobs, optimizer schedule) |
| `frontend` | `docker/frontend.Dockerfile`| Next.js dashboard                               |

All services share one `core` Docker network. Configuration is env-driven through a single
`Settings` object (`backend/app/config.py`) — every external endpoint lives there and nowhere else.

## Backend modules

Each numbered module from `plan.md` maps to a package under `backend/app/`:

- `blockchain/` (M1) — on-chain indexer/listener, source of truth for trades
- `ingest/` — Gamma / Data-API / CLOB REST pulls
- `wallet_discovery/` (M2), `wallet_analytics/` (M3), `adaptive_intelligence/` (M4), `ranking/` (M5)
- `strategy_optimizer/` (M6), `backtesting/` (M7)
- `risk/` (M8), `copy_trading/` (M9), `portfolio/` (M10)
- `workers/` — Celery app + beat schedule + tasks

**Status:** Phase 0 — skeleton packages, data model, `/health`, infra. Modules are filled in per
the phased delivery plan.

## Data model

See `backend/app/database/models/` and the "Data model" section of `plan.md`. Conventions:
UTC timestamps, `NUMERIC` for money/quantity (never float), lowercase `VARCHAR(42)` addresses,
Alembic migration per phase (never edit a shipped migration).
