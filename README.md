# Copytrader — Polymarket Intelligence & Copy-Trading Platform

Discover Polymarket wallets, score them quantitatively, backtest portfolio strategies, and execute
copy trades under strict risk controls. Paper trading is the default; live execution is gated
behind explicit configuration and a kill switch.

The full implementation plan lives in [`docs/plan.md`](docs/plan.md).

## Stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy 2 (async), Alembic, Celery + Redis, PostgreSQL 16
- **Frontend**: Next.js 14 (App Router), TypeScript, Tailwind CSS, TanStack Query, zustand
- **Desktop**: Tauri 2 thin client (Phase 7) with a configurable backend URL
- **Infra**: Docker Compose — postgres, redis, api, worker, beat, frontend

## Quickstart

```bash
cp docker/.env.example .env          # fill in RPC keys etc. as they become needed
docker compose -f docker/docker-compose.yml up -d --build
curl http://localhost:8000/health    # -> {"status":"ok",...}
open http://localhost:3000           # dashboard; set backend URL under /settings
```

Development mode (hot reload, bind mounts):

```bash
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up
```

## Repository layout

- `backend/` — FastAPI app, workers, tests (`backend/app/<module>/` per plan module)
- `frontend/` — Next.js dashboard
- `desktop/` — Tauri wrapper (placeholder until Phase 7)
- `docker/` — Dockerfiles, compose files, `.env.example`
- `docs/` — plan, architecture, API, runbook, deploy notes
- `scripts/` — ops helpers and seed scripts

## Checks

Backend checks run inside the Docker image (no host Python needed):

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps api pytest
docker compose -f docker/docker-compose.yml run --rm --no-deps api ruff check .
docker compose -f docker/docker-compose.yml run --rm --no-deps api black --check .
docker compose -f docker/docker-compose.yml run --rm --no-deps api mypy app
```

Frontend checks:

```bash
cd frontend && npm run lint && npx tsc --noEmit
```
