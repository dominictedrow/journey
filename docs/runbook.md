# Runbook

Day-2 operations for the Copytrader stack.

## Start / stop

```bash
# prod-style (detached)
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml down

# dev (hot reload, bind mounts)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up
```

`.env` at the repo root is loaded by every app service; start from `docker/.env.example`.

## Health & logs

```bash
curl http://localhost:8000/health
docker compose -f docker/docker-compose.yml ps
docker compose -f docker/docker-compose.yml logs -f api worker beat
```

## Database

```bash
# psql shell
docker compose -f docker/docker-compose.yml exec postgres psql -U copytrader copytrader

# apply migrations manually (api runs `alembic upgrade head` on start)
docker compose -f docker/docker-compose.yml run --rm api alembic upgrade head

# create a new migration (dev override bind-mounts backend/ so the file lands on the host)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml \
  run --rm api alembic revision --autogenerate -m "describe change"
```

Postgres data persists in the `pgdata` named volume. Back it up with `pg_dump` before upgrades.

## Kill switch (Phase 5+)

The global halt lives in the DB + Redis and is exposed via the `/risk` API and the dashboard.
When tripped, all execution (paper and live) stops until manually cleared.
