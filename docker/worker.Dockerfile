# Celery worker/beat image — same base as the API, different command.
# Build context is the repo root: docker build -f docker/worker.Dockerfile .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies against a stub package first so source edits don't bust the pip layer.
COPY backend/pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py && pip install -e .[dev]
COPY backend/ ./

# --concurrency=2: the outbound-API throttles (app/ingest/http.py) are per-process, so the
# pool size is part of the rate-limit math — 2 processes keeps the live tail responsive while
# a long backfill runs, without multiplying request rates by the host's CPU count.
CMD ["celery", "-A", "app.workers.celery_app:celery_app", "worker", "--loglevel=INFO", "--concurrency=2"]
