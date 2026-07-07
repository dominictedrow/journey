# Build context is the repo root: docker build -f docker/backend.Dockerfile .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies against a stub package first so source edits don't bust the pip layer.
COPY backend/pyproject.toml ./
RUN mkdir -p app && touch app/__init__.py && pip install -e .[dev]
COPY backend/ ./

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
