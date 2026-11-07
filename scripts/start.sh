#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export DATABASE_URL="${DATABASE_URL:-postgresql://localhost/market_research}"

python3 db/init.py

python3 -m collectors.scheduler &
echo $! > .scheduler.pid
echo "Scheduler started (PID $(cat .scheduler.pid))"

FLASK_APP="app:create_app()" python3 -m flask run --port "${PORT:-5000}" &
echo $! > .flask.pid
echo "Flask started (PID $(cat .flask.pid)) on port ${PORT:-5000}"
