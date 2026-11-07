#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

mkdir -p backups
FILENAME="backups/market_research_$(date +%Y%m%d).sql"
pg_dump "$DATABASE_URL" > "$FILENAME"
echo "Backup written to $FILENAME"
