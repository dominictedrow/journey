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

echo "Running all collectors..."
python3 -m collectors.fred      && echo "  fred done"
python3 -m collectors.bls       && echo "  bls done"
python3 -m collectors.prices    && echo "  prices done"
python3 -m collectors.crypto    && echo "  crypto done"
python3 -m collectors.etf_flows && echo "  etf_flows done"
python3 -m collectors.onchain   && echo "  onchain done"
python3 -m collectors.imf       && echo "  imf done"
python3 -m collectors.cftc      && echo "  cftc done"
python3 -m collectors.news      && echo "  news done"
echo "All collectors complete."
