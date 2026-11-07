#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -f .scheduler.pid ]; then
    kill "$(cat .scheduler.pid)" 2>/dev/null && echo "Scheduler stopped." || echo "Scheduler not running."
    rm -f .scheduler.pid
fi

if [ -f .flask.pid ]; then
    kill "$(cat .flask.pid)" 2>/dev/null && echo "Flask stopped." || echo "Flask not running."
    rm -f .flask.pid
fi
