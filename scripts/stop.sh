#!/usr/bin/env bash
# Stop background jobs whose command line matches a pattern: scripts/stop.sh <pattern>
# (pkill -f from `wsl -e bash -lc "..."` also matches the calling shell, whose argv contains the pattern)
pids=$(pgrep -f "$1" | grep -vx "$$")
[ -n "$pids" ] && kill $pids
sleep 3
echo "still running: $(pgrep -fc "$1" | head -1)"
