#!/bin/sh
# Dom's kill switch on the Mac: stops Claude driving the game (same as tools\stop.cmd).
# Writes harness/logs/STOP and shuts the drive server down; only Dom deletes logs/STOP.
here="$(cd "$(dirname "$0")" && pwd)"
py="$here/../../.venv/bin/python"
[ -x "$py" ] || py=python3
mkdir -p "$here/../logs" && echo stop > "$here/../logs/STOP"   # even if python fails
exec "$py" "$here/drive.py" stop
