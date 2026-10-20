#!/usr/bin/env bash
# Detached background job: scripts/bg.sh <logname> <script.py|script.sh> [args...]  -> logs/<logname>.log, prints PID
cd ~/casmi
name=$1; shift
case "$1" in *.sh) run=(bash) ;; *) run=(~/miniforge3/envs/casmi/bin/python) ;; esac
PYTHONPATH=. setsid nohup "${run[@]}" "$@" > "logs/$name.log" 2>&1 < /dev/null &
echo "PID $! log logs/$name.log"
