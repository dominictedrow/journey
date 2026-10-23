#!/usr/bin/env bash
# Detached background job: scripts/bg.sh <logname> <script.py|script.sh> [args...]  -> logs/<logname>.log, prints PID
cd ~/envenda
name=$1; shift
case "$1" in *.sh) run=(bash) ;; *) run=(~/miniforge3/envs/casmi/bin/python) ;; esac
PYTHONPATH=. setsid nohup "${run[@]}" "$@" > "logs/$name.log" 2>&1 < /dev/null &
echo "PID $! log logs/$name.log"
sleep 5  # if the wsl call returns instantly, WSL can kill the child before it detaches
