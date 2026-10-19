#!/usr/bin/env bash
# Detached background job: scripts/bg.sh <logname> <python args...>  -> logs/<logname>.log, prints PID
cd ~/casmi
name=$1; shift
PYTHONPATH=. setsid nohup ~/miniforge3/envs/casmi/bin/python "$@" > "logs/$name.log" 2>&1 < /dev/null &
echo "PID $! log logs/$name.log"
