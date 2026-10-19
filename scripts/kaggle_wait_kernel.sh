#!/usr/bin/env bash
# Wait for a Kaggle kernel run to finish, then fetch its log/output: scripts/kaggle_wait_kernel.sh <owner/slug> [tries=60]
K=~/miniforge3/envs/casmi/bin/kaggle
for i in $(seq 1 "${2:-60}"); do
  s=$($K kernels status "$1" 2>&1)
  case "$s" in *COMPLETE*|*complete*|*ERROR*|*error*|*CANCEL*|*cancel*) break ;; esac
  sleep 15
done
echo "$s"
mkdir -p logs/kaggle_out && $K kernels output "$1" -p logs/kaggle_out -o 2>&1 | tail -n 3
