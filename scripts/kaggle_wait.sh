#!/usr/bin/env bash
# Wait until a Kaggle dataset is ready: scripts/kaggle_wait.sh <owner/slug> [tries=40]
K=~/miniforge3/envs/casmi/bin/kaggle
for i in $(seq 1 "${2:-40}"); do
  s=$($K datasets status "$1" 2>&1)
  case "$s" in *ready*|*error*) break ;; esac
  sleep 15
done
echo "$s"
$K datasets files "$1" 2>&1 | head -12
