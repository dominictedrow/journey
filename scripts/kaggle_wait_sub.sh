#!/usr/bin/env bash
# Wait for the latest competition submission to leave PENDING: scripts/kaggle_wait_sub.sh [tries=40]
K=~/miniforge3/envs/casmi/bin/kaggle
C=enveda-CASMI26-molecule-id-mass-spectra
for i in $(seq 1 "${1:-40}"); do
  s=$($K competitions submissions $C 2>/dev/null | grep -m1 "^ *[0-9]")
  case "$s" in *PENDING*) sleep 15 ;; *) break ;; esac
done
echo "$s"
