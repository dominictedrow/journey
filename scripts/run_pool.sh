#!/usr/bin/env bash
# P4.2 full pool build, sequential, resumable (finished chunks are skipped). Launch: scripts/bg.sh pool_v1 scripts/run_pool.sh
set -euo pipefail
cd ~/envenda
PY=~/miniforge3/envs/casmi/bin/python
for s in train lotus coconut pubchem; do PYTHONPATH=. $PY scripts/build_pool.py ingest $s; done
rm -rf data/processed/pool_v1/pool  # merge output only; raw/ chunks are kept
PYTHONPATH=. $PY scripts/build_pool.py merge
echo "ALL DONE $(date +%T)"
