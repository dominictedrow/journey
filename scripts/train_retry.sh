# Rerun train_encoder.py (which resumes from runs/<name>/state.pt) until it exits 0; max 10 attempts.
# Usage: bash scripts/bg.sh <log> scripts/train_retry.sh --name X [args...]
cd ~/envenda
for i in $(seq 10); do
  PYTHONPATH=. ~/miniforge3/envs/casmi/bin/python scripts/train_encoder.py "$@" && exit 0
  echo "attempt $i failed ($(date)); retrying in 30s"
  sleep 30
done
exit 1
