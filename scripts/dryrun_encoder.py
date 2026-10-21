"""Local dry run of the P5.4 Kaggle path on data/test.parquet: submission format check + known-answer MRR.
Usage: python scripts/dryrun_encoder.py <ckpt> [workers=4]"""
import json
import subprocess
import sys

from casmi.kaggle.predict import predict_encoder, validate

ck = sys.argv[1]
w = int(sys.argv[2]) if len(sys.argv) > 2 else 4
tm, _ = predict_encoder('data/test.parquet', 'data/kaggle/bundle/library.npz', 'data/processed/pool_v1', ck,
                        'data/processed/v1/vocab.json', 'logs/dryrun_encoder.csv', workers=w)
print(json.dumps(tm), 'rows', validate('logs/dryrun_encoder.csv', 'data/test.parquet'), flush=True)
subprocess.run([sys.executable, 'scripts/score_dryrun.py', 'logs/dryrun_encoder.csv'])
