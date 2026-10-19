"""Known-answer check of a local Kaggle dry run: local test.parquet is a copy of train (P1), so
logs/test_in_train.csv holds the true SMILES. Usage: python scripts/score_dryrun.py <submission.csv>"""
import csv
import sys
from collections import defaultdict

from casmi.metric import score

truth = defaultdict(set)
for r in csv.DictReader(open('logs/test_in_train.csv')):
    truth[r['molecule_id']].add(r['normalized_smiles'])
sub = {r['molecule_id']: r['smiles'] for r in csv.DictReader(open(sys.argv[1]))}
print(f'known-answer MRR@25 over {len(truth)} molecules: {score({k: ",".join(v) for k, v in truth.items()}, sub):.4f}')
