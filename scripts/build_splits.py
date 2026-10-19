"""P2.3: freeze val splits per ik14 -> data/processed/v1/splits.parquet (D-019; [2M...] excluded, D-020).
Order: scaffold slice first, then V-lib (enveda & other libs), then V-pool / V-novel (enveda only).
v_scaf: enveda molecules of held-out scaffolds; 'held': non-enveda molecules of those scaffolds (no train, no queries)."""
import hashlib
import json
import os
import random
import subprocess
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

OUT = 'data/processed/v1/splits.parquet'
SEED = 2026
N_LIB, N_POOL, N_NOVEL, N_SCAF = 1000, 2000, 1000, 1000
MAX_SCAF_SIZE = 20  # ponytail: only small scaffolds held out so one benzene-sized scaffold can't eat the train set


def main():
    if os.path.exists(OUT):
        print(f'{OUT} exists; splits are frozen (changing them is a gate)')
        return
    s = pq.read_table('data/processed/v1/structures.parquet', columns=['smiles', 'ik14', 'scaffold'])
    ik = dict(zip(s['smiles'].to_pylist(), s['ik14'].to_pylist()))
    scaf = {}
    for k, sc in zip(s['ik14'].to_pylist(), s['scaffold'].to_pylist()):
        scaf.setdefault(k, sc)
    env, oth = Counter(), Counter()
    for b in pq.ParquetFile('data/train.parquet').iter_batches(batch_size=262144, columns=['normalized_smiles', 'ingest_lib', 'adduct']):
        for smi, lib, a in zip(*(b.column(i).to_pylist() for i in range(3))):
            if a and a.startswith('[2M'):
                continue
            (env if lib.startswith('enveda') else oth)[ik[smi]] += 1
    mols = sorted(set(env) | set(oth))
    rng = random.Random(SEED)
    split = dict.fromkeys(mols, 'train')

    by_scaf = defaultdict(list)
    for k in mols:
        by_scaf[scaf[k]].append(k)
    cand = sorted(sc for sc, ks in by_scaf.items() if sc and len(ks) <= MAX_SCAF_SIZE and any(k in env for k in ks))
    rng.shuffle(cand)
    n = 0
    for sc in cand:
        if n >= N_SCAF:
            break
        for k in by_scaf[sc]:
            split[k] = 'v_scaf' if k in env else 'held'
            n += k in env

    free = [k for k in mols if split[k] == 'train']
    both = [k for k in free if k in env and k in oth]
    only = [k for k in free if k in env and k not in oth]
    for k in rng.sample(both, N_LIB):
        split[k] = 'v_lib'
    pick = rng.sample(only, N_POOL + N_NOVEL)
    for k in pick[:N_POOL]:
        split[k] = 'v_pool'
    for k in pick[N_POOL:]:
        split[k] = 'v_novel'

    tbl = pa.table({
        'ik14': mols, 'split': [split[k] for k in mols], 'scaffold': [scaf[k] for k in mols],
        'n_enveda': pa.array([env[k] for k in mols], pa.int32()), 'n_other': pa.array([oth[k] for k in mols], pa.int32()),
    })
    pq.write_table(tbl, OUT, compression='zstd')
    mol_counts = Counter(split.values())
    q = Counter()
    for k in mols:
        q[split[k]] += env[k]
    stats = {'molecules': dict(mol_counts), 'enveda_spectra_by_split': dict(q),
             'v_scaf_scaffolds': len({scaf[k] for k in mols if split[k] == 'v_scaf'})}
    mp = 'data/processed/v1/manifest.json'
    m = json.load(open(mp))
    m['splits'] = {
        'step': 'P2.3', 'decisions': ['D-019', 'D-020'], 'script': 'scripts/build_splits.py',
        'config': {'seed': SEED, 'n_lib': N_LIB, 'n_pool': N_POOL, 'n_novel': N_NOVEL, 'n_scaf': N_SCAF, 'max_scaf_size': MAX_SCAF_SIZE, 'drop_2M': True},
        'git_sha': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
        'files': {'splits.parquet': hashlib.sha256(open(OUT, 'rb').read()).hexdigest()}, 'stats': stats,
    }
    json.dump(m, open(mp, 'w'), indent=2)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
