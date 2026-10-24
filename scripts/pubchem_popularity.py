"""D-036 (gate 3): PubChem popularity prior per pool_v1 row, same order as cid.npy:
popularity.npy f16 = ln(1 + n_substances) + ln(1 + n_pubmed), 0 for rows with no CID.
Inputs data/external/pubchem/CID-SID.gz and CID-PMID.gz (scripts/pubchem_popularity.sh downloads them).
Usage: python scripts/pubchem_popularity.py"""
import hashlib
import json

import numpy as np
import pandas as pd

OUT = 'data/processed/pool_v1'
EXT = 'data/external/pubchem'


def counts(path, n):
    """Rows per CID (first column) of a PubChem CID-* gz table; CIDs above n-1 are dropped."""
    c = np.zeros(n, np.int32)
    for i, ch in enumerate(pd.read_csv(path, sep='\t', header=None, usecols=[0], dtype=np.uint32,
                                       chunksize=20_000_000)):
        u, k = np.unique(ch[0].to_numpy(), return_counts=True)
        keep = u < n
        c[u[keep]] += k[keep].astype(np.int32)  # u unique within a chunk, so += is safe
        print(path, i, flush=True)
    return c


def main():
    cid = np.load(f'{OUT}/cid.npy')
    n = int(cid.max()) + 1
    # ponytail: CID-PMID rows are (cid, pmid, type), so a pmid listed under 2 types counts twice; dedup if it matters
    pop = np.log1p(counts(f'{EXT}/CID-SID.gz', n)[cid]) + np.log1p(counts(f'{EXT}/CID-PMID.gz', n)[cid])
    pop[cid == 0] = 0
    pop = pop.astype(np.float16)
    np.save(f'{OUT}/popularity.npy', pop)
    m = json.load(open(f'{OUT}/manifest.json'))
    m['popularity'] = {'step': 'P7', 'decisions': ['D-036'], 'formula': 'ln1p(n_sid) + ln1p(n_pmid)',
                       'sha256': hashlib.sha256(open(f'{OUT}/popularity.npy', 'rb').read()).hexdigest(),
                       'nonzero': int((pop > 0).sum()), 'mean': float(pop.astype(np.float32).mean())}
    json.dump(m, open(f'{OUT}/manifest.json', 'w'), indent=1)
    print(json.dumps(m['popularity']))


if __name__ == '__main__':
    main()
