"""P3.2: build the Kaggle dataset bundle + kernel dir under data/kaggle/ (gitignored).
bundle/: code/casmi (source + npz), library.npz (all reference spectra: train + val queries), wheels/ (pinned RDKit),
casmi_bundle.json (git sha), dataset-metadata.json, and with a checkpoint: model.pt + vocab.json (P5.4).
pool/: hard links to pool_v1 inference files (mass/ik14/src .npy, manifest.json, pool/bin_*.parquet) for the separate
casmi26-pool dataset.  kernel/: notebook.py + kernel-metadata.json.
Usage: python -m casmi.kaggle.build_bundle [ckpt]   (then 🛑 gate 7 before `kaggle datasets create/version`)"""
import json
import os
import shutil
import subprocess
import sys

from casmi.data.splits import VAL
from casmi.retrieval.library import Library

USER = 'dominictedrow'
COMP = 'enveda-CASMI26-molecule-id-mass-spectra'
DATASET = f'{USER}/casmi26-bundle'
POOL_DATASET = f'{USER}/casmi26-pool'
POOL = 'data/processed/pool_v1'
KERNEL = f'{USER}/casmi26-submit'
OUT = 'data/kaggle'
RDKIT = 'rdkit==2026.3.3'
PYS = ('3.11', '3.12', '3.13')  # Kaggle image version unknown; pip picks the matching wheel


def main():
    b, k = f'{OUT}/bundle', f'{OUT}/kernel'
    shutil.rmtree(f'{b}/code', ignore_errors=True)
    shutil.copytree('casmi', f'{b}/code/casmi', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    lib = Library('data/processed/v1', ('train',) + VAL)  # submission library: everything, nothing excluded
    if not os.path.exists(f'{b}/library.npz') or os.path.getmtime(f'{b}/library.npz') < os.path.getmtime(lib.path):
        shutil.copy(lib.path, f'{b}/library.npz')
    ck = sys.argv[1] if len(sys.argv) > 1 else None
    if ck:
        import torch
        sd = torch.load(ck, map_location='cpu')['model']
        torch.save({'model': {k.replace('_orig_mod.', ''): v for k, v in sd.items()}, 'source': ck}, f'{b}/model.pt')
        shutil.copy('data/processed/v1/vocab.json', f'{b}/vocab.json')
        p = f'{OUT}/pool'
        os.makedirs(f'{p}/pool', exist_ok=True)
        for f in ['mass.npy', 'ik14.npy', 'src.npy', 'manifest.json'] + [f'pool/{x}' for x in os.listdir(f'{POOL}/pool')]:
            if not os.path.exists(f'{p}/{f}'):
                os.link(f'{POOL}/{f}', f'{p}/{f}')
        json.dump({'title': 'casmi26-pool', 'id': POOL_DATASET, 'licenses': [{'name': 'CC0-1.0'}]},
                  open(f'{p}/dataset-metadata.json', 'w'), indent=1)
    for py in PYS:
        subprocess.run([sys.executable, '-m', 'pip', 'download', '-q', '--no-deps', '--only-binary=:all:', '--python-version', py,
                        '--platform', 'manylinux_2_28_x86_64', '--platform', 'manylinux2014_x86_64',
                        '-d', f'{b}/wheels', RDKIT], check=True)
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', 'casmi'], capture_output=True, text=True).stdout)
    json.dump({'git_sha': sha, 'dirty': dirty, 'library': os.path.basename(lib.path), 'model': ck}, open(f'{b}/casmi_bundle.json', 'w'))
    json.dump({'title': 'casmi26-bundle', 'id': DATASET, 'licenses': [{'name': 'CC0-1.0'}]},
              open(f'{b}/dataset-metadata.json', 'w'), indent=1)
    os.makedirs(k, exist_ok=True)
    shutil.copy('casmi/kaggle/notebook.py', f'{k}/notebook.py')
    json.dump({'id': KERNEL, 'title': 'casmi26-submit', 'code_file': 'notebook.py', 'language': 'python',
               'kernel_type': 'script', 'is_private': True, 'enable_gpu': False, 'enable_internet': False,
               'dataset_sources': [DATASET] + ([POOL_DATASET] if ck else []), 'competition_sources': [COMP], 'kernel_sources': []},
              open(f'{k}/kernel-metadata.json', 'w'), indent=1)
    print(json.dumps({'git_sha': sha, 'dirty': dirty, 'wheels': sorted(os.listdir(f'{b}/wheels'))}))


if __name__ == '__main__':
    main()
