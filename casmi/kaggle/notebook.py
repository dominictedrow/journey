"""Kaggle kernel (script): offline install -> features on the fly -> rank -> submission.csv, with per-stage timings.
Locally: KAGGLE_INPUT=<dir with bundle + competition data> KAGGLE_WORK=<out dir> python casmi/kaggle/notebook.py"""
import glob
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
INPUT = os.environ.get('KAGGLE_INPUT', '/kaggle/input')
WORK = os.environ.get('KAGGLE_WORK', '/kaggle/working')


def find(pattern):
    hits = sorted(glob.glob(f'{INPUT}/**/{pattern}', recursive=True))
    assert hits, f'{pattern} not found under {INPUT}'
    return hits[0]


print(sys.version, flush=True)
bundle = os.path.dirname(find('casmi_bundle.json'))
wheels = f'{bundle}/wheels'
if os.path.isdir(wheels):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--find-links', wheels,
                        'rdkit==2026.3.3'], capture_output=True, text=True)
    print('rdkit install:', 'ok' if r.returncode == 0 else r.stderr[-500:], flush=True)
# numba cache=True needs a writable source dir; /kaggle/input is read-only
shutil.copytree(f'{bundle}/code', f'{WORK}/code', dirs_exist_ok=True)
sys.path.insert(0, f'{WORK}/code')

from casmi.kaggle.predict import predict, validate  # noqa: E402

test = find('test.parquet')
sub = f'{WORK}/submission.csv'
timings, _ = predict(test, f'{bundle}/library.npz', sub)
n = validate(sub, test)
timings['total'] = round(time.time() - T0, 2)
print(f'submission.csv: {n} molecules; timings {timings}', flush=True)
shutil.rmtree(f'{WORK}/code')  # keep only submission.csv in the output
