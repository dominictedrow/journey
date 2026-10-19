"""P1.5 plots from the arrays scripts/audit.py saved. Writes reports/audit_*.png."""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

tr, te = np.load('logs/audit_train.npz'), np.load('logs/audit_test.npz')


def hist(name, key, bins, xlabel, log=False, clip=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for d, lab in ((tr, 'train'), (te, 'test')):
        x = d[key][np.isfinite(d[key])]
        if clip is not None:
            x = np.clip(x, *clip)
        ax.hist(x, bins=bins, density=True, histtype='step', label=f'{lab} (n={len(x):,})')
    ax.set_xlabel(xlabel); ax.set_ylabel('density'); ax.legend()
    if log:
        ax.set_xscale('log')
    fig.tight_layout(); fig.savefig(f'reports/audit_{name}.png', dpi=110); plt.close(fig)


hist('peaks', 'npk', np.logspace(0, 4, 60), 'peaks per spectrum', log=True, clip=(1, 1e4))
hist('precursor_mz', 'prec', np.linspace(0, 1500, 75), 'precursor m/z', clip=(0, 1500))
hist('ce', 'ce', np.linspace(0, 120, 61), 'collision energy (first value, eV)', clip=(0, 120))

e = np.abs(tr['err']); e = e[np.isfinite(e)]
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.clip(e, 1e-3, 1e4), bins=np.logspace(-3, 4, 70))
for t in (5, 20):
    ax.axvline(t, ls='--', c='k', lw=0.8)
ax.set_xscale('log'); ax.set_xlabel('|precursor error| (ppm), train'); ax.set_ylabel('spectra')
fig.tight_layout(); fig.savefig('reports/audit_precursor_error.png', dpi=110); plt.close(fig)
print('ok')
