import json

import torch

from casmi.data.loader import make_loader
from casmi.models.encoder import Encoder, feature_stats, loss_fn, sinusoid

ROOT = 'data/processed/v1'


def test_sinusoid_keeps_mda_precision():
    a, b = sinusoid(torch.tensor([1500.0000], dtype=torch.float64)), sinusoid(torch.tensor([1500.0005], dtype=torch.float64))
    assert (a - b).abs().max() > 0.5  # 0.5 mDa apart must still differ at the 1 mDa wavelength


def test_forward_backward_on_val_batch():
    vocab = json.load(open(f'{ROOT}/vocab.json'))
    model = Encoder({k: len(v) for k, v in vocab.items()}, d=64, layers=2, heads=4)
    model.set_stats(*feature_stats(ROOT, n=2000))
    b = next(iter(make_loader(ROOT, 'v_pool', workers=0, shuffle=False, max_spec_per_mol=None, spec_per_batch=64)))
    fp, form = model(b)
    M = len(b['mol_idx'])
    assert fp.shape == (M, 2048) and form.shape == (M, 10)
    loss, _ = loss_fn(fp, form, b)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
