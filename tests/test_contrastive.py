import numpy as np
import torch

from casmi.features.formula import fkey_counts, formula_key
from casmi.models.contrastive import D_EMB, MolEncoder, nce_loss


def test_fkey_roundtrip():
    v = np.array([[18, 25, 5, 1, 0, 0, 0, 0, 0, 0], [63, 40, 2, 9, 1, 1, 3, 0, 1, 0]], np.int16)
    assert (fkey_counts(formula_key(v)) == v).all()


def test_nce_masks():
    torch.manual_seed(0)
    m = torch.nn.Module()
    m.mol, m.logit_scale = MolEncoder(), torch.nn.Parameter(torch.tensor(2.0))
    M, K = 4, 3
    fp = torch.randint(0, 256, (M, 256), dtype=torch.uint8)
    fp[1] = fp[0]  # stereoisomer pair: must not be each other's negative
    b = {'mol_fp': fp, 'pos_formula': torch.randint(0, 9, (M, 10)),
         'neg_fp': torch.randint(0, 256, (M, K, 256), dtype=torch.uint8), 'neg_formula': torch.randint(0, 9, (M, K, 10)),
         'n_neg': torch.tensor([3, 0, 1, 2])}
    z = torch.nn.functional.normalize(torch.randn(M, D_EMB), dim=-1)
    loss = nce_loss(m, z, b)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.logit_scale.grad is not None
