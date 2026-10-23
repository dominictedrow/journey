"""P6 contrastive retrieval (D-029): the P5 Encoder's pooled molecule vector -> 256-d projection; candidate molecules ->
256-d via an MLP over the 2048-bit fingerprint + log1p element counts, so candidate embeddings need only fp + formula.
InfoNCE (learnable temperature) over in-batch molecules + mined hard negatives; P5 focal fp + formula losses stay as
auxiliaries so the fp head keeps serving the P6.3 Tanimoto blend."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from casmi.features.fp import FP_BITS
from casmi.models.encoder import N_EL, Encoder, loss_fn, mlp, unpack_fp

D_EMB = 256


class MolEncoder(nn.Module):
    def __init__(self, d=1024, o=D_EMB):
        super().__init__()
        self.net = mlp(FP_BITS + N_EL, d, o)

    def forward(self, bits, counts):
        """bits float [..., 2048], counts int [..., 10] -> L2-normalised [..., D_EMB]."""
        return Fn.normalize(self.net(torch.cat([bits, torch.log1p(counts.float().clamp(min=0))], -1)).float(), dim=-1)


class Contrastive(nn.Module):
    def __init__(self, vocab_sizes, **kw):
        super().__init__()
        self.enc = Encoder(vocab_sizes, **kw)
        d = self.enc.norm.normalized_shape[0]
        self.proj = nn.Linear(d, D_EMB)
        self.mol = MolEncoder()
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, b):
        """-> (fp logits [M, 2048], formula [M, 10], spectrum-side embedding [M, D_EMB] L2-normalised)."""
        m = self.enc.molecules(b)
        return self.enc.fp_head(m), self.enc.formula_head(m), Fn.normalize(self.proj(m).float(), dim=-1)


def nce_loss(model, z, b):
    """InfoNCE of spectrum embeddings z [M, E] against [in-batch positives | own hard negatives]. In-batch molecules with
    the same packed fp as the positive (stereoisomers) and padded negatives are masked out."""
    pos = model.mol(unpack_fp(b['mol_fp']), b['pos_formula'])
    M, K = b['neg_fp'].shape[:2]
    neg = model.mol(unpack_fp(b['neg_fp'].flatten(0, 1)), b['neg_formula'].flatten(0, 1)).view(M, K, -1)
    s = model.logit_scale.exp().clamp(max=100)
    inb = s * z @ pos.T
    same = (b['mol_fp'][:, None] == b['mol_fp'][None]).all(-1) & ~torch.eye(M, dtype=torch.bool, device=z.device)
    hard = (s * (z[:, None] * neg).sum(-1)).masked_fill(torch.arange(K, device=z.device)[None] >= b['n_neg'][:, None],
                                                        float('-inf'))
    logits = torch.cat([inb.masked_fill(same, float('-inf')), hard], 1)
    return Fn.cross_entropy(logits, torch.arange(M, device=z.device))


def total_loss(model, out, b, nce_w=1.0):
    fp, form, z = out
    aux, parts = loss_fn(fp, form, b)
    nce = nce_loss(model, z, b)
    parts['nce'] = nce.detach()
    return aux + nce_w * nce, parts
