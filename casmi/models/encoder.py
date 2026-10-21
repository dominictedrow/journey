"""P5 spectrum encoder (PLAN P5): one token per peak plus one precursor token per spectrum, no positional encoding,
pre-LN transformer, attention pooling over a molecule's spectra, 2048-bit Morgan fingerprint head (focal BCE) and a
formula-consistency auxiliary head (element counts of the molecular formula)."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from casmi.features.fp import FP_BITS
from casmi.features.peaks import F, FEATS, LOSS_MASS

N_FREQ = 32           # sinusoid frequencies per mass encoding
WL = (1e-3, 1e3)      # wavelength range in Da (PLAN P5): keeps exact-mass precision
LOSS_ID = FEATS.index('loss_id')
N_EL = 10


def sinusoid(x, n=N_FREQ, wl=WL):
    """Masses (any shape, Da) -> [..., 2n]. Phases in float64: m/z 2000 at a 1 mDa wavelength is ~1e7 rad."""
    w = torch.logspace(math.log10(wl[0]), math.log10(wl[1]), n, device=x.device, dtype=torch.float64)
    ph = torch.remainder(x.double()[..., None] * (2 * math.pi / w), 2 * math.pi)
    return torch.cat([ph.sin(), ph.cos()], -1).float()


def mlp(i, d, o):
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Linear(d, o))


class Encoder(nn.Module):
    def __init__(self, vocab_sizes, d=512, layers=8, heads=8, k_formulas=16, dropout=0.1):
        super().__init__()
        self.register_buffer('stats', torch.zeros(2, F + 1))  # mean/std of [peakfeat, sqrt inten]; set_stats()
        self.peak = mlp(4 * N_FREQ + F + 1 + 32, d, d)
        self.loss_emb = nn.Embedding(len(LOSS_MASS) + 1, 32)
        self.prec = mlp(2 * N_FREQ + k_formulas * (N_EL + 4) + 2 + 1 + 6, d, d)
        self.adduct = nn.Embedding(vocab_sizes['adduct'] + 1, d)
        self.instrument = nn.Embedding(vocab_sizes['instrument_type'] + 1, d)
        self.polarity = nn.Embedding(vocab_sizes['ionization_mode'] + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation='gelu', batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.pool = nn.Linear(d, 1)
        self.fp_head = mlp(d, 2 * d, FP_BITS)
        self.formula_head = mlp(d, d, N_EL)

    @torch.no_grad()
    def set_stats(self, peakfeat, sqrt_inten):
        x = torch.cat([peakfeat.float(), sqrt_inten.float()[:, None]], 1)
        self.stats.copy_(torch.stack([x.mean(0), x.std(0).clamp(min=1e-3)]))

    def spectra(self, b):
        """Per-spectrum embeddings [B, d] (precursor-token output)."""
        mz, L = b['mz'], b['mz'].shape[1]
        pad = torch.arange(L, device=mz.device)[None] >= b['npeaks'][:, None]
        prec = b['prec']
        cont = torch.cat([b['peakfeat'].float(), b['inten'].float().clamp(min=0).sqrt()[..., None]], -1)
        cont = ((cont - self.stats[0]) / self.stats[1]).masked_fill(pad[..., None], 0)
        loss_id = b['peakfeat'][..., LOSS_ID].float().round().long().clamp(0, len(LOSS_MASS))
        peaks = self.peak(torch.cat([sinusoid(mz), sinusoid(prec[:, None] - mz), cont, self.loss_emb(loss_id)], -1))
        nm = b['neutral_mass']
        known = torch.isfinite(nm)
        fs = torch.stack([b['formula_score'], b['formula_ppm'] / 10, b['formula_iso_score'], b['formula_rdbe'] / 10], -1)
        fmask = (torch.arange(fs.shape[1], device=mz.device)[None] < b['nformulas'][:, None])[..., None]
        form = torch.cat([b['formulas'].float() / 10, torch.nan_to_num(fs)], -1) * fmask
        ce = torch.nan_to_num(b['ce'], nan=-1.0) / 50
        p = torch.cat([sinusoid(torch.where(known, nm, prec)), form.flatten(1), ce, known[:, None].float(),
                       torch.nan_to_num(b['iso'])], -1)
        tok = self.prec(p) + self.adduct(b['adduct']) + self.instrument(b['instrument']) + self.polarity(b['polarity'])
        x = torch.cat([tok[:, None], peaks], 1)
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        return self.norm(self.enc(x, src_key_padding_mask=mask)[:, 0])

    def forward(self, b):
        """-> (fp logits [M, 2048], formula counts [M, 10]) for the M molecules in the batch."""
        h = self.spectra(b)
        seg, M = b['seg'], b['mol_fp'].shape[0]
        a = self.pool(h).squeeze(-1).float()
        amax = torch.full((M,), -1e4, device=h.device).scatter_reduce(0, seg, a, 'amax')
        e = (a - amax[seg]).exp()
        w = e / torch.zeros(M, device=h.device).index_add(0, seg, e)[seg]
        mol = torch.zeros(M, h.shape[1], device=h.device, dtype=h.dtype).index_add(0, seg, h * w[:, None].to(h.dtype))
        return self.fp_head(mol), self.formula_head(mol)


def unpack_fp(fp):
    """uint8 [M, 256] -> float [M, 2048] bits (np.packbits order)."""
    bits = torch.arange(7, -1, -1, device=fp.device)
    return ((fp[..., None].int() >> bits) & 1).flatten(1).float()


def loss_fn(fp_logits, formula_pred, b, gamma=2.0, formula_w=0.1):
    y = unpack_fp(b['mol_fp'])
    bce = Fn.binary_cross_entropy_with_logits(fp_logits.float(), y, reduction='none')
    pt = torch.exp(-bce)
    focal = ((1 - pt) ** gamma * bce).sum(1).mean() / 10  # ponytail: /10 keeps focal ~O(1); tune with lr if needed
    f = b['mol_formula'].float()
    ok = (f >= 0).all(1)
    form = Fn.smooth_l1_loss(formula_pred.float()[ok], torch.log1p(f[ok])) if ok.any() else fp_logits.sum() * 0
    return focal + formula_w * form, {'focal': focal.detach(), 'formula': form.detach()}


def tanimoto(pred_bits, fp_bits):
    """[M, 2048] 0/1 float tensors -> [M] Tanimoto of matching rows."""
    inter = (pred_bits * fp_bits).sum(1)
    return inter / (pred_bits.sum(1) + fp_bits.sum(1) - inter).clamp(min=1)


def feature_stats(root, n=50_000):
    """Evenly spaced train spectra -> (peakfeat [P, F], sqrt inten [P]) over valid peaks, for Encoder.set_stats."""
    d = f'{root}/train'
    npk = np.load(f'{d}/npeaks.npy', mmap_mode='r')
    idx = np.linspace(0, len(npk) - 1, n).astype(np.int64)
    pf, it = np.load(f'{d}/peakfeat.npy', mmap_mode='r')[idx], np.load(f'{d}/inten.npy', mmap_mode='r')[idx]
    valid = np.arange(pf.shape[1])[None] < npk[idx][:, None]
    return torch.from_numpy(pf[valid].astype(np.float32)), torch.from_numpy(np.sqrt(it[valid].astype(np.float32)))
