"""S03 model tests (CPU; each is small). The brief's checklist:

patchify ≡ unfold+Linear; sincos regenerates for a new grid; manual ≡ sdpa ≡ nn.MultiheadAttention; masked keys get
zero weight; DropPath identity in eval; residual shapes; SliceViT forward; heads zero-init → loss ln 2; padding does not
change the output; overfit 8 studies; bf16 forward/backward; MAE shapes/loss.
"""
import math

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from knee.model.attention import CrossAttention, MultiHeadSelfAttention, attend  # noqa: E402
from knee.model.block import Block, CrossBlock, DropPath  # noqa: E402
from knee.model.heads import AttnMILHead, ClsStudyHead, FindingQueryHead  # noqa: E402
from knee.model.knee_vit import KneeViT, drop_windows  # noqa: E402
from knee.model.patch_embed import ConvStem, PatchEmbed  # noqa: E402
from knee.model.pos_embed import PosEmbed2D, SeriesTypeEmbed, SlicePosEmbed, sincos_2d  # noqa: E402
from knee.model.slice_vit import SliceViT  # noqa: E402

TINY = {"slice_vit": {"img_size": 32, "patch": 8, "dim": 32, "depth": 2, "heads": 2, "drop_path": 0.0},
        "head": {"kind": "attn_mil", "att_dim": 16, "dropout": 0.0}}


def fake_bag(B=2, L=6, px=32, n_real=(4, 6), seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, n in enumerate(n_real):
        mask[i, :n] = True
    b = {"x": torch.randint(0, 256, (B, L, 3, px, px), generator=g, dtype=torch.uint8), "window_mask": mask,
         "z": torch.rand(B, L, generator=g), "z_mm": torch.randn(B, L, generator=g) * 30}
    for k, n in (("plane", 3), ("fluid", 2), ("contrast", 6), ("field", 3), ("slot", 6), ("series_idx", 4)):
        b[k] = torch.randint(0, n, (B, L), generator=g)
    return b


# ------------------------------------------------------------------------------------------ tokenizers
def test_patchify_equals_unfold_linear():
    torch.manual_seed(0)
    pe = PatchEmbed(in_ch=3, embed_dim=24, patch=8)
    x = torch.randn(2, 3, 32, 40)
    lin = nn.Linear(3 * 8 * 8, 24)
    lin.weight.data = pe.proj.weight.data.reshape(24, -1)  # conv kernel [E, C, P, P] flattened = the embedding matrix
    lin.bias.data = pe.proj.bias.data
    patches = F.unfold(x, kernel_size=8, stride=8).transpose(1, 2)  # [N, n, C·P·P], row-major patch order
    torch.testing.assert_close(pe(x), lin(patches), rtol=1e-5, atol=1e-5)
    assert pe(x).shape == (2, 4 * 5, 24)


@pytest.mark.parametrize("norm", ["ln", "gn", "bn"])
def test_conv_stem_grid_matches_patchify(norm):
    stem = ConvStem(3, 64, 16, norm)
    assert stem(torch.randn(2, 3, 64, 96)).shape == (2, 4 * 6, 64)
    assert stem.grid(64, 96) == PatchEmbed(3, 64, 16).grid(64, 96)
    with pytest.raises(ValueError):
        ConvStem(3, 64, 12)


# ------------------------------------------------------------------------------------------ position / type
def test_sincos_2d_regenerates_and_is_distinct():
    a, b = sincos_2d(4, 4, 32), sincos_2d(6, 5, 32)
    assert a.shape == (16, 32) and b.shape == (30, 32)
    assert torch.cdist(a, a).fill_diagonal_(1).min() > 1e-3  # every position gets a different code
    pe = PosEmbed2D(32, "sincos", (4, 4))
    assert pe(torch.zeros(1, 30, 32), (6, 5)).shape == (1, 30, 32)
    assert not list(pe.parameters())


def test_learned_pos_resizes():
    pe = PosEmbed2D(16, "learned", (4, 4))
    assert pe(torch.zeros(2, 16, 16), (4, 4)).shape == (2, 16, 16)
    assert pe(torch.zeros(2, 64, 16), (8, 8)).shape == (2, 64, 16)


def test_slice_pos_zero_init_and_series_type():
    sp = SlicePosEmbed(16, use_mm=True)
    t = torch.randn(2, 5, 16)
    torch.testing.assert_close(sp(t, torch.rand(2, 5), torch.randn(2, 5)), t)  # neutral at init
    st = SeriesTypeEmbed(16, ("plane", "fluid"), max_series=2)
    out = st(t, {"plane": torch.zeros(2, 5, dtype=torch.long), "fluid": torch.ones(2, 5, dtype=torch.long),
                 "series_idx": torch.full((2, 5), 7)})  # series_idx beyond the table is clamped
    assert out.shape == t.shape and not torch.equal(out, t)


# ------------------------------------------------------------------------------------------ attention
def test_manual_equals_sdpa_equals_torch_mha():
    torch.manual_seed(0)
    E, h = 32, 4
    x = torch.randn(3, 7, E)
    key_mask = torch.ones(3, 7, dtype=torch.bool)
    key_mask[0, 5:] = False
    key_mask[2, 1:] = False
    ours = MultiHeadSelfAttention(E, h, impl="manual").eval()
    ref = nn.MultiheadAttention(E, h, batch_first=True).eval()
    ref.in_proj_weight.data.copy_(ours.qkv.weight.data)  # both pack [Wq; Wk; Wv]
    ref.in_proj_bias.data.copy_(ours.qkv.bias.data)
    ref.out_proj.weight.data.copy_(ours.proj.weight.data)
    ref.out_proj.bias.data.copy_(ours.proj.bias.data)
    y_manual = ours(x, key_mask)
    ours.impl = "sdpa"
    y_sdpa = ours(x, key_mask)
    y_ref, _ = ref(x, x, x, key_padding_mask=~key_mask, need_weights=False)  # torch: True = ignore
    torch.testing.assert_close(y_manual, y_sdpa, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y_manual, y_ref, rtol=1e-5, atol=1e-5)


def test_masked_keys_get_zero_weight_and_no_nan():
    q = k = v = torch.randn(2, 2, 5, 8)
    m = torch.tensor([[True, True, False, False, False], [False] * 5])  # sample 1: no real key at all
    out, attn = attend(q, k, v, m, impl="manual", return_attn=True)
    assert attn[0, :, :, 2:].abs().max() == 0
    torch.testing.assert_close(attn.sum(-1), torch.ones(2, 2, 5))
    assert torch.isfinite(out).all()
    assert torch.isfinite(attend(q, k, v, m, impl="sdpa")).all()


def test_cross_attention_manual_equals_sdpa():
    ca = CrossAttention(16, 2, qk_norm=True, impl="manual").eval()
    q, ctx = torch.randn(2, 12, 16), torch.randn(2, 9, 16)
    m = torch.rand(2, 9) > 0.3
    m[:, 0] = True
    a = ca(q, ctx, m)
    ca.impl = "sdpa"
    torch.testing.assert_close(a, ca(q, ctx, m), rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------------------------------ blocks
def test_droppath_identity_in_eval_and_scales_in_train():
    dp = DropPath(0.5)
    x = torch.ones(1000, 4)
    assert torch.equal(dp.eval()(x), x)
    y = dp.train()(x)
    assert set(y.unique().tolist()) <= {0.0, 2.0} and abs(y.mean().item() - 1.0) < 0.15


def test_block_shapes_and_grads_reach_all_params():
    blk = Block(32, 4, qk_norm=True, drop_path=0.1)
    x = torch.randn(2, 10, 32, requires_grad=True)
    y = blk(x, torch.ones(2, 10, dtype=torch.bool))
    assert y.shape == x.shape
    y.square().sum().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in blk.parameters())
    cb = CrossBlock(32, 4)
    assert cb(torch.randn(2, 12, 32), torch.randn(2, 10, 32)).shape == (2, 12, 32)


# ------------------------------------------------------------------------------------------ SliceViT
@pytest.mark.parametrize("pool,tok", [("gap", "conv_stem"), ("cls", "patchify")])
def test_slice_vit_forward(pool, tok):
    m = SliceViT(img_size=32, patch=8, dim=32, depth=2, heads=2, pool=pool, tokenizer=tok)
    assert m(torch.randn(5, 3, 32, 32)).shape == (5, 32)
    assert m(torch.randn(2, 3, 48, 48)).shape == (2, 32)  # another resolution: sincos regenerates
    assert m.layer_id("blocks.1.attn.qkv.weight") == 2 and m.layer_id("norm.weight") == 3
    assert m.layer_id("tokenizer.proj.0.weight") == 0


def test_grad_checkpointing_same_output():
    torch.manual_seed(0)
    m = SliceViT(img_size=32, patch=8, dim=32, depth=2, heads=2, drop_path=0.0).train()
    x = torch.randn(3, 3, 32, 32, requires_grad=True)
    a = m(x)
    m.grad_checkpointing = True
    torch.testing.assert_close(a, m(x))


# ------------------------------------------------------------------------------------------ heads + KneeViT
@pytest.mark.parametrize("head", [AttnMILHead(32, gated=True), FindingQueryHead(32, heads=2), ClsStudyHead(32, heads=2)])
def test_heads_zero_init_and_mask(head):
    head.eval()
    t = torch.randn(2, 7, 32)
    m = torch.ones(2, 7, dtype=torch.bool)
    m[0, 4:] = False
    out = head(t, m)
    assert out.shape == (2, 12) and out.abs().max() == 0  # zero-init logits → p = 0.5
    with torch.no_grad():  # give it non-trivial weights, then check padding invariance
        for p in head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    t2 = t.clone()
    t2[0, 4:] = torch.randn(3, 32) * 100
    torch.testing.assert_close(head(t, m), head(t2, m), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("head", ["attn_mil", "finding_query", "cls"])
def test_knee_vit_init_loss_is_ln2(head):
    cfg = {**TINY, "head": {"kind": head} if head != "attn_mil" else TINY["head"]}
    if head != "attn_mil":
        cfg["head"]["heads"] = 2
    m = KneeViT(cfg).eval()
    logits = m(fake_bag())
    y = torch.randint(0, 2, logits.shape).float()
    assert math.isclose(F.binary_cross_entropy_with_logits(logits, y).item(), math.log(2), rel_tol=1e-6)


def test_padding_does_not_change_output():
    torch.manual_seed(0)
    m = KneeViT({**TINY, "series_type": {"fields": ["plane", "fluid", "contrast", "field", "slot"],
                                         "max_series": 4}}).eval()
    with torch.no_grad():
        for p in m.head.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    b = fake_bag(B=2, L=6, n_real=(4, 6))
    b2 = {k: v.clone() for k, v in b.items()}
    b2["x"][0, 4:] = 255 - b2["x"][0, 4:]  # garbage in the padded slots
    b2["z"][0, 4:] = 0.123
    a1, a2 = m(b), m(b2)
    torch.testing.assert_close(a1, a2)
    # a bag with the padding physically removed gives the same answer for that study
    b3 = {k: v[:1, :4] for k, v in b.items()}
    torch.testing.assert_close(m(b3), a1[:1], rtol=1e-5, atol=1e-5)
    # window_chunk is a memory knob only
    m.window_chunk = 1
    torch.testing.assert_close(m(b), a1, rtol=1e-5, atol=1e-5)


def test_drop_windows_keeps_one():
    mask = torch.ones(50, 4, dtype=torch.bool)
    mask[:, 2:] = False
    out = drop_windows(mask, 0.99)
    assert (out.sum(1) >= 1).all() and not (out & ~mask).any()


def test_overfit_8_studies():
    """fp32, tiny model: the loss must go to ~0 on 8 fixed studies (the pipeline can learn at all)."""
    torch.manual_seed(0)
    m = KneeViT({**TINY, "head": {"kind": "finding_query", "heads": 2, "ctx_depth": 1, "query_depth": 1}})
    b = fake_bag(B=8, L=4, n_real=(4, 3, 4, 2, 4, 4, 1, 3))
    y = torch.randint(0, 2, (8, 12), generator=torch.Generator().manual_seed(1)).float()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    m.train()
    for _ in range(150):
        loss = F.binary_cross_entropy_with_logits(m(b), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()


def test_bf16_forward_backward():
    m = KneeViT(TINY).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = m(fake_bag())
        loss = F.binary_cross_entropy_with_logits(logits.float(), torch.rand_like(logits.float()))
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_return_attn_shapes():
    m = KneeViT({**TINY, "head": {"kind": "finding_query", "heads": 2}}).eval()
    logits, attn = m(fake_bag(L=6), return_attn=True)
    assert logits.shape == (2, 12) and attn.shape == (2, 12, 6)
    m2 = KneeViT(TINY).eval()
    _, attn2 = m2(fake_bag(L=6), return_attn=True)
    torch.testing.assert_close(attn2.sum(-1), torch.ones(2, 12))


def test_attn_mil_empty_bag_is_finite():
    """An all-padding bag gave softmax(all -inf) = NaN logits and NaN grads (review 2026-09-30). Now it gets
    the bias-only logits, and the other bag in the batch is unchanged."""
    torch.manual_seed(0)
    head = AttnMILHead(8, att_dim=16, dropout=0.0)
    nn.init.normal_(head.cls_w)
    tokens = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [False] * 5])
    out = head(tokens, mask)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out[1], head.cls_b.detach())
    torch.testing.assert_close(out[:1], head(tokens[:1], mask[:1]))
    out.sum().backward()
    assert torch.isfinite(tokens.grad).all()


@pytest.mark.parametrize("norm,same", [("ln", True), ("gn", False)])
def test_conv_stem_norm_is_local(norm, same):
    """MAE zeroes the masked pixels before the stem. With norm="ln" a token outside the masked region's receptive
    field (31 px for P=16) is identical whether or not the far region is zeroed; GroupNorm's per-image statistics
    leak the masking into every token, a pretrain/fine-tune mismatch (review 2026-09-30)."""
    torch.manual_seed(0)
    stem = ConvStem(3, 64, 16, norm).eval()
    x = torch.randn(1, 3, 64, 64)
    x_masked = x.clone()
    x_masked[..., 48:] = 0  # zero the right-most patch column; token column 0 sees pixels < 32
    a, b = stem(x).reshape(1, 4, 4, 64), stem(x_masked).reshape(1, 4, 4, 64)
    assert torch.allclose(a[:, :, 0], b[:, :, 0], atol=1e-5) == same
