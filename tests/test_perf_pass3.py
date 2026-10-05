"""Performance pass 3 (2026-10-01): fused residual branch, channels_last SliceViT stem, Track A per-stage gradient
checkpointing, the compile cache helpers. Each change is meant to be exact; these tests hold it to that."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from knee.model.block import Block, CrossBlock, DropPath, LayerScale, add_branch  # noqa: E402
from knee.model.slice_vit import SliceViT  # noqa: E402


def _reference_branch(x, branch, ls: LayerScale, dp: DropPath):
    """The pre-pass-3 formula: x + DropPath(LayerScale(branch)) with DropPath's old x * mask / keep."""
    y = ls(branch)
    if dp.p > 0 and dp.training:
        keep = 1.0 - dp.p
        mask = y.new_empty((y.shape[0],) + (1,) * (y.ndim - 1)).bernoulli_(keep)
        y = y * mask / keep
    return x + y


@pytest.mark.parametrize("gamma", [0.1, None])
@pytest.mark.parametrize("p,training", [(0.0, True), (0.3, True), (0.3, False)])
def test_add_branch_matches_reference(gamma, p, training):
    ls, dp = LayerScale(8, gamma), DropPath(p)
    dp.train(training)
    x = torch.randn(16, 5, 8, requires_grad=True)
    b = torch.randn(16, 5, 8, requires_grad=True)
    torch.manual_seed(1)
    out = add_branch(x, b, ls, dp)
    torch.manual_seed(1)
    ref = _reference_branch(x, b, ls, dp)
    torch.testing.assert_close(out, ref)
    g = torch.randn_like(out)
    grads = torch.autograd.grad(out, [x, b, *ls.parameters()], g)
    rgrads = torch.autograd.grad(ref, [x, b, *ls.parameters()], g)
    for a, r in zip(grads, rgrads):
        torch.testing.assert_close(a, r)


def test_add_branch_keeps_dtype_promotion():
    """Under autocast the fp32 γ promoted the branch, so the residual stream was fp32 in training; unchanged."""
    ls, dp = LayerScale(8, 0.1), DropPath(0.2)
    x = torch.randn(4, 3, 8, dtype=torch.bfloat16)
    b = torch.randn(4, 3, 8, dtype=torch.bfloat16)
    assert add_branch(x, b, ls, dp).dtype == _reference_branch(x, b, ls, dp).dtype == torch.float32
    assert add_branch(x, b, LayerScale(8, None), DropPath(0.0)).dtype == torch.bfloat16



def test_add_branch_bf16_mask_matches_reference():
    """Under bf16 autocast the old path drew the DropPath mask in γ's fp32; a bf16 mask stores 1/keep as 1.1094
    instead of 1.1111 at p=0.1 (review 2026-10-01). Values, not just the dtype."""
    ls, dp = LayerScale(8, 0.1), DropPath(0.1)
    x = torch.randn(64, 3, 8, dtype=torch.bfloat16)
    b = torch.randn(64, 3, 8, dtype=torch.bfloat16)
    torch.manual_seed(2)
    out = add_branch(x, b, ls, dp)
    torch.manual_seed(2)
    ref = _reference_branch(x, b, ls, dp)
    torch.testing.assert_close(out, ref, rtol=1e-6, atol=1e-6)
    assert dp.scale(b, torch.float32).dtype == torch.float32 and dp.scale(b).dtype == torch.bfloat16

@pytest.mark.parametrize("cls", [Block, CrossBlock])
def test_blocks_use_fused_branch_same_output(cls):
    torch.manual_seed(0)
    blk = cls(32, 4, drop_path=0.25, layerscale=0.1).train()
    x, ctx = torch.randn(6, 7, 32), torch.randn(6, 9, 32)
    args = (x,) if cls is Block else (x, ctx)
    torch.manual_seed(3)
    out = blk(*args)
    torch.manual_seed(3)
    if cls is Block:
        h = _reference_branch(x, blk.attn(blk.norm1(x)), blk.ls1, blk.drop_path)
        ref = _reference_branch(h, blk.mlp(blk.norm2(h)), blk.ls2, blk.drop_path)
    else:
        h = _reference_branch(x, blk.attn(blk.norm_q(x), blk.norm_ctx(ctx)), blk.ls1, blk.drop_path)
        ref = _reference_branch(h, blk.mlp(blk.norm2(h)), blk.ls2, blk.drop_path)
    torch.testing.assert_close(out, ref)


def test_slice_vit_channels_last_same_output_and_grads():
    torch.manual_seed(0)
    a = SliceViT(img_size=64, patch=16, dim=64, depth=2, heads=4, drop_path=0.0)
    b = SliceViT(img_size=64, patch=16, dim=64, depth=2, heads=4, drop_path=0.0)
    b.load_state_dict(a.state_dict())
    b.use_channels_last()
    x = torch.randn(3, 3, 64, 64)
    oa, ob = a(x), b(x)
    torch.testing.assert_close(ob, oa, rtol=1e-5, atol=1e-5)
    oa.square().sum().backward()
    ob.square().sum().backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(pb.grad, pa.grad, rtol=1e-4, atol=1e-5, msg=n)
    assert set(a.state_dict()) == set(b.state_dict())  # checkpoints stay interchangeable


def test_mae_channels_last_same_loss():
    from knee.ssl.mae import MAE
    torch.manual_seed(0)
    enc = SliceViT(img_size=64, patch=16, dim=64, depth=2, heads=4, drop_path=0.0)
    m1 = MAE(enc, dec_dim=32, dec_depth=1, dec_heads=4)
    m2 = MAE(SliceViT(img_size=64, patch=16, dim=64, depth=2, heads=4, drop_path=0.0), dec_dim=32, dec_depth=1,
             dec_heads=4)
    m2.load_state_dict(m1.state_dict())
    m2.encoder.use_channels_last()
    x = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8)
    l1 = m1(x, torch.Generator().manual_seed(5))["loss"]
    l2 = m2(x, torch.Generator().manual_seed(5))["loss"]
    torch.testing.assert_close(l2, l1, rtol=1e-5, atol=1e-6)


def test_trackA_stage_checkpointing():
    pytest.importorskip("timm")
    from knee.model.trackA import TrackAModel
    m = TrackAModel({"backbone": "coatnet_nano_rw_224", "pretrained": False, "grad_checkpointing": [0, 1],
                     "window_chunk": 4})
    flags = [s.grad_checkpointing for s in m.backbone.stages]
    assert flags[:2] == [True, True] and not any(flags[2:])
    m.set_grad_checkpointing(False)
    assert not any(s.grad_checkpointing for s in m.backbone.stages)
    m.set_grad_checkpointing(True)
    assert all(s.grad_checkpointing for s in m.backbone.stages)
    with pytest.raises(ValueError):
        m.set_grad_checkpointing([7])


def test_trackA_stage_checkpointing_same_grads():
    """Checkpointing only recomputes: with frozen BN (the default) the loss and gradients are unchanged."""
    pytest.importorskip("timm")
    from knee.model.trackA import TrackAModel
    torch.manual_seed(0)
    cfg = {"backbone": "coatnet_nano_rw_224", "pretrained": False, "window_chunk": 4}
    a = TrackAModel({**cfg, "grad_checkpointing": False}).train()
    b = TrackAModel({**cfg, "grad_checkpointing": [0, 1]}).train()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 224, 224)  # the nano model's fixed size
    a.backbone(x).square().sum().backward()
    b.backbone(x).square().sum().backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if pa.grad is not None:
            torch.testing.assert_close(pb.grad, pa.grad, rtol=1e-4, atol=1e-6, msg=n)



def test_trackA_checkpointed_stages_recompute():
    """The gradient test above passes with or without checkpointing; this one checks that exactly the listed stages
    run their blocks a second time in backward."""
    pytest.importorskip("timm")
    from knee.model.trackA import TrackAModel
    m = TrackAModel({"backbone": "coatnet_nano_rw_224", "pretrained": False, "grad_checkpointing": [0, 1],
                     "window_chunk": 4}).train()
    calls = [0] * len(m.backbone.stages)
    for i, st in enumerate(m.backbone.stages):  # wrap forward: module hooks don't fire on the checkpoint recompute
        blk, fwd = st.blocks[0], st.blocks[0].forward

        def counted(*a, fwd=fwd, i=i, **k):
            calls[i] += 1
            return fwd(*a, **k)
        blk.forward = counted
    m.backbone(torch.randn(1, 3, 224, 224)).sum().backward()
    assert calls == [2, 2] + [1] * (len(calls) - 2)


def test_trackA_stage_checkpointing_config_backbone():
    """Stage indices on the backbone the config actually uses (4 stages), not only the nano model."""
    pytest.importorskip("timm")
    from knee.model.trackA import TrackAModel
    m = TrackAModel({"backbone": "coatnet_rmlp_2_rw_384", "pretrained": False, "grad_checkpointing": [0, 1],
                     "window_chunk": 4})
    assert [s.grad_checkpointing for s in m.backbone.stages] == [True, True, False, False]

def test_compile_cache_helpers_are_noops_without_env(monkeypatch, tmp_path):
    from knee.model import compiled
    monkeypatch.delenv(compiled.COMPILE_CACHE_IN, raising=False)
    monkeypatch.delenv(compiled.COMPILE_CACHE_OUT, raising=False)
    assert compiled.load_compile_cache() is False
    assert compiled.save_compile_cache("cuda:0") is None
    # nothing compiled (torch returns None) → nothing written, even with an output dir
    monkeypatch.setenv(compiled.COMPILE_CACHE_OUT, str(tmp_path))
    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", lambda: None)
    assert compiled.save_compile_cache("cuda:0") is None
    assert list(tmp_path.iterdir()) == []
    # a corrupt cache file is reported, never raised
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not a cache")
    monkeypatch.setenv(compiled.COMPILE_CACHE_IN, str(bad))
    monkeypatch.setattr(compiled, "_cache_loaded", False)
    assert compiled.load_compile_cache() is False


# ---- model review 2026-10-01 (second pass)

def test_droppath_scale_fp32_without_layerscale():
    """With LayerScale off, a bf16 keep factor rounds 1/0.9 down to 1.1094; the factor is now drawn in fp32."""
    dp = DropPath(0.1).train()
    torch.manual_seed(0)
    y = add_branch(torch.zeros(4096, 1, 2, dtype=torch.bfloat16), torch.ones(4096, 1, 2, dtype=torch.bfloat16),
                   LayerScale(2, None), dp)
    assert y.dtype == torch.float32
    kept = y[y != 0]
    assert kept.numel() and torch.allclose(kept, torch.tensor(1 / 0.9))


def test_sincos_pos2d_compiles_once_per_shape():
    """The dict cache in PosEmbed2D made Dynamo compile the first shape twice (cache empty, then filled)."""
    from torch._dynamo.utils import counters
    m = SliceViT(dim=32, depth=1, heads=2, img_size=32).eval()
    assert not any("sincos" in k for k in m.state_dict())  # non-persistent: old checkpoints load unchanged
    m.compile(backend="eager", dynamic=False)
    counters.clear()
    with torch.no_grad():
        for n in (2, 2, 4, 4):
            m(torch.randn(n, 3, 32, 32))
    assert counters["stats"]["unique_graphs"] == 2


def test_sincos_pos2d_other_grid_matches_base():
    from knee.model.pos_embed import PosEmbed2D, sincos_2d
    pe = PosEmbed2D(32, "sincos", (4, 4))
    assert torch.equal(pe.table_for((4, 4), "cpu"), sincos_2d(4, 4, 32)[None])
    assert torch.equal(pe.table_for((2, 3), "cpu"), sincos_2d(2, 3, 32)[None])


def test_index_gather_scatter_matches_bool_mask():
    """KneeViT.forward gathers/scatters real windows by index (one sync) instead of bool masks (two); same output and
    gradients as the old x[mask] / tokens[mask] = feats."""
    from knee.model.knee_vit import KneeViT
    torch.manual_seed(0)
    m = KneeViT({"slice_vit": {"dim": 32, "depth": 1, "heads": 2, "img_size": 32}}).eval()  # eval: no dropout
    for p in m.parameters():  # non-zero head weights so gradients reach the encoder
        if p.ndim and not p.abs().sum():
            torch.nn.init.normal_(p, std=0.1)
    B, L = 2, 5
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    types_ = {k: torch.zeros(B, L, dtype=torch.long) for k in ("plane", "fluid", "contrast", "field")}
    batch = {"x": torch.randint(0, 256, (B, L, 3, 32, 32), dtype=torch.uint8), "window_mask": mask,
             "z": torch.rand(B, L), **types_}
    out = m(batch)
    out.sum().backward()
    grads = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    feats = m.encode_windows(batch["x"][mask])
    tok = feats.new_zeros((B, L, feats.shape[-1]))
    tok[mask] = feats
    ref = m.head(m.series_type(m.slice_pos(tok, batch["z"]), types_), mask)
    ref.sum().backward()
    assert torch.equal(out, ref)
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, m.parameters()))
