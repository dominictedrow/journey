"""Performance step 2 (docs/plan/03_performance_plan.md): fixed-shape chunk buckets for torch.compile, and pure-half
inference. Bucketing must not change the outputs (SliceViT: bit-for-bit up to float rounding, train and eval; Track A:
exact in eval); fp16 must stay within fp16 rounding of fp32.
"""
import pytest

torch = pytest.importorskip("torch")

from knee.model.compiled import bucket_for, default_buckets, encode_chunks, is_half, plan_chunks, to_half  # noqa: E402
from knee.model.knee_vit import KneeViT  # noqa: E402

TINY = {"slice_vit": {"img_size": 32, "patch": 8, "dim": 32, "depth": 2, "heads": 2, "drop_path": 0.0},
        "head": {"kind": "attn_mil", "att_dim": 16, "dropout": 0.0}, "window_chunk": 16}


def _bag(L: int, px: int = 32, n_real: int | None = None, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(1, L, dtype=torch.bool)
    mask[0, : (L if n_real is None else n_real)] = True
    zl = torch.zeros(1, L, dtype=torch.long)
    return {"x": torch.randint(0, 255, (1, L, 3, px, px), dtype=torch.uint8, generator=g), "window_mask": mask,
            "z": torch.rand(1, L, generator=g), "plane": zl, "fluid": zl, "contrast": zl, "field": zl,
            "slot": (torch.arange(L) % 5)[None]}


def test_buckets():
    assert default_buckets(16) == [4, 8, 16]
    assert default_buckets(64) == [4, 8, 16, 32, 64]
    assert default_buckets(6) == [4, 6]
    assert default_buckets(24, min_bucket=8) == [8, 16, 24]
    assert bucket_for(1, [4, 8, 16]) == 4 and bucket_for(9, [4, 8, 16]) == 16
    with pytest.raises(ValueError):
        bucket_for(17, [4, 8, 16])
    assert plan_chunks(44, [4, 8, 16, 32, 64]) == [(32, 32), (8, 8), (4, 4)]
    assert plan_chunks(15, [4, 8, 16]) == [(8, 8), (4, 4), (3, 4)]
    assert plan_chunks(60, [4, 8, 16]) == [(16, 16)] * 3 + [(8, 8), (4, 4)]
    assert plan_chunks(0, [4, 8]) == []
    for n in range(1, 200):  # every window covered once, padding < the smallest bucket
        plan = plan_chunks(n, [4, 8, 16, 32, 64])
        assert sum(r for r, _ in plan) == n and sum(s - r for r, s in plan) < 4


@pytest.mark.parametrize("n", [1, 5, 16, 23, 44, 61])
def test_encoder_sees_only_bucket_shapes(n):
    seen = []
    x = torch.randn(n, 3)
    out = encode_chunks(lambda c: (seen.append(c.shape[0]), c * 2)[1], x, 16, [4, 8, 16])
    assert set(seen) <= {4, 8, 16}
    torch.testing.assert_close(torch.cat(out), x * 2)


@pytest.mark.parametrize("train", [False, True])
def test_kneevit_bucketed_equals_plain(train):
    torch.manual_seed(0)
    m = KneeViT(TINY)
    m.train(train)
    b = _bag(23)  # 16 + 7 → the ragged chunk pads 7 → 8
    ref = m(b)
    g_ref = torch.autograd.grad(ref.sum(), [p for p in m.parameters() if p.requires_grad], allow_unused=True)
    m.chunk_buckets = default_buckets(m.window_chunk)
    out = m(b)
    g_out = torch.autograd.grad(out.sum(), [p for p in m.parameters() if p.requires_grad], allow_unused=True)
    torch.testing.assert_close(out, ref)
    for a, r in zip(g_out, g_ref):
        if r is not None:
            torch.testing.assert_close(a, r)


def test_trackA_bucketed_equals_plain_in_eval():
    pytest.importorskip("timm")
    from knee.model.trackA import TrackAModel
    torch.manual_seed(0)
    m = TrackAModel({"backbone": "resnet10t", "pretrained": False, "grad_checkpointing": False, "window_chunk": 4,
                     "head": {"kind": "attn_mil", "att_dim": 16, "dropout": 0.0}}).eval()  # resnet: has BatchNorm
    b = _bag(7, px=64, n_real=6)
    with torch.no_grad():
        ref = m(b)
        m.chunk_buckets = [2, 4]
        torch.testing.assert_close(m(b), ref)
        m.use_channels_last()
        torch.testing.assert_close(m(b), ref, rtol=1e-4, atol=1e-5)


def _cpu_half_ok() -> bool:
    try:
        torch.nn.functional.layer_norm(torch.randn(2, 8).half(), (8,))
        torch.nn.functional.conv2d(torch.randn(1, 3, 8, 8).half(), torch.randn(4, 3, 3, 3).half())
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(not _cpu_half_ok(), reason="this torch build has no fp16 CPU kernels")
@pytest.mark.parametrize("head", ["attn_mil", "finding_query"])
def test_half_model_runs_and_matches_fp32(head):
    """Every dtype hand-off works in pure fp16 (no autocast): a float32 tensor meeting an fp16 Linear would raise."""
    from knee.model.fuse import prepare_for_inference
    torch.manual_seed(0)
    cfg = {**TINY, "slice_pos": {"n_freq": 8, "use_mm": True},
           "head": {"kind": "attn_mil", "att_dim": 16, "dropout": 0.0} if head == "attn_mil"
           else {"kind": "finding_query", "heads": 2}}
    m = KneeViT(cfg)
    torch.nn.init.normal_(m.slice_pos.proj.weight, std=0.1)  # zero-init by default: make the position path matter
    for p in m.head.parameters():  # zero-init logit layers too
        if p.abs().sum() == 0:
            torch.nn.init.normal_(p, std=0.1)
    prepare_for_inference(m)
    b = {**_bag(21, n_real=19), "z_mm": torch.randn(1, 21) * 30}
    with torch.no_grad():
        ref = m(b)
        to_half(m)
        assert is_half(m)
        out = m(b)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_slice_pos_frequencies_stay_fp32_after_half():
    from knee.model.pos_embed import SlicePosEmbed
    e = SlicePosEmbed(8, n_freq=8)
    z = torch.tensor([0.123456])
    f32 = e._feat(z)
    e.half()
    torch.testing.assert_close(e._feat(z), f32)  # not rebuilt from fp16-rounded frequencies


def test_compile_encoder_same_output_on_cpu():
    """CPU inductor smoke test: in-place compile keeps state_dict keys and the outputs, and every bag size maps onto
    the few bucket shapes."""
    import shutil
    if shutil.which("cc") is None and shutil.which("clang") is None and shutil.which("gcc") is None:
        pytest.skip("no C compiler for inductor")
    torch.manual_seed(0)
    m = KneeViT(TINY).eval()
    keys = list(m.state_dict())
    bags = [_bag(n, seed=n) for n in (3, 11, 20)]
    with torch.no_grad():
        refs = [m(b) for b in bags]
        m.compile_encoder()
        assert list(m.state_dict()) == keys
        for b, r in zip(bags, refs):
            torch.testing.assert_close(m(b), r, rtol=1e-4, atol=1e-5)
