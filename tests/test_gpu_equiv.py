"""GPU equivalence of Track A's memory/speed switches (2026-10-07, E003): partial gradient checkpointing and
channels_last must not change the maths. Skipped without CUDA. TF32 is switched off: with it on (PyTorch's default
for convolutions) channels_last picks different tensor-core kernels and the outputs differ by ~10% (E003), which is
kernel-choice noise, not a bug. The T4 has no TF32. The nano backbone keeps this fast; E003 ran the real
coatnet_rmlp_2_rw_384 on real cached slices with the same result."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

from knee.model.trackA import TrackAModel  # noqa: E402

CFG = {"backbone": "coatnet_nano_rw_224", "pretrained": False, "window_chunk": 4}


@pytest.fixture(autouse=True)
def _no_tf32():
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def _run(sd, ck, cl, train):
    m = TrackAModel({**CFG, "grad_checkpointing": ck}).cuda()
    m.load_state_dict(sd)
    if cl:
        m.use_channels_last()
    m.train(train)
    torch.manual_seed(1234)  # drop-path draws are replayed by the checkpoint recompute
    torch.cuda.manual_seed(1234)
    x = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8, device="cuda")
    f = m._encode_chunk(x)
    f.square().mean().backward()
    g = torch.cat([p.grad.flatten() for p in m.parameters() if p.grad is not None])
    return f.detach(), g


@pytest.mark.parametrize("train", [True, False])
def test_stage_checkpointing_and_channels_last_exact_on_gpu(train):
    torch.manual_seed(0)
    sd = {k: v.clone() for k, v in TrackAModel({**CFG, "grad_checkpointing": True}).state_dict().items()}
    f0, g0 = _run(sd, True, False, train)
    f1, g1 = _run(sd, [0, 1], False, train)  # partial checkpointing: bit-identical features
    assert torch.equal(f1, f0)
    assert (g1 - g0).norm() / g0.norm() < 1e-5
    for ck in (True, [0, 1]):  # channels_last: same maths, different summation order (E003: feat 4e-5, grad 3e-4)
        f2, g2 = _run(sd, ck, True, train)
        torch.testing.assert_close(f2, f0, rtol=1e-3, atol=1e-3)
        assert (g2 - g0).norm() / g0.norm() < 5e-3
