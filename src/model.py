"""Stacked bidirectional LSTM with attention pooling, for isolated sign recognition.

Uses nn.LSTM (cuDNN's fused implementation) rather than a hand-written cell:
same gate math (sigmoid forget/input/output gates, tanh candidate, cell-state
recurrence), but cuDNN packs all 4 gates' weights into one matrix and runs the
per-timestep loop inside its own optimized CUDA kernel instead of round-tripping
through Python once per timestep -- far better GPU utilization for the same result.
Three of these are stacked explicitly (not via nn.LSTM's own num_layers) so a
LayerNorm + residual + ReLU can sit between them.

Shape of the block, and why each piece is what it is:

  input projection   Linear(694 -> 384) -> LayerNorm -> ReLU, applied once.
                     The previous version instead re-added a raw-input
                     projection before every activation. That looks like a residual
                     but it is a shortcut: mean_t(ReLU(W.x_t + LSTM_out)) is,
                     at the readout, an order-invariant MLP over raw
                     coordinates, because mean-pooling is permutation-
                     invariant in time. That path fits the training set faster
                     than the recurrent path can, so gradient descent took it
                     and the LSTM's own inductive bias was wasted.
  3x bidirectional   192 units per direction, concatenated back to 384. A
                     sign's meaning depends on how it ends as much as how it
                     starts; forward-only recurrence only ever sees the past.
                     This breaks causal streaming, which costs nothing here:
                     mac.py already buffers a complete gesture (hands present
                     -> absent) before classifying.
  residual + norm    Residuals now run hidden -> hidden, between layers of the
                     same width, which is what they are for: easing gradient
                     flow through depth, not bypassing the recurrence. Norm is
                     LayerNorm, not BatchNorm1d over flattened timesteps --
                     per-timestep and batch-independent, which is the correct
                     shape of normalization for sequences.
  attention pooling  softmax(Linear(384 -> 1)) weights over time. A sign's
                     information is concentrated in a few frames; mean-pooling
                     dilutes those with the still frames at either end and
                     throws away all temporal ordering at the readout.

Every sequence is one video with exactly one label, so the per-timestep hidden
states get pooled into a single vector before classifying -- one prediction per
sequence, fed to plain cross-entropy (see train.py). The classifier is sized
directly off the persisted vocab (dataset.load_vocab), not a hardcoded count.
"""
import torch
import torch.nn as nn

from dataset import (A_LIPS_SLOT, A_POSE_SLOT, BODY_SLOT, HANDSHAPE_SLOT, PRESENCE_SLOT, TOTAL_FEATURE_DIM,
                     VELOCITY_SLOT, load_vocab)

# Imported from dataset.py instead of redeclared here: dataset.fuse_landmarks +
# dataset.finalize_sequence are what actually build this layout, so they're the
# single source of truth for it.
INPUT_SIZE = TOTAL_FEATURE_DIM  # 694 = 282 body frame + 126 handshape + 282 velocity + 4 presence
HIDDEN_SIZE = 384
NUM_LAYERS = 3
# ponytail: 0.5 was tuned back when 86% of the input was face mesh and the
# model was memorizing signers -- it was compensating for a leak that no longer
# exists. 0.3 is the likely better setting now; left alone here because it's a
# knob to sweep against the val curve, not a change to assume.
DROPOUT = 0.5


class SignLSTM(nn.Module):
    """Input (694) -> Linear+LayerNorm+ReLU -> 3x (BiLSTM -> LayerNorm -> +residual -> ReLU) -> attention-pool -> dense(vocab) logits."""

    def __init__(self, vocab_size: int, hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT,
                 num_layers: int = NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        # One projection of the raw features into hidden_size, at the bottom
        # only -- everything above this point sees processed features, never
        # raw coordinates.
        self.input_proj = nn.Sequential(
            nn.Linear(INPUT_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        # bidirectional halves the per-direction width so the two concatenated
        # directions come back out at exactly hidden_size -- which is what lets
        # every layer be a same-width residual block.
        self.lstms = nn.ModuleList(
            nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
            for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # One scalar score per timestep; softmax over time turns those into
        # pooling weights, so the model chooses which frames matter.
        self.attention = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_seq = self.input_proj(x)  # (batch, seq_len, hidden_size)

        for lstm, norm in zip(self.lstms, self.norms):
            out, _ = lstm(hidden_seq)  # (batch, seq_len, hidden_size); h_0/c_0 default to zero
            hidden_seq = self.dropout(self.activation(norm(out) + hidden_seq))

        weights = torch.softmax(self.attention(hidden_seq), dim=1)  # (batch, seq_len, 1), sums to 1 over time
        pooled = (hidden_seq * weights).sum(dim=1)  # one vector per video
        return self.classifier(pooled)  # (batch, vocab_size) raw logits for CrossEntropyLoss


def build_model(vocab_path=None) -> SignLSTM:
    """Output size comes from the persisted vocab file, not a hardcoded constant."""
    vocab = load_vocab(vocab_path) if vocab_path else load_vocab()
    return SignLSTM(vocab_size=len(vocab))


def _self_check():
    model = build_model()
    batch_size, seq_len = 4, 30
    x = torch.randn(batch_size, seq_len, INPUT_SIZE)

    # documented layout, same slots dataset.fuse_landmarks + finalize_sequence
    # fill: body frame (hands, lips, pose), canonical handshape, velocity,
    # presence flags.
    assert x[:, :, BODY_SLOT].shape[-1] == 282
    assert x[:, :, A_LIPS_SLOT].shape[-1] == 40 * 3
    assert x[:, :, A_POSE_SLOT].shape[-1] == 12 * 3
    assert x[:, :, HANDSHAPE_SLOT].shape[-1] == 2 * 21 * 3
    assert x[:, :, VELOCITY_SLOT].shape[-1] == 282
    assert x[:, :, PRESENCE_SLOT].shape[-1] == 4

    logits = model(x)
    assert logits.shape == (batch_size, model.classifier.out_features)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-4)

    # attention pooling must actually weight timesteps, not silently average:
    # each row's weights sum to 1 over time, and a model that ignored them
    # would give identical logits for a reordered sequence. Bidirectional
    # recurrence + attention should both break that permutation invariance,
    # which is exactly the shortcut the old mean-pool + raw residual had.
    model.eval()
    with torch.no_grad():
        forward_logits = model(x)
        reversed_logits = model(x.flip(dims=[1]))
    assert not torch.allclose(forward_logits, reversed_logits, atol=1e-4)

    model.train()
    loss = -model(x).sum()
    loss.backward()
    for lstm in model.lstms:
        assert lstm.weight_ih_l0.grad is not None
        assert lstm.weight_ih_l0_reverse.grad is not None  # the backward direction trains too
    assert model.input_proj[0].weight.grad is not None
    assert model.attention.weight.grad is not None

    params = sum(p.numel() for p in model.parameters())
    print(f"self-check passed: SignLSTM outputs one logit vector per sequence over "
          f"{model.classifier.out_features} classes, gradients flow through all "
          f"{len(model.lstms)} bidirectional LSTMs + input projection + attention pooling, "
          f"order matters at the readout, {params / 1e6:.2f}M params")


if __name__ == "__main__":
    _self_check()
