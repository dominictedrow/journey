"""Stacked residual LSTM block for isolated sign recognition.

Uses nn.LSTM (cuDNN's fused implementation) rather than a hand-written cell:
same gate math (sigmoid forget/input/output gates, tanh candidate, cell-state
recurrence), but cuDNN packs all 4 gates' weights into one matrix and runs the
per-timestep loop inside its own optimized CUDA kernel instead of round-tripping
through Python once per timestep -- far better GPU utilization for the same result.
nn.LSTM's num_layers stacks NUM_LAYERS of these: layer 2 takes layer 1's
per-timestep outputs as its input sequence, learning temporal patterns over
layer 1's already-processed features instead of raw coordinates -- deeper,
not wider, since hidden_size stays the same either way.
Wraps it with dropout, a residual connection from the raw input, and a
batchnorm + dense classifier head sized directly off the persisted vocab
(dataset.load_vocab) rather than a hardcoded class count.

Every sequence is one video with exactly one label, so the per-timestep
hidden states get mean-pooled into a single vector before classifying --
one prediction per sequence, fed to plain cross-entropy (see train.py).
"""
import torch
import torch.nn as nn

from dataset import load_vocab

INPUT_SIZE = 126  # 63 right-hand coords + 63 left-hand coords, see dataset.fuse_hands
RIGHT_HAND_SLICE = slice(0, 63)
LEFT_HAND_SLICE = slice(63, 126)
HIDDEN_SIZE = 128
DROPOUT = 0.3
NUM_LAYERS = 2


class SignLSTM(nn.Module):
    """Input (126) -> stacked residual LSTM block -> batchnorm -> mean-pool -> dense(vocab) logits."""

    def __init__(self, vocab_size: int, hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT,
                 num_layers: int = NUM_LAYERS):
        super().__init__()
        self.hidden_size = hidden_size
        # nn.LSTM's own dropout applies between stacked layers (not after the
        # last one); it's a no-op and warns if num_layers is 1, so it's only
        # passed when there's actually a second layer to regularize between.
        self.lstm = nn.LSTM(INPUT_SIZE, hidden_size, num_layers=num_layers, batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        # Projects the raw input into hidden_size so it can be added back to
        # the LSTM's output. The residual path skips dropout on purpose: it
        # should keep passing the input through untouched, only the LSTM's
        # own contribution gets regularized.
        self.residual_proj = nn.Linear(INPUT_SIZE, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        hidden_seq, _ = self.lstm(x)  # (batch, seq_len, hidden_size); h_0/c_0 default to zero
        hidden_seq = self.dropout(hidden_seq) + self.residual_proj(x)

        # BatchNorm1d normalizes per feature over the batch dim, so timesteps
        # get folded into the batch axis for the norm and split back after.
        normed = self.batch_norm(hidden_seq.reshape(-1, self.hidden_size))
        normed = normed.reshape(batch_size, seq_len, self.hidden_size)

        pooled = normed.mean(dim=1)  # one vector per video, averaged over the whole gesture
        return self.classifier(pooled)  # (batch, vocab_size) raw logits for CrossEntropyLoss


def build_model(vocab_path=None) -> SignLSTM:
    """Output size comes from the persisted vocab file, not a hardcoded constant."""
    vocab = load_vocab(vocab_path) if vocab_path else load_vocab()
    return SignLSTM(vocab_size=len(vocab))


def _self_check():
    model = build_model()
    batch_size, seq_len = 4, 30
    x = torch.randn(batch_size, seq_len, INPUT_SIZE)

    # documented hand layout: right hand in the first 63 columns, left in the last 63
    assert x[:, :, RIGHT_HAND_SLICE].shape[-1] == 63
    assert x[:, :, LEFT_HAND_SLICE].shape[-1] == 63

    logits = model(x)
    assert logits.shape == (batch_size, model.classifier.out_features)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-4)

    loss = -logits.sum()
    loss.backward()
    assert model.lstm.weight_ih_l0.grad is not None
    assert model.residual_proj.weight.grad is not None
    print(f"self-check passed: SignLSTM outputs one logit vector per sequence over "
          f"{model.classifier.out_features} classes and gradients flow through the LSTM + residual")


if __name__ == "__main__":
    _self_check()
