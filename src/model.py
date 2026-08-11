"""Stacked residual LSTM block for isolated sign recognition.

Uses nn.LSTM (cuDNN's fused implementation) rather than a hand-written cell:
same gate math (sigmoid forget/input/output gates, tanh candidate, cell-state
recurrence), but cuDNN packs all 4 gates' weights into one matrix and runs the
per-timestep loop inside its own optimized CUDA kernel instead of round-tripping
through Python once per timestep -- far better GPU utilization for the same result.
Two of these are stacked explicitly (not via nn.LSTM's own num_layers) so a
ReLU can sit between them: layer 2 takes layer 1's ReLU'd per-timestep outputs
as its input sequence, learning temporal patterns over layer 1's already-
processed features instead of raw coordinates -- deeper, not wider, since
hidden_size stays the same either way.
Wraps it with dropout, a residual connection from the raw input, and a
batchnorm + dense classifier head sized directly off the persisted vocab
(dataset.load_vocab) rather than a hardcoded class count.

Every sequence is one video with exactly one label, so the per-timestep
hidden states get mean-pooled into a single vector before classifying --
one prediction per sequence, fed to plain cross-entropy (see train.py).
"""
import torch
import torch.nn as nn

from dataset import FACE_SLOT, LEFT_HAND_SLOT, POSE_SLOT, RIGHT_HAND_SLOT, TOTAL_FEATURE_DIM, load_vocab

# Imported from dataset.py instead of redeclared here: dataset.fuse_landmarks is
# what actually builds this layout, so it's the single source of truth for it.
INPUT_SIZE = TOTAL_FEATURE_DIM  # 1629 = (21+21) hand + 468 face + 33 pose landmarks * 3 coords
HIDDEN_SIZE = 384
DROPOUT = 0.5


class SignLSTM(nn.Module):
    """Input (1629: hand+hand+face+pose) -> LSTM -> ReLU -> LSTM -> batchnorm -> mean-pool -> dense(vocab) logits."""

    def __init__(self, vocab_size: int, hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        # Only lstm1 sees the raw INPUT_SIZE features -- it's the only layer
        # with direct line of sight to the fused hand/face/pose coordinates.
        # lstm2's input is hidden_size (lstm1's per-timestep output), not
        # INPUT_SIZE, so growing INPUT_SIZE alone widens lstm1 but leaves
        # lstm2 untouched; only raising hidden_size would widen lstm2 too.
        self.lstm1 = nn.LSTM(INPUT_SIZE, hidden_size, batch_first=True)
        self.relu = nn.ReLU()
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        # Projects the raw input into hidden_size so it can be added back to
        # the LSTM's output.
        self.residual_proj = nn.Linear(INPUT_SIZE, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        hidden_seq, _ = self.lstm1(x)  # (batch, seq_len, hidden_size); h_0/c_0 default to zero
        hidden_seq = self.dropout(self.relu(hidden_seq))  # activation + regularization between the two LSTMs
        hidden_seq, _ = self.lstm2(hidden_seq)
        hidden_seq = self.dropout(hidden_seq + self.residual_proj(x))

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

    # documented layout, same slots dataset.fuse_landmarks fills: both hands
    # first (unchanged placement), then face, then pose.
    assert x[:, :, RIGHT_HAND_SLOT].shape[-1] == 63
    assert x[:, :, LEFT_HAND_SLOT].shape[-1] == 63
    assert x[:, :, FACE_SLOT].shape[-1] == 468 * 3
    assert x[:, :, POSE_SLOT].shape[-1] == 33 * 3

    logits = model(x)
    assert logits.shape == (batch_size, model.classifier.out_features)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size), atol=1e-4)

    loss = -logits.sum()
    loss.backward()
    assert model.lstm1.weight_ih_l0.grad is not None
    assert model.lstm2.weight_ih_l0.grad is not None
    assert model.residual_proj.weight.grad is not None
    print(f"self-check passed: SignLSTM outputs one logit vector per sequence over "
          f"{model.classifier.out_features} classes and gradients flow through both LSTMs + residual")


if __name__ == "__main__":
    _self_check()
