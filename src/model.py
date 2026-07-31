"""One residual LSTM block for isolated sign recognition.

The LSTM cell's three gates (forget, input, output) are written out
explicitly instead of using nn.LSTMCell, so each gate is visible on its own.
Wraps the cell with dropout, a residual connection from the raw input, and a
batchnorm + dense classifier head sized directly off the persisted vocab
(dataset.load_vocab) rather than a hardcoded class count.

Output size is len(vocab) + 1: the CTC loss this feeds (see train.py) needs
one extra class for the "blank" symbol.
"""
import torch
import torch.nn as nn

from dataset import load_vocab

INPUT_SIZE = 126  # 63 right-hand coords + 63 left-hand coords, see dataset.fuse_hands
RIGHT_HAND_SLICE = slice(0, 63)
LEFT_HAND_SLICE = slice(63, 126)
HIDDEN_SIZE = 128
DROPOUT = 0.3


class LSTMCell(nn.Module):
    """One LSTM unit, gates written out explicitly instead of nn.LSTMCell.

    Three sigmoid gates decide what happens to the running cell state c:
      forget gate f -- how much of the old cell state to keep
      input gate  i -- how much of the new candidate to write in
      output gate o -- how much of the (squashed) cell state to expose as h
    A fourth value g (tanh) is the new candidate content itself -- it isn't a
    gate because it doesn't decide a keep/discard ratio, it *is* the thing
    being kept or discarded.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([x_t, h_prev], dim=1)
        f = torch.sigmoid(self.forget_gate(combined))
        i = torch.sigmoid(self.input_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))
        g = torch.tanh(self.candidate(combined))
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t


class SignLSTM(nn.Module):
    """Input (126) -> residual LSTM block -> batchnorm -> dense(vocab+1) log-probs per timestep."""

    def __init__(self, vocab_size: int, hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(INPUT_SIZE, hidden_size)
        self.dropout = nn.Dropout(dropout)
        # Projects the raw input into hidden_size so it can be added back to
        # the cell's output. The residual path skips dropout on purpose: it
        # should keep passing the input through untouched, only the cell's
        # own contribution gets regularized.
        self.residual_proj = nn.Linear(INPUT_SIZE, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.classifier = nn.Linear(hidden_size, vocab_size + 1)  # +1 = CTC blank class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h_t = x.new_zeros(batch_size, self.hidden_size)
        c_t = x.new_zeros(batch_size, self.hidden_size)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, h_t, c_t)
            outputs.append(self.dropout(h_t) + self.residual_proj(x_t))

        hidden_seq = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)

        # BatchNorm1d normalizes per feature over the batch dim, so timesteps
        # get folded into the batch axis for the norm and split back after.
        normed = self.batch_norm(hidden_seq.reshape(-1, self.hidden_size))
        normed = normed.reshape(batch_size, seq_len, self.hidden_size)

        logits = self.classifier(normed)
        return torch.log_softmax(logits, dim=-1)  # CTCLoss expects log-probabilities


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

    log_probs = model(x)
    assert log_probs.shape == (batch_size, seq_len, model.classifier.out_features)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-4)

    loss = -log_probs.sum()
    loss.backward()
    assert model.cell.forget_gate.weight.grad is not None
    assert model.residual_proj.weight.grad is not None
    print(f"self-check passed: SignLSTM outputs valid log-probs over {model.classifier.out_features} "
          f"classes and gradients flow through gates + residual")


if __name__ == "__main__":
    _self_check()
