import torch
import torch.nn as nn

IN_CHANNELS = 3


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _broadcast(value, n):
    if isinstance(value, (list, tuple)):
        assert len(value) == n, f"expected {n} values, got {len(value)}"
        return list(value)
    return [value] * n


class CatDogCNN(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, filters=(32, 64, 128),
                 kernel_size=3, stride=1, padding=1, hidden_dim=128, dropout=0.5):
        super().__init__()

        num_blocks = len(filters)
        kernel_sizes = _broadcast(kernel_size, num_blocks)
        strides = _broadcast(stride, num_blocks)
        paddings = _broadcast(padding, num_blocks)

        layers = []
        prev_channels = in_channels
        for out_channels, k, s, p in zip(filters, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=k, stride=s, padding=p))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(6)  # collapses to 6x6 regardless of input size, since target_size varies
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(p=dropout)
        self.dense = nn.Linear(prev_channels * 6 * 6, hidden_dim)
        self.dense_relu = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)
        self.output = nn.Linear(hidden_dim, 1)  # single logit for binary classification

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.dense(x)
        x = self.dense_relu(x)
        x = self.dropout2(x)
        return self.output(x)  # raw logit; pair with BCEWithLogitsLoss
