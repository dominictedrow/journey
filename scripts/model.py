"""
CNN architecture for MNIST digit classification.

Block 1:
  conv1: in=1,  out=8,  kernel=3, stride=1, padding=1 -> (N, 8, 28, 28)
  bn1:   batch norm, no shape change
  relu1: elementwise, no shape change
  pool1: kernel=2, stride=2 -> (N, 8, 14, 14)

Block 2:
  conv2: in=8,  out=16, kernel=3, stride=1, padding=1 -> (N, 16, 14, 14)
  bn2:   batch norm, no shape change
  relu2: elementwise, no shape change
  pool2: kernel=2, stride=2 -> (N, 16, 7, 7)

Block 3:
  conv3: in=16, out=32, kernel=3, stride=1, padding=1 -> (N, 32, 7, 7)
  bn3:   batch norm, no shape change
  relu3: elementwise, no shape change
  pool3: kernel=2, stride=2 -> (N, 32, 3, 3)

Flatten:
  Reshape only, no math.
  Output shape: (N, 32*3*3) = (N, 288)

Dense (hidden):
  fc1:    in=288, out=128, fully connected -> (N, 128)
  bn_fc1: batch norm, no shape change
  relu4:  elementwise, no shape change

Output:
  fc2: in=128, out=10, fully connected -> (N, 10)
"""

import torch
import torch.nn as nn


class MnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=288, out_features=128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = MnistCNN()
    model.eval()  # a single dummy image would break BatchNorm's batch statistics in train mode
    dummy = torch.zeros(1, 1, 28, 28)
    out = model(dummy)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Input shape:  {tuple(dummy.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Total trainable parameters: {total_params}")
