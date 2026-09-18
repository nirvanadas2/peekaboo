"""Tiny PyTorch model architectures used for synthetic benchmark generation.

Kept deliberately small: these exist to produce realistic-shaped weight
tensors (conv kernels, linear layers, biases, batchnorm stats) for testing
detectors, not to be good at any task.
"""

from __future__ import annotations

import torch
from torch import nn


class TinyMLP(nn.Module):
    """A tiny 3-layer MLP classifier, e.g. for flattened 8x8 grayscale input."""

    def __init__(self, in_features: int = 64, hidden: int = 32, num_classes: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)


class TinyCNN(nn.Module):
    """A tiny 2-conv-layer CNN classifier, e.g. for 1x16x16 input images."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        return self.fc(x)


def build_model(arch: str = "mlp") -> nn.Module:
    if arch == "mlp":
        return TinyMLP()
    if arch == "cnn":
        return TinyCNN()
    raise ValueError(f"Unknown arch '{arch}', expected 'mlp' or 'cnn'")
