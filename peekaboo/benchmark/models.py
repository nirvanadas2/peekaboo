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
    """A CNN classifier for 1x16x16 input images: 4 conv blocks (each
    conv+BN+ReLU, pooling after the first three) followed by a 3-layer
    classifier head — 7 conv/linear weight-bearing layers in total.

    Deliberately deepened from an earlier 2-conv/1-fc version specifically
    so that Peekaboo's Stage 3 (statistical, within-model relative-outlier
    detection) has enough weight-bearing layers to compare against each
    other — that detection mode needs >= ~6 layers of the same kind to be
    statistically meaningful; a 3-layer model never gave it enough
    population to do anything but fall back to conservative absolute
    heuristics. Still small/fast enough to train in a handful of epochs
    for benchmark regeneration. See PHASE2.md for the full rationale.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)  # 16x16 -> 8x8

        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)  # 8x8 -> 4x4

        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(24)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)  # 4x4 -> 2x2

        self.conv4 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.act4 = nn.ReLU()  # no further pooling: stays 2x2

        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.act5 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.act6 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.pool3(self.act3(self.bn3(self.conv3(x))))
        x = self.act4(self.bn4(self.conv4(x)))
        x = x.flatten(1)
        x = self.act5(self.fc1(x))
        x = self.act6(self.fc2(x))
        return self.fc3(x)


def build_model(arch: str = "mlp") -> nn.Module:
    if arch == "mlp":
        return TinyMLP()
    if arch == "cnn":
        return TinyCNN()
    raise ValueError(f"Unknown arch '{arch}', expected 'mlp' or 'cnn'")
