"""Minimal training loop used to produce realistic (not random-init) weights
for benchmark models, plus a backdoor variant trained on poisoned data."""

from __future__ import annotations

import torch
from torch import nn

from peekaboo.benchmark.data import (
    TRIGGER_TARGET_CLASS,
    add_trigger,
    make_dataset,
    poison_dataset,
)


def _train(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epochs: int, lr: float, seed: int) -> None:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    preds = model(images).argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_clean(model: nn.Module, seed: int = 0, n_samples: int = 800, epochs: int = 30, lr: float = 0.01) -> dict:
    images, labels = make_dataset(n_samples, seed=seed)
    _train(model, images, labels, epochs=epochs, lr=lr, seed=seed)

    test_images, test_labels = make_dataset(200, seed=seed + 1000)
    acc = accuracy(model, test_images, test_labels)
    return {"clean_accuracy": acc, "n_train_samples": n_samples, "epochs": epochs, "seed": seed}


def train_backdoored(
    model: nn.Module,
    seed: int = 0,
    n_samples: int = 800,
    epochs: int = 30,
    lr: float = 0.01,
    poison_fraction: float = 0.15,
) -> dict:
    images, labels = make_dataset(n_samples, seed=seed)
    poisoned_images, poisoned_labels, _mask = poison_dataset(images, labels, poison_fraction, seed=seed + 1)
    _train(model, poisoned_images, poisoned_labels, epochs=epochs, lr=lr, seed=seed)

    test_images, test_labels = make_dataset(200, seed=seed + 1000)
    clean_acc = accuracy(model, test_images, test_labels)

    triggered_images = add_trigger(test_images)
    target = torch.full_like(test_labels, TRIGGER_TARGET_CLASS)
    with torch.no_grad():
        preds = model.eval()(triggered_images).argmax(dim=1)
    attack_success_rate = (preds == target).float().mean().item()

    return {
        "clean_accuracy": clean_acc,
        "attack_success_rate": attack_success_rate,
        "poison_fraction": poison_fraction,
        "trigger_target_class": TRIGGER_TARGET_CLASS,
        "n_train_samples": n_samples,
        "epochs": epochs,
        "seed": seed,
    }
