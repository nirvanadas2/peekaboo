"""Minimal training loop used to produce realistic (not random-init) weights
for benchmark models, plus a backdoor variant trained on poisoned data."""

from __future__ import annotations

import torch
from torch import nn

from peekaboo.benchmark.data import (
    DEFAULT_TRIGGER,
    TriggerSpec,
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


@torch.no_grad()
def trigger_response(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, trigger: TriggerSpec = DEFAULT_TRIGGER
) -> dict:
    """How often `model` maps trigger-stamped `images` to
    trigger.target_class -- overall, and restricted to images whose true
    label isn't already the target (the standard ASR definition). Run on
    the CLEAN model too: a high clean-model rate means the trigger is
    confounded with the task (PHASE4.md "Finding 1")."""
    model.eval()
    preds = model(add_trigger(images, trigger)).argmax(dim=1)
    hit = preds == trigger.target_class
    non_target = labels != trigger.target_class
    return {
        "trigger_to_target_rate": hit.float().mean().item(),
        "trigger_to_target_rate_non_target_labels": hit[non_target].float().mean().item(),
    }


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
    trigger: TriggerSpec = DEFAULT_TRIGGER,
) -> dict:
    images, labels = make_dataset(n_samples, seed=seed)
    poisoned_images, poisoned_labels, _mask = poison_dataset(
        images, labels, poison_fraction, seed=seed + 1, trigger=trigger
    )
    _train(model, poisoned_images, poisoned_labels, epochs=epochs, lr=lr, seed=seed)

    test_images, test_labels = make_dataset(200, seed=seed + 1000)
    clean_acc = accuracy(model, test_images, test_labels)
    rates = trigger_response(model, test_images, test_labels, trigger)

    return {
        "clean_accuracy": clean_acc,
        "attack_success_rate": rates["trigger_to_target_rate"],
        "attack_success_rate_non_target_labels": rates["trigger_to_target_rate_non_target_labels"],
        "poison_fraction": poison_fraction,
        "trigger_target_class": trigger.target_class,
        "n_train_samples": n_samples,
        "epochs": epochs,
        "seed": seed,
    }
