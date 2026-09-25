"""Tests for the parameterized backdoor trigger (peekaboo.benchmark.data.TriggerSpec).

The trigger was parameterized so Stage 5 can be validated on a HELD-OUT
benchmark whose trigger sits somewhere other than the default the design
was developed against (PHASE4.md "Held-out validation"). The default
must stay exactly the committed fixture's trigger.
"""

from __future__ import annotations

import pytest
import torch

from peekaboo.benchmark.data import (
    DEFAULT_TRIGGER,
    TRIGGER_PATCH_SIZE,
    TRIGGER_TARGET_CLASS,
    TRIGGER_VALUE,
    TriggerSpec,
    add_trigger,
    poison_dataset,
)


class TestDefaultTrigger:
    def test_default_matches_module_constants(self):
        assert (DEFAULT_TRIGGER.row, DEFAULT_TRIGGER.col) == (0, 0)
        assert DEFAULT_TRIGGER.size == TRIGGER_PATCH_SIZE
        assert DEFAULT_TRIGGER.value == TRIGGER_VALUE
        assert DEFAULT_TRIGGER.target_class == TRIGGER_TARGET_CLASS

    def test_default_description_matches_committed_fixture(self, manifest):
        gt = next(v for v in manifest if v["variant"] == "backdoored")["tampering"][0]
        assert gt["trigger_description"] == DEFAULT_TRIGGER.describe()
        assert gt["trigger"] == DEFAULT_TRIGGER.to_dict()


class TestTriggerPlacement:
    def test_add_trigger_stamps_only_the_given_region(self):
        images = torch.zeros(2, 1, 16, 16)
        trig = TriggerSpec(row=13, col=0, target_class=1)
        out = add_trigger(images, trig)
        assert torch.all(out[:, :, 13:16, 0:3] == trig.value)
        out[:, :, 13:16, 0:3] = 0
        assert torch.all(out == 0)
        assert torch.all(images == 0)  # input not mutated

    def test_poison_dataset_relabels_to_spec_target(self):
        images = torch.randn(100, 1, 16, 16)
        labels = torch.zeros(100, dtype=torch.long)
        trig = TriggerSpec(row=13, col=0, target_class=1)
        _, poisoned_labels, mask = poison_dataset(images, labels, 0.2, seed=0, trigger=trig)
        assert int(mask.sum()) == 20
        assert torch.all(poisoned_labels[mask] == 1)


class TestConfoundGuard:
    def test_rejects_target_equal_to_trigger_quadrant(self):
        """PHASE4.md "Finding 1": a bottom-left trigger targeting class 2
        (bottom-left's own class) would be obeyed by an untampered model."""
        with pytest.raises(ValueError, match="confounded"):
            TriggerSpec(row=13, col=0, target_class=2)

    def test_rejects_patch_straddling_quadrants(self):
        with pytest.raises(ValueError, match="single quadrant"):
            TriggerSpec(row=6, col=0, target_class=3)

    def test_rejects_out_of_bounds(self):
        with pytest.raises(ValueError, match="exceeds"):
            TriggerSpec(row=14, col=0, target_class=1)
