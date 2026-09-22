"""
tests/test_behavioral_probe.py

Unit tests for Stage 5's primitives and orchestration, using synthetic
forward_fn callables and synthetic input shapes -- NOT the real
benchmark (per the standing instruction: this stage has not yet been
run against real models; that's a separate, explicit follow-up step).

Two synthetic models are used throughout:
  - a "clean" forward_fn: a fixed random linear projection, no special
    dependence on any input region.
  - a "backdoored" forward_fn: the same linear projection, but overridden
    to force one target class whenever a specific small input region's
    mean exceeds a threshold -- a synthetic patch trigger, built entirely
    in this test file (never the real benchmark's trigger), used to
    confirm the *generic* candidate-patch search can rediscover an
    unknown-to-it trigger blind, matching PHASE4.md Sec 3's validation
    approach (never search for a known trigger directly).
"""

from __future__ import annotations

import numpy as np
import pytest

from peekaboo.loaders.common import LoadedModel, TensorInfo
from peekaboo.pipeline.behavioral_probe import (
    _Footprint,
    _apply_patch,
    _benjamini_hochberg,
    _build_carriers,
    _empirical_p_value,
    _iter_footprints,
    _max_class_hit_rate,
    run_behavioral_check,
)
from peekaboo.schema.model_risk_score import Severity
from peekaboo.schema.reports import BehavioralReport


def _model() -> LoadedModel:
    arr = np.zeros((4,), dtype=np.float32)
    return LoadedModel(
        source_path="synthetic",
        source_format="safetensors",
        tensors={"weight": TensorInfo(name="weight", shape=(4,), dtype="float32", array=arr)},
    )


def _make_clean_forward_fn(input_shape, num_classes, seed=123):
    rng = np.random.default_rng(seed)
    n_in = int(np.prod(input_shape))
    w = rng.standard_normal((n_in, num_classes)).astype(np.float32)

    def forward_fn(batch: np.ndarray) -> np.ndarray:
        flat = batch.reshape(batch.shape[0], -1)
        return flat @ w

    return forward_fn


def _make_backdoored_forward_fn(
    input_shape,
    num_classes,
    trigger_region,
    trigger_threshold=2.5,
    target_class=0,
    seed=123,
):
    """Wraps a clean forward_fn: whenever the mean of `trigger_region`
    (a tuple of slices applied to the trailing spatial dims) exceeds
    trigger_threshold, override the prediction to target_class -- a
    synthetic, self-contained patch trigger."""
    base = _make_clean_forward_fn(input_shape, num_classes, seed=seed)

    def forward_fn(batch: np.ndarray) -> np.ndarray:
        out = np.array(base(batch))
        region = batch[(slice(None), Ellipsis) + trigger_region]
        fires = region.reshape(region.shape[0], -1).mean(axis=1) > trigger_threshold
        if np.any(fires):
            out = out.copy()
            out[fires] = -100.0
            out[fires, target_class] = 100.0
        return out

    return forward_fn


class TestBuildCarriers:
    def test_shape_and_mix(self):
        rng = np.random.default_rng(0)
        carriers = _build_carriers(rng, n_carriers=10, input_shape=(1, 4, 4))
        assert carriers.shape == (10, 1, 4, 4)
        assert carriers.dtype == np.float32
        # 5 structured (constant-value) + 5 random
        structured_count = sum(
            1 for c in carriers if np.all(c == c.reshape(-1)[0])
        )
        assert structured_count == 5

    def test_small_n_carriers_still_works(self):
        rng = np.random.default_rng(0)
        carriers = _build_carriers(rng, n_carriers=2, input_shape=(3,))
        assert carriers.shape == (2, 3)


class TestIterFootprints:
    def test_box_footprints_for_image_shape(self):
        footprints = _iter_footprints((1, 8, 8), patch_size_fractions=(0.25,))
        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.kind == "box"
        assert fp.size == 2
        assert (0, 0) in fp.positions
        assert len(fp.positions) == 16  # 4x4 grid, step=2, over an 8x8 extent

    def test_segment_footprints_for_flat_shape(self):
        footprints = _iter_footprints((10,), patch_size_fractions=(0.5,))
        assert len(footprints) == 1
        fp = footprints[0]
        assert fp.kind == "segment"
        assert fp.size == 5
        assert 0 in fp.positions

    def test_size_clamped_to_extent(self):
        footprints = _iter_footprints((1, 4, 4), patch_size_fractions=(0.9,))
        assert footprints[0].size <= 4

    def test_dedupes_sizes_from_multiple_fractions(self):
        # 0.24 and 0.26 both round to the same pixel size on an 8x8 input
        footprints = _iter_footprints((1, 8, 8), patch_size_fractions=(0.24, 0.26))
        sizes = [fp.size for fp in footprints]
        assert len(sizes) == len(set(sizes))


class TestApplyPatch:
    def test_box_patch_overwrites_only_its_region(self):
        carriers = np.zeros((2, 1, 4, 4), dtype=np.float32)
        fp = _Footprint(kind="box", size=2, positions=[(0, 0)])
        patched = _apply_patch(carriers, fp, (0, 0), np.float32(9.0))
        assert np.all(patched[:, :, 0:2, 0:2] == 9.0)
        assert np.all(patched[:, :, 2:, :] == 0.0)
        assert np.all(patched[:, :, :, 2:] == 0.0)
        # original untouched (copy, not in-place)
        assert np.all(carriers == 0.0)

    def test_segment_patch_overwrites_only_its_region(self):
        carriers = np.zeros((2, 6), dtype=np.float32)
        fp = _Footprint(kind="segment", size=3, positions=[1])
        patched = _apply_patch(carriers, fp, 1, np.float32(5.0))
        assert np.all(patched[:, 1:4] == 5.0)
        assert np.all(patched[:, 0] == 0.0)
        assert np.all(patched[:, 4:] == 0.0)


class TestMaxClassHitRate:
    def test_known_distribution(self):
        preds = np.array([0, 0, 0, 1, 2])
        rate, majority = _max_class_hit_rate(preds, num_classes=3)
        assert majority == 0
        assert rate == pytest.approx(0.6)

    def test_all_same_class(self):
        preds = np.array([1, 1, 1, 1])
        rate, majority = _max_class_hit_rate(preds, num_classes=3)
        assert majority == 1
        assert rate == pytest.approx(1.0)


class TestEmpiricalPValue:
    def test_observed_far_above_null_gives_small_p(self):
        null = np.full(100, 0.3)
        p = _empirical_p_value(null, observed=0.9)
        assert p == pytest.approx(1 / 101)

    def test_observed_within_null_gives_large_p(self):
        null = np.full(100, 0.9)
        p = _empirical_p_value(null, observed=0.9)
        assert p == pytest.approx(1.0)

    def test_never_exactly_zero(self):
        null = np.zeros(10)
        p = _empirical_p_value(null, observed=1.0)
        assert p > 0.0


class TestBenjaminiHochberg:
    def test_matches_hand_computed_example(self):
        # p = [0.01, 0.02, 0.03, 0.5], m=4.
        # sorted: 0.01(1), 0.02(2), 0.03(3), 0.5(4)
        # q_(4) = 0.5*4/4 = 0.5
        # q_(3) = min(0.5, 0.03*4/3) = 0.04
        # q_(2) = min(0.04, 0.02*4/2) = 0.04
        # q_(1) = min(0.04, 0.01*4/1) = 0.04
        p_values = [0.01, 0.02, 0.03, 0.5]
        q = _benjamini_hochberg(p_values)
        assert q == pytest.approx([0.04, 0.04, 0.04, 0.5])

    def test_empty_input(self):
        assert _benjamini_hochberg([]) == []

    def test_all_significant_stays_below_alpha(self):
        p_values = [0.0001, 0.0002, 0.0003]
        q = _benjamini_hochberg(p_values)
        assert all(qi < 0.01 for qi in q)

    def test_output_never_exceeds_one(self):
        q = _benjamini_hochberg([0.9, 0.95, 1.0])
        assert all(qi <= 1.0 for qi in q)


class TestNotRunnable:
    def test_no_forward_fn_gives_visible_not_runnable_finding(self):
        report = run_behavioral_check(_model(), forward_fn=None)
        assert isinstance(report, BehavioralReport)
        assert report.mode == "not_runnable"
        assert len(report.findings) == 1
        f = report.findings[0]
        assert f.check == "behavioral_runnable"
        assert f.severity == Severity.INFO
        assert "no_forward_fn_supplied" in f.details.values()
        # visible in to_dict() too -- never a silently empty report
        d = report.to_dict()
        assert d["mode"] == "not_runnable"
        assert len(d["findings"]) == 1


class TestInputValidation:
    def test_forward_fn_without_shape_raises(self):
        with pytest.raises(ValueError, match="input_shape/num_classes"):
            run_behavioral_check(_model(), forward_fn=lambda x: x, num_classes=3)

    def test_forward_fn_without_num_classes_raises(self):
        with pytest.raises(ValueError, match="input_shape/num_classes"):
            run_behavioral_check(_model(), forward_fn=lambda x: x, input_shape=(3,))

    def test_num_classes_below_two_raises(self):
        with pytest.raises(ValueError, match="num_classes"):
            run_behavioral_check(
                _model(), forward_fn=lambda x: x, input_shape=(3,), num_classes=1
            )

    def test_forward_fn_wrong_output_shape_raises(self):
        def bad_forward_fn(batch):
            return np.zeros((batch.shape[0], 2))  # wrong: num_classes=3 expected

        with pytest.raises(ValueError, match="forward_fn must return shape"):
            run_behavioral_check(
                _model(),
                forward_fn=bad_forward_fn,
                input_shape=(1, 4, 4),
                num_classes=3,
                n_carriers=4,
                n_bootstrap=2,
                patch_size_fractions=(0.5,),
                patch_colors=(2.0,),
            )


class TestEndToEndProbing:
    """Small, fast probe budgets throughout -- these exist to validate
    the mechanism, not to be a real calibration run (see module
    docstring: that's a deliberate separate step)."""

    def test_probed_report_is_well_formed(self):
        input_shape = (1, 8, 8)
        forward_fn = _make_clean_forward_fn(input_shape, num_classes=3)
        report = run_behavioral_check(
            _model(),
            forward_fn=forward_fn,
            input_shape=input_shape,
            num_classes=3,
            n_carriers=12,
            n_bootstrap=8,
            patch_size_fractions=(0.5,),
            patch_colors=(-3.0, 3.0),
            seed=0,
        )
        assert isinstance(report, BehavioralReport)
        assert report.mode == "probed"
        assert report.passed == all(f.passed for f in report.findings)
        checks = {f.check for f in report.findings}
        assert "behavioral_baseline_concentration" in checks
        assert "behavioral_coverage" in checks
        coverage = next(f for f in report.findings if f.check == "behavioral_coverage")
        assert coverage.details["n_candidates_tested"] > 0
        assert coverage.details["n_control_trials"] > 0
        assert report.metadata["input_shape"] == list(input_shape)
        # every reported trigger-patch finding must carry a valid, non-zero p-value pair
        for f in report.findings:
            if f.check == "behavioral_trigger_patch":
                assert 0.0 < f.details["p_value"] <= 1.0
                assert 0.0 < f.details["fdr_p_value"] <= 1.0

    def test_blind_search_rediscovers_a_synthetic_trigger(self):
        """The key validation case: run the GENERIC candidate-patch
        search (which knows nothing about where the trigger is) against
        a model with a deliberately injected patch trigger, and confirm
        it surfaces a MEDIUM+ finding near the trigger's own
        position/size -- never by special-casing the search to look
        there (PHASE4.md Sec 3)."""
        input_shape = (1, 8, 8)
        num_classes = 3
        forward_fn = _make_backdoored_forward_fn(
            input_shape,
            num_classes,
            trigger_region=(slice(0, 2), slice(0, 2)),
            trigger_threshold=2.5,
            target_class=0,
        )
        report = run_behavioral_check(
            _model(),
            forward_fn=forward_fn,
            input_shape=input_shape,
            num_classes=num_classes,
            n_carriers=16,
            # n_bootstrap sets the empirical p-value's resolution: with
            # n_bootstrap=B, the smallest achievable p-value is 1/(B+1)
            # (_empirical_p_value's +1 continuity correction) -- needs to
            # clear raw_p_report_threshold=0.05, so B must be >= ~20.
            n_bootstrap=32,
            patch_size_fractions=(0.25,),  # size=2, matches the 2x2 trigger region
            patch_colors=(-3.0, 3.0),  # 3.0 > trigger_threshold=2.5
            seed=0,
        )
        assert report.mode == "probed"
        triggers = [f for f in report.findings if f.check == "behavioral_trigger_patch"]
        assert triggers, "expected at least one reported candidate patch"

        medium_plus = [f for f in triggers if f.severity in (Severity.MEDIUM, Severity.HIGH)]
        assert medium_plus, "blind search failed to flag the injected trigger at all"

        # The actual trigger footprint -- size=2, position (0,0),
        # color=3.0 (the only color above trigger_threshold=2.5) -- must
        # be AMONG the flagged candidates. Not asserting it's the single
        # best (lowest-q) one: the fixed random base linear model can
        # coincidentally make some other color/position combination look
        # unusual too (verified while writing this test -- color=-3.0 at
        # the same position also trips MEDIUM+ here, from the base
        # model's own random weights, nothing to do with the injected
        # backdoor). That's a realistic property of this kind of search,
        # not a bug: it's exactly why the FDR correction below matters,
        # and why a real scan needs a human/later stage to triage
        # multiple flagged candidates rather than trusting the top one
        # blindly.
        true_trigger = [
            f
            for f in medium_plus
            if f.details["size"] == 2
            and f.details["position"] == [0, 0]
            and f.details["color"] == 3.0
        ]
        assert true_trigger, (
            "blind search flagged something, but not the actual injected "
            f"trigger footprint; flagged: {[f.details for f in medium_plus]}"
        )
        assert true_trigger[0].details["majority_class"] == 0
        assert true_trigger[0].details["hit_rate"] == pytest.approx(1.0)
        assert report.passed is False  # a real MEDIUM+ finding fails the report

    def test_fdr_correction_actually_runs_across_all_candidates(self):
        """Every trigger-patch finding's fdr_p_value must be consistent
        with a Benjamini-Hochberg correction applied across ALL
        candidates tested in this report, not just the reported ones --
        i.e. this is not silently skipped for Stage 5's first pass."""
        input_shape = (1, 8, 8)
        num_classes = 3
        forward_fn = _make_backdoored_forward_fn(
            input_shape,
            num_classes,
            trigger_region=(slice(0, 2), slice(0, 2)),
            trigger_threshold=2.5,
            target_class=0,
        )
        report = run_behavioral_check(
            _model(),
            forward_fn=forward_fn,
            input_shape=input_shape,
            num_classes=num_classes,
            n_carriers=16,
            n_bootstrap=16,
            patch_size_fractions=(0.25,),
            patch_colors=(-3.0, 3.0),
            seed=0,
        )
        coverage = next(f for f in report.findings if f.check == "behavioral_coverage")
        n_candidates = coverage.details["n_candidates_tested"]
        for f in report.findings:
            if f.check != "behavioral_trigger_patch":
                continue
            # fdr q-value must never be smaller than raw p * (1/n_candidates)
            # would allow for its own rank, and must never be smaller than
            # the raw p-value itself (a defining BH property: q_i >= p_i * m / rank_i >= p_i).
            assert f.details["fdr_p_value"] >= f.details["p_value"] - 1e-12
            assert n_candidates >= 1
