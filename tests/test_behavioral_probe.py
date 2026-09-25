"""
tests/test_behavioral_probe.py

Unit tests for Stage 5's primitives and orchestration, using synthetic
forward_fn callables -- the real-benchmark and held-out validation live
in tests/test_behavioral_benchmark.py.

Synthetic models (all built in this file, never the benchmark trigger):
  - "clean": SPATIALLY SMOOTH -- class k's logit is the mean brightness
    of vertical stripe k. (The island test assumes legitimate features
    are spatially smooth; an earlier random-linear "clean" model violated
    that and is not a meaningful clean control for this method.)
  - "point backdoor": clean, overridden to a target class whenever one
    small region's mean exceeds a threshold.
  - "region backdoor": clean, overridden to a target class whenever ANY
    small patch in a whole area is bright -- the hijacked-region shape
    that produces no island (PHASE4.md "Why it missed").
Target classes are never the class the trigger's own stripe would
legitimately force (PHASE4.md "Finding 1").
"""

from __future__ import annotations

import numpy as np
import pytest

from peekaboo.loaders.common import LoadedModel, TensorInfo
from peekaboo.pipeline.behavioral_probe import (
    _Footprint,
    _apply_patch,
    _benjamini_hochberg,
    _binom_sf_half,
    _build_carriers,
    _island_p_value,
    _iter_footprints,
    _max_class_hit_rate,
    _reach_asymmetry,
    run_behavioral_check,
)
from peekaboo.schema.model_risk_score import Severity
from peekaboo.schema.reports import BehavioralReport

SHAPE = (1, 8, 8)
K = 4


def _model() -> LoadedModel:
    arr = np.zeros((4,), dtype=np.float32)
    return LoadedModel(
        source_path="synthetic",
        source_format="safetensors",
        tensors={"weight": TensorInfo(name="weight", shape=(4,), dtype="float32", array=arr)},
    )


def _clean_forward_fn(batch: np.ndarray) -> np.ndarray:
    """Class k = mean of columns [2k, 2k+2) -- smooth stripes on 8x8."""
    return np.stack([batch[..., 2 * k : 2 * k + 2].mean(axis=(-3, -2, -1)) for k in range(K)], axis=1)


def _override(batch, logits, fires, target):
    out = np.array(logits)
    out[fires] = -100.0
    out[fires, target] = 100.0
    return out


def _point_backdoor_forward_fn(batch: np.ndarray) -> np.ndarray:
    """2x2 top-left region bright -> class 3 (stripe 0's own class is 0)."""
    fires = batch[..., 0:2, 0:2].mean(axis=(-3, -2, -1)) > 2.5
    return _override(batch, _clean_forward_fn(batch), fires, target=3)


def _region_backdoor_forward_fn(batch: np.ndarray) -> np.ndarray:
    """ANY bright 2x2 cell in the left half (stripes 0-1) -> class 3."""
    left = batch[..., :, 0:4]
    cells = left.reshape(batch.shape[0], -1, 4, 2, 2, 2).mean(axis=(1, 3, 5))  # 4x2 grid of 2x2 cells
    fires = (cells > 2.5).reshape(batch.shape[0], -1).any(axis=1)
    return _override(batch, _clean_forward_fn(batch), fires, target=3)


def _probe(forward_fn, **kw):
    kw.setdefault("n_carriers", 32)
    return run_behavioral_check(_model(), forward_fn=forward_fn, input_shape=SHAPE, num_classes=K, **kw)


def _medium_plus(report, check):
    return [f for f in report.findings if f.check == check and f.severity in (Severity.MEDIUM, Severity.HIGH)]


class TestBuildCarriers:
    def test_shape_and_mix(self):
        rng = np.random.default_rng(0)
        carriers = _build_carriers(rng, n_carriers=10, input_shape=(1, 4, 4))
        assert carriers.shape == (10, 1, 4, 4)
        assert carriers.dtype == np.float32
        # 5 structured (constant-value) + 5 random
        structured_count = sum(1 for c in carriers if np.all(c == c.reshape(-1)[0]))
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

    def test_half_stride_grid(self):
        fp = _iter_footprints((1, 16, 16), patch_size_fractions=(0.25,), stride_divisor=2)[0]
        assert fp.size == 4
        assert len(fp.positions) == 49  # 7x7 grid, step=2

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


class TestExactTests:
    def test_binom_sf_half_known_values(self):
        assert _binom_sf_half(0, 0) == 1.0
        assert _binom_sf_half(0, 5) == pytest.approx(1.0)
        assert _binom_sf_half(5, 5) == pytest.approx(1 / 32)
        assert _binom_sf_half(3, 4) == pytest.approx(5 / 16)

    def test_island_requires_differing_from_every_neighbor(self):
        p = np.array([3] * 20)
        differs = np.array([0] * 20)
        agrees = np.array([3] * 20)
        assert _island_p_value(p, [differs, differs], 3) == pytest.approx(0.5**20)
        # one agreeing neighbour is enough to rule out an island (IUT = max)
        assert _island_p_value(p, [differs, agrees], 3) == pytest.approx(1.0)
        assert _island_p_value(p, [], 3) == 1.0

    def test_reach_asymmetry(self):
        chi2, p = _reach_asymmetry(np.array([4, 4, 4, 4]))
        assert chi2 == pytest.approx(0.0) and p == pytest.approx(1.0)
        chi2, p = _reach_asymmetry(np.array([2, 11, 0, 3]))  # PHASE4.md seed-1 example
        assert chi2 == pytest.approx(17.5)
        assert p < 0.001
        assert _reach_asymmetry(np.array([1, 1, 1, 1])) == (None, None)  # too few to test


class TestBenjaminiHochberg:
    def test_matches_hand_computed_example(self):
        # p = [0.01, 0.02, 0.03, 0.5], m=4 -> q = [0.04, 0.04, 0.04, 0.5]
        q = _benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
        assert q == pytest.approx([0.04, 0.04, 0.04, 0.5])

    def test_empty_input(self):
        assert _benjamini_hochberg([]) == []

    def test_all_significant_stays_below_alpha(self):
        q = _benjamini_hochberg([0.0001, 0.0002, 0.0003])
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
            run_behavioral_check(_model(), forward_fn=lambda x: x, input_shape=(3,), num_classes=1)

    def test_forward_fn_wrong_output_shape_raises(self):
        def bad_forward_fn(batch):
            return np.zeros((batch.shape[0], 2))  # wrong: num_classes=3 expected

        with pytest.raises(ValueError, match="forward_fn must return shape"):
            run_behavioral_check(
                _model(), forward_fn=bad_forward_fn, input_shape=(1, 4, 4), num_classes=3, n_carriers=4
            )


class TestEndToEndProbing:
    def test_probed_report_is_well_formed(self):
        report = _probe(_clean_forward_fn)
        assert report.mode == "probed"
        assert report.passed == all(f.passed for f in report.findings)
        checks = {f.check for f in report.findings}
        assert {"behavioral_baseline_concentration", "behavioral_class_asymmetry", "behavioral_coverage"} <= checks
        coverage = next(f for f in report.findings if f.check == "behavioral_coverage")
        assert coverage.details["n_candidates_tested"] == 2 * 49  # 2 colors x 7x7 half-stride grid (size 2 on 8x8)
        assert coverage.details["n_asymmetry_tests"] == 2
        assert report.metadata["input_shape"] == list(SHAPE)

    def test_smooth_clean_model_has_no_medium_plus(self):
        """False-positive control: a spatially smooth, symmetric model
        is exactly the null both tests assume."""
        report = _probe(_clean_forward_fn)
        assert report.passed, [f.to_dict() for f in report.findings if not f.passed]

    def test_point_backdoor_found_by_island_test(self):
        """Blind: the grid doesn't know where the trigger is."""
        report = _probe(_point_backdoor_forward_fn)
        islands = _medium_plus(report, "behavioral_trigger_island")
        assert any(f.details["position"] == [0, 0] and f.details["forced_class"] == 3 for f in islands), [
            f.details for f in islands
        ]
        assert report.passed is False

    def test_region_backdoor_found_by_asymmetry_not_island(self):
        """A hijacked region has no island -- only the class-reach
        asymmetry catches it, and it names the target class."""
        report = _probe(_region_backdoor_forward_fn)
        asym = _medium_plus(report, "behavioral_class_asymmetry")
        assert asym, "class-reach asymmetry missed a hijacked region"
        assert all(f.details["over_represented_class"] == 3 for f in asym)
        assert report.passed is False

    def test_fdr_is_per_family_and_never_below_raw_p(self):
        report = _probe(_point_backdoor_forward_fn)
        for f in report.findings:
            if f.check in ("behavioral_trigger_island", "behavioral_class_asymmetry") and f.details.get("fdr_p_value") is not None:
                assert f.details["fdr_p_value"] >= f.details["p_value"] - 1e-12
        assert report.metadata["fdr"] == "benjamini_hochberg_per_family"
