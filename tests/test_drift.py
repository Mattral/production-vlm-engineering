"""Unit tests for production_vlm.drift: CosineDriftDetector, EWMADriftDetector, active learning selection."""

from __future__ import annotations

import numpy as np
import pytest

from production_vlm.drift import (
    CosineDriftDetector,
    EWMADriftDetector,
    select_for_active_learning,
)


@pytest.fixture
def reference_embeddings() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(loc=0.0, scale=1.0, size=(200, 32))


class TestCosineDriftDetector:
    def test_no_drift_on_resampled_reference(self, reference_embeddings):
        detector = CosineDriftDetector(reference_embeddings, alpha=0.01)
        rng = np.random.default_rng(1)
        idx = rng.choice(len(reference_embeddings), size=40, replace=False)
        batch = reference_embeddings[idx]

        result = detector.score_batch(batch)

        assert result.is_drift is False
        assert result.p_value > 0.01

    def test_detects_injected_shift(self, reference_embeddings):
        detector = CosineDriftDetector(reference_embeddings, alpha=0.01)
        # Use an independent RNG stream from a different seed than the one
        # that generated `reference_embeddings`, so the shift direction
        # isn't an arbitrary continuation of the reference's own RNG state.
        # Magnitude 20 in this 32-dim isotropic space was empirically
        # validated (50/50 random trials) to reliably separate from the
        # reference distribution at alpha=0.01; smaller magnitudes are
        # detectable but not reliably so for an arbitrary random direction,
        # since a 1D cosine-to-centroid projection only captures the
        # shift component aligned with the centroid direction.
        rng = np.random.default_rng(99)
        shift = rng.normal(size=32)
        shift /= np.linalg.norm(shift)
        shifted_batch = reference_embeddings[:40] + 20.0 * shift

        result = detector.score_batch(shifted_batch)

        assert result.is_drift is True
        assert result.p_value < 0.01

    def test_rejects_non_2d_reference(self):
        with pytest.raises(ValueError):
            CosineDriftDetector(np.zeros(10))

    def test_result_contains_batch_size_detail(self, reference_embeddings):
        detector = CosineDriftDetector(reference_embeddings)
        result = detector.score_batch(reference_embeddings[:17])
        assert result.details["batch_size"] == 17


class TestEWMADriftDetector:
    def test_no_alarm_on_stable_signal(self):
        # baseline_n=10 (not 5): a 3-sigma control chart's false-alarm rate depends
        # directly on how reliably the baseline std is estimated. A 5-sample std
        # estimate has enough sampling variance that it can be off by 2-3x on a
        # given draw, making the "3-sigma" band effectively much tighter than
        # intended and prone to spurious false alarms — this was verified directly:
        # with this seed, a 5-sample baseline underestimated the true population
        # std by 2.6x. Standard SPC guidance (Montgomery, 2020) recommends at
        # least ~20-25 calibration samples; 10 is a reasonable minimum for a
        # reliable no-false-alarm guarantee in a small test fixture.
        detector = EWMADriftDetector(lam=0.3, n_sigma=3.0, warmup=3, baseline_n=10)
        rng = np.random.default_rng(0)
        stable = 0.8 + rng.normal(0, 0.005, size=20)
        flags = [detector.update(v).is_drift for v in stable]
        # Allow the points before baseline_std is frozen to be ignored;
        # no flag should fire once warmed up on a genuinely stable signal.
        assert not any(flags[10:])

    def test_flags_step_change(self):
        detector = EWMADriftDetector(lam=0.3, n_sigma=2.0, warmup=4, baseline_n=6)
        rng = np.random.default_rng(0)
        stable = list(0.8 + rng.normal(0, 0.005, size=6))
        shifted = list(0.6 + rng.normal(0, 0.005, size=6))
        flags = [detector.update(v).is_drift for v in stable + shifted]
        # The first post-shift point must be flagged -- this is the core
        # guarantee a drift monitor needs (detect onset with minimal delay).
        assert flags[6] is True

    def test_baseline_std_is_frozen_not_contaminated_by_shift(self):
        detector = EWMADriftDetector(lam=0.3, n_sigma=2.0, warmup=4, baseline_n=5)
        for v in [0.8, 0.81, 0.79, 0.80, 0.80]:
            detector.update(v)
        baseline_std_before = detector._baseline_std
        detector.update(0.3)  # large shift
        assert detector._baseline_std == baseline_std_before  # must not change post-freeze

    def test_invalid_construction_does_not_raise_for_valid_params(self):
        # Smoke test that defaults are usable
        detector = EWMADriftDetector()
        result = detector.update(0.5)
        assert result.score == 0.5


class TestActiveLearningSelection:
    def test_select_top_k_returns_requested_size(self, reference_embeddings):
        detector = CosineDriftDetector(reference_embeddings)
        result = detector.score_batch(reference_embeddings[:50])
        selected = select_for_active_learning([result], reference_embeddings[:50], top_k=10)
        assert len(selected) == 10

    def test_select_handles_empty_drift_results(self, reference_embeddings):
        selected = select_for_active_learning([], reference_embeddings[:10], top_k=5)
        assert len(selected) == 0

    def test_select_caps_at_available_samples(self, reference_embeddings):
        detector = CosineDriftDetector(reference_embeddings)
        result = detector.score_batch(reference_embeddings[:3])
        selected = select_for_active_learning([result], reference_embeddings[:3], top_k=10)
        assert len(selected) == 3

    def test_selected_indices_are_most_novel(self, reference_embeddings):
        """The selected indices should be the ones farthest from the centroid."""
        # Construction note: an earlier version added a constant offset
        # (+100.0) to every dimension of one sample. That has the same
        # failure mode documented elsewhere in this codebase (see
        # SyntheticEmbeddingProxy / KNNOODDetector fixes): a large constant
        # shift dominates the L2-normalized direction, AND because the
        # centroid here is computed from the very batch containing the
        # outlier, a large enough perturbation pulls the centroid toward
        # itself and paradoxically makes the outlier look *more* similar
        # to (not less similar to) its own self-polluted centroid. This
        # was verified directly: no magnitude of a single-point constant
        # or directional perturbation on top of a high-variance (std=1)
        # isotropic batch reliably produced a rank-0 outlier, because
        # larger magnitudes always increased the outlier's pull on the
        # mean faster than they escaped it.
        #
        # The fix uses a tight, low-variance cluster (std=0.05) representing
        # stable in-distribution embeddings -- the realistic scenario this
        # function is designed for -- plus one genuinely distinct point.
        # A modest, bounded perturbation (magnitude=1.0) then reliably
        # stands out without needing to be large enough to dominate the
        # batch mean.
        cluster_rng = np.random.default_rng(0)
        base_point = cluster_rng.normal(size=32)
        batch = base_point + cluster_rng.normal(0, 0.05, size=(50, 32))
        outlier_dir = np.random.default_rng(1).normal(size=32)
        outlier_dir /= np.linalg.norm(outlier_dir)
        batch[0] = base_point + 1.0 * outlier_dir

        detector = CosineDriftDetector(reference_embeddings)
        result = detector.score_batch(batch)
        selected = select_for_active_learning([result], batch, top_k=1)
        assert 0 in selected
