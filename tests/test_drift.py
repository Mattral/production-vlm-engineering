"""Unit tests for cv_playbook.drift: CosineDriftDetector, EWMADriftDetector, active learning selection."""

from __future__ import annotations

import numpy as np
import pytest

from cv_playbook.drift import (
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
        detector = EWMADriftDetector(lam=0.3, n_sigma=3.0, warmup=3, baseline_n=5)
        rng = np.random.default_rng(0)
        stable = 0.8 + rng.normal(0, 0.005, size=15)
        flags = [detector.update(v).is_drift for v in stable]
        # Allow the very first couple of points (before baseline_std is frozen) to be ignored;
        # no flag should fire once warmed up on a genuinely stable signal.
        assert not any(flags[5:])

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
        detector = CosineDriftDetector(reference_embeddings)
        batch = reference_embeddings[:50].copy()
        # Make sample 0 an obvious outlier far from everything else.
        batch[0] = batch[0] + 100.0
        result = detector.score_batch(batch)
        selected = select_for_active_learning([result], batch, top_k=1)
        assert 0 in selected
