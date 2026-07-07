"""Unit tests for production_vlm.robustness: perturbations, OOD, guard, chart_reader."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from production_vlm.robustness import (
    GuardConfig,
    GuardDecision,
    HallucinationGuard,
    KNNOODDetector,
    NaturalPerturbation,
    apply_perturbation,
)
from production_vlm.robustness.chart_reader import read_tallest_bar
from production_vlm.utils.synthetic_charts import generate_synthetic_chart

# ---------------------------------------------------------------------------
# Perturbation tests
# ---------------------------------------------------------------------------


class TestNaturalPerturbations:
    @pytest.fixture
    def sample_image(self):
        chart = generate_synthetic_chart(seed=1, chart_type="bar", render_image=True)
        return chart.image

    def test_all_kinds_registered(self):
        assert set(NaturalPerturbation.ALL.keys()) == {
            "brightness",
            "contrast",
            "gaussian_noise",
            "gaussian_blur",
            "rotation",
            "occlusion",
        }

    def test_severity_zero_minimal_change(self, sample_image):
        for kind in NaturalPerturbation.ALL:
            result = apply_perturbation(sample_image, kind, severity=0.0)
            arr_orig = np.asarray(sample_image, dtype=np.float64)
            arr_pert = np.asarray(result.perturbed_image, dtype=np.float64)
            # Severity 0 should produce negligible change (some methods like
            # rotation at 0 degrees still anti-alias slightly, so allow 1.0 mean diff)
            assert np.abs(arr_orig - arr_pert).mean() <= 1.0, f"{kind} changed too much at severity=0"

    def test_severity_one_changes_image(self, sample_image):
        for kind in NaturalPerturbation.ALL:
            result = apply_perturbation(sample_image, kind, severity=1.0)
            arr_orig = np.asarray(sample_image, dtype=np.float64)
            arr_pert = np.asarray(result.perturbed_image, dtype=np.float64)
            assert np.abs(arr_orig - arr_pert).mean() > 1.0, f"{kind} had no effect at severity=1"

    def test_brightness_monotonically_darkens(self, sample_image):
        means = []
        for sev in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = apply_perturbation(sample_image, "brightness", sev)
            means.append(np.asarray(r.perturbed_image, dtype=np.float64).mean())
        # Each step should be <= previous (darker or same)
        for i in range(1, len(means)):
            assert means[i] <= means[i - 1] + 0.5, f"brightness not monotonically darkening at step {i}"

    def test_unknown_kind_raises(self, sample_image):
        with pytest.raises(ValueError, match="Unknown perturbation kind"):
            apply_perturbation(sample_image, "nonexistent_kind", 0.5)

    def test_output_is_pil_rgb(self, sample_image):
        result = apply_perturbation(sample_image, "gaussian_noise", 0.5, seed=42)
        assert isinstance(result.perturbed_image, Image.Image)
        assert result.perturbed_image.mode == "RGB"

    def test_result_contains_metadata(self, sample_image):
        result = apply_perturbation(sample_image, "gaussian_blur", 0.7)
        assert result.name == "gaussian_blur"
        assert result.severity == pytest.approx(0.7)
        assert "radius" in result.params


# ---------------------------------------------------------------------------
# OOD detection tests
# ---------------------------------------------------------------------------


class TestKNNOODDetector:
    @pytest.fixture
    def reference_embeddings(self):
        from production_vlm.utils.synthetic_charts import generate_synthetic_chart
        from production_vlm.utils.vision_encoder import SyntheticEmbeddingProxy

        encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=12.0)
        charts = [generate_synthetic_chart(seed=i, render_image=False) for i in range(150)]
        return encoder.encode_charts(charts, style_shift_flags=[False] * 150)

    def test_rejects_invalid_k(self, reference_embeddings):
        with pytest.raises(ValueError):
            KNNOODDetector(reference_embeddings, k=0)

    def test_rejects_non_2d_embeddings(self):
        with pytest.raises(ValueError):
            KNNOODDetector(np.zeros(32))

    def test_low_fp_on_in_distribution(self, reference_embeddings):
        from production_vlm.utils.synthetic_charts import generate_synthetic_chart
        from production_vlm.utils.vision_encoder import SyntheticEmbeddingProxy

        encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=12.0)
        detector = KNNOODDetector(reference_embeddings, k=5, percentile=15.0)
        holdout = [generate_synthetic_chart(seed=2000 + i, render_image=False) for i in range(40)]
        holdout_emb = encoder.encode_charts(holdout, style_shift_flags=[False] * 40)
        fp_rate = sum(r.is_ood for r in detector.score_batch(holdout_emb)) / 40
        assert fp_rate <= 0.25

    def test_high_tp_on_shifted(self, reference_embeddings):
        from production_vlm.utils.synthetic_charts import generate_synthetic_chart
        from production_vlm.utils.vision_encoder import SyntheticEmbeddingProxy

        encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=12.0)
        detector = KNNOODDetector(reference_embeddings, k=5, percentile=15.0)
        shifted = [generate_synthetic_chart(seed=3000 + i, style_shift=True, render_image=False) for i in range(40)]
        shifted_emb = encoder.encode_charts(shifted, style_shift_flags=[True] * 40)
        tp_rate = sum(r.is_ood for r in detector.score_batch(shifted_emb)) / 40
        assert tp_rate >= 0.85

    def test_score_returns_ood_result_fields(self, reference_embeddings):
        from production_vlm.utils.synthetic_charts import generate_synthetic_chart
        from production_vlm.utils.vision_encoder import SyntheticEmbeddingProxy

        encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=12.0)
        detector = KNNOODDetector(reference_embeddings)
        chart = generate_synthetic_chart(seed=99, render_image=False)
        emb = encoder.encode_charts([chart], style_shift_flags=[False])[0]
        result = detector.score(emb)
        assert hasattr(result, "is_ood")
        assert hasattr(result, "ood_score")
        assert 0.0 <= result.ood_score <= 2.0


# ---------------------------------------------------------------------------
# Hallucination guard tests
# ---------------------------------------------------------------------------


class TestHallucinationGuard:
    def test_correct_answer_passes(self):
        guard = HallucinationGuard()
        result = guard.check(
            "South shows throughput of 67.6 req/s, making it the highest.",
            "South has throughput of 67.6 req/s, which is the highest.",
            "South: 67.6 req/s; North: 50.0 req/s",
        )
        assert result.decision == GuardDecision.PASS

    def test_hallucinated_answer_rejected(self):
        guard = HallucinationGuard()
        result = guard.check(
            "North has throughput of 999.9 req/s which is enormous.",
            "South has throughput of 67.6 req/s, which is the highest.",
            "South: 67.6 req/s; North: 50.0 req/s",
        )
        assert result.decision == GuardDecision.REJECT
        assert result.output_text == guard.config.fallback_message

    def test_fallback_message_returned_on_reject(self):
        cfg = GuardConfig(fallback_message="Cannot verify this answer.")
        guard = HallucinationGuard(cfg)
        result = guard.check("totally wrong answer with no grounding at all", "South: 67.6", "South: 67.6")
        if result.decision == GuardDecision.REJECT:
            assert result.output_text == "Cannot verify this answer."

    def test_pass_answer_preserves_original_text(self):
        guard = HallucinationGuard()
        original = "South shows throughput of 67.6 req/s, making it the highest."
        result = guard.check(original, "South has throughput of 67.6 req/s, which is the highest.", "South: 67.6 req/s")
        if result.decision == GuardDecision.PASS:
            assert result.output_text == original

    def test_check_batch_length_validation(self):
        guard = HallucinationGuard()
        with pytest.raises(ValueError):
            guard.check_batch(["a", "b"], ["ref"], ["ev"])

    def test_check_batch_returns_correct_length(self):
        guard = HallucinationGuard()
        results = guard.check_batch(
            ["South: 67.6", "something"], ["South: 67.6", "South: 67.6"], ["South: 67.6", "South: 67.6"]
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Chart reader tests
# ---------------------------------------------------------------------------


class TestChartReader:
    def test_100_pct_baseline_on_clean_bar_charts(self):
        correct = 0
        n = 20
        for i in range(n):
            chart = generate_synthetic_chart(seed=i, chart_type="bar", render_image=True)
            result = read_tallest_bar(
                chart.image,
                len(chart.categories),
                int(np.argmax(chart.values)),
                plot_bbox=chart.plot_bbox,
            )
            correct += result.correct
        assert correct == n, f"Expected 100% accuracy on clean charts, got {correct}/{n}"

    def test_degrades_under_heavy_occlusion(self):
        """Accuracy should drop under heavy occlusion (severity=1.0)."""
        correct_clean, correct_occluded = 0, 0
        n = 20
        for i in range(n):
            chart = generate_synthetic_chart(seed=i, chart_type="bar", render_image=True)
            true_max = int(np.argmax(chart.values))

            r_clean = read_tallest_bar(chart.image, len(chart.categories), true_max, plot_bbox=chart.plot_bbox)
            correct_clean += r_clean.correct

            pert = apply_perturbation(chart.image, "occlusion", severity=1.0, seed=i)
            r_occ = read_tallest_bar(pert.perturbed_image, len(chart.categories), true_max, plot_bbox=chart.plot_bbox)
            correct_occluded += r_occ.correct

        assert correct_clean == n
        assert correct_occluded < n

    def test_robust_to_brightness_perturbation(self):
        """Brightness should not degrade reading since bg detection is adaptive."""
        correct = 0
        n = 20
        for i in range(n):
            chart = generate_synthetic_chart(seed=i, chart_type="bar", render_image=True)
            pert = apply_perturbation(chart.image, "brightness", severity=0.8, seed=i)
            result = read_tallest_bar(
                pert.perturbed_image,
                len(chart.categories),
                int(np.argmax(chart.values)),
                plot_bbox=chart.plot_bbox,
            )
            correct += result.correct
        assert correct == n

    def test_result_contains_heights_for_each_bar(self):
        chart = generate_synthetic_chart(seed=5, chart_type="bar", render_image=True)
        result = read_tallest_bar(
            chart.image,
            len(chart.categories),
            int(np.argmax(chart.values)),
            plot_bbox=chart.plot_bbox,
        )
        assert len(result.bar_heights_px) == len(chart.categories)
        assert all(h >= 0 for h in result.bar_heights_px)
