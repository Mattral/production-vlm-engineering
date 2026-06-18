"""Unit tests for cv_playbook.utils.vision_encoder.SyntheticEmbeddingProxy."""

from __future__ import annotations

import numpy as np

from cv_playbook.utils.synthetic_charts import generate_synthetic_chart
from cv_playbook.utils.vision_encoder import SyntheticEmbeddingProxy


class TestSyntheticEmbeddingProxy:
    def test_output_shape(self):
        encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0)
        charts = [generate_synthetic_chart(seed=i, render_image=False) for i in range(10)]
        embeddings = encoder.encode_charts(charts, style_shift_flags=[False] * 10)
        assert embeddings.shape == (10, 64)

    def test_deterministic_given_same_seed_and_chart(self):
        encoder1 = SyntheticEmbeddingProxy(embedding_dim=32, seed=42)
        encoder2 = SyntheticEmbeddingProxy(embedding_dim=32, seed=42)
        chart = generate_synthetic_chart(seed=1, render_image=False)
        e1 = encoder1.encode_charts([chart], style_shift_flags=[False])
        e2 = encoder2.encode_charts([chart], style_shift_flags=[False])
        np.testing.assert_array_equal(e1, e2)

    def test_style_shift_moves_embedding_away_from_normal_centroid(self):
        encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=12.0)
        normal_charts = [generate_synthetic_chart(seed=i, render_image=False) for i in range(60)]
        normal_emb = encoder.encode_charts(normal_charts, style_shift_flags=[False] * 60)

        shifted_charts = [generate_synthetic_chart(seed=1000 + i, style_shift=True, render_image=False) for i in range(60)]
        shifted_emb = encoder.encode_charts(shifted_charts, style_shift_flags=[True] * 60)

        centroid = normal_emb.mean(axis=0)
        normal_centroid_dist = np.linalg.norm(normal_emb - centroid, axis=1).mean()
        shifted_centroid_dist = np.linalg.norm(shifted_emb - centroid, axis=1).mean()

        assert shifted_centroid_dist > normal_centroid_dist * 1.2

    def test_higher_shift_magnitude_increases_separation(self):
        charts_normal = [generate_synthetic_chart(seed=i, render_image=False) for i in range(30)]
        charts_shifted = [generate_synthetic_chart(seed=2000 + i, style_shift=True, render_image=False) for i in range(30)]

        def centroid_distance(magnitude: float) -> float:
            encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=magnitude)
            normal_emb = encoder.encode_charts(charts_normal, style_shift_flags=[False] * 30)
            shifted_emb = encoder.encode_charts(charts_shifted, style_shift_flags=[True] * 30)
            return float(np.linalg.norm(normal_emb.mean(0) - shifted_emb.mean(0)))

        low = centroid_distance(2.0)
        high = centroid_distance(20.0)
        assert high > low

    def test_encode_protocol_method_handles_raw_images(self):
        encoder = SyntheticEmbeddingProxy(embedding_dim=16, seed=0)
        chart = generate_synthetic_chart(seed=1, render_image=True)
        embeddings = encoder.encode([chart.image, chart.image])
        assert embeddings.shape == (2, 16)
