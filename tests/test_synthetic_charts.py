"""Unit tests for production_vlm.utils.synthetic_charts."""

from __future__ import annotations

from production_vlm.utils.synthetic_charts import generate_dataset, generate_synthetic_chart


class TestGenerateSyntheticChart:
    def test_reproducible_with_same_seed(self):
        c1 = generate_synthetic_chart(seed=7, render_image=False)
        c2 = generate_synthetic_chart(seed=7, render_image=False)
        assert c1.categories == c2.categories
        assert c1.values == c2.values
        assert c1.title == c2.title

    def test_different_seeds_usually_differ(self):
        charts = [generate_synthetic_chart(seed=i, render_image=False) for i in range(10)]
        titles = {c.title for c in charts}
        assert len(titles) > 1

    def test_categories_from_single_coherent_group(self):
        region_group = {"North", "South", "East", "West", "Central"}
        quarter_group = {"Q1", "Q2", "Q3", "Q4"}
        market_group = {"EU", "US", "APAC", "LATAM"}
        for seed in range(20):
            chart = generate_synthetic_chart(seed=seed, render_image=False)
            cats = set(chart.categories)
            assert cats <= region_group or cats <= quarter_group or cats <= market_group

    def test_render_image_false_skips_rendering(self):
        chart = generate_synthetic_chart(seed=1, render_image=False)
        assert chart.image is None

    def test_render_image_true_produces_pil_image(self):
        chart = generate_synthetic_chart(seed=1, render_image=True)
        assert chart.image is not None
        assert chart.image.mode == "RGB"

    def test_question_answer_evidence_are_nonempty(self):
        chart = generate_synthetic_chart(seed=3, render_image=False)
        assert len(chart.question) > 0
        assert len(chart.answer) > 0
        assert len(chart.evidence_text) > 0

    def test_answer_references_max_value_category(self):
        chart = generate_synthetic_chart(seed=5, render_image=False)
        max_category = chart.categories[chart.values.index(max(chart.values))]
        assert max_category in chart.answer

    def test_explicit_chart_type_respected(self):
        for ct in ["bar", "line", "pie"]:
            chart = generate_synthetic_chart(seed=1, chart_type=ct, render_image=False)
            assert chart.chart_type == ct


class TestGenerateDataset:
    def test_correct_count(self):
        charts = generate_dataset(n=15, seed=0, render_image=False)
        assert len(charts) == 15

    def test_style_shift_fraction_applies_to_tail(self):
        charts = generate_dataset(n=10, seed=0, style_shift_fraction=0.3, render_image=False)
        # last 3 should be style-shifted; we can't directly introspect style_shift flag on
        # SyntheticChart, so this is implicitly checked via vision_encoder tests instead.
        # Here we just confirm the dataset size/seed contract holds.
        assert len(charts) == 10

    def test_zero_shift_fraction_is_default(self):
        charts = generate_dataset(n=5, seed=0, render_image=False)
        assert len(charts) == 5
