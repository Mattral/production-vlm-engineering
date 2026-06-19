"""Unit tests for production_vlm.eval: numeric_accuracy, grounding_score, faithfulness_score."""

from __future__ import annotations

from production_vlm.eval import faithfulness_score, grounding_score, numeric_accuracy


class TestNumericAccuracy:
    def test_exact_match(self):
        result = numeric_accuracy("The value is 42.5", "The value is 42.5")
        assert result.score == 1.0
        assert result.matched == 1

    def test_within_tolerance(self):
        result = numeric_accuracy("Revenue was $4.21M", "Revenue was $4.2M", tolerance=0.02)
        assert result.score == 1.0

    def test_outside_tolerance_fails(self):
        result = numeric_accuracy("Revenue was $10M", "Revenue was $4.2M", tolerance=0.02)
        assert result.score == 0.0

    def test_handles_commas_and_percent(self):
        result = numeric_accuracy("1,234 units, 45%", "1234 units, 45.0%")
        assert result.score == 1.0

    def test_partial_match_multiple_numbers(self):
        result = numeric_accuracy("North: 50, South: 999", "North: 50, South: 60")
        assert result.matched == 1
        assert result.total_reference == 2
        assert result.score == 0.5

    def test_no_reference_numbers_no_prediction_numbers_scores_one(self):
        result = numeric_accuracy("no numbers here", "also none here")
        assert result.score == 1.0

    def test_no_reference_numbers_but_prediction_has_some_scores_zero(self):
        result = numeric_accuracy("there are 5 of them", "qualitative answer only")
        assert result.score == 0.0

    def test_does_not_double_count_same_predicted_number(self):
        # Reference asks for two distinct values; prediction only has one
        # number that happens to match both -- should match at most once
        # per occurrence in prediction, not let one predicted token satisfy two refs.
        result = numeric_accuracy("50", "50 and 50")
        assert result.matched == 1
        assert result.total_reference == 2


class TestGroundingScore:
    def test_full_overlap(self):
        result = grounding_score("revenue increased sharply", "revenue increased sharply this quarter")
        assert result.score == 1.0

    def test_no_overlap(self):
        result = grounding_score("completely unrelated text", "revenue numbers here")
        assert result.score == 0.0

    def test_empty_prediction_scores_zero(self):
        result = grounding_score("", "some evidence text")
        assert result.score == 0.0

    def test_stop_words_excluded(self):
        result = grounding_score("the and for with", "revenue increased")
        # All tokens are stop words / too short, so pred_tokens is empty -> score 0
        assert result.score == 0.0


class TestFaithfulnessScore:
    def test_combines_numeric_and_grounding(self):
        result = faithfulness_score(
            prediction="South region had 67.6 throughput",
            reference="South has throughput of 67.6 req/s",
            evidence_text="South: 67.6 req/s; North: 50.0 req/s",
        )
        assert 0.0 <= result.score <= 1.0
        assert result.numeric.score > 0
        assert result.grounding.score > 0

    def test_weak_answer_scores_low(self):
        result = faithfulness_score(
            prediction="The value appears moderate",
            reference="South has throughput of 67.6 req/s, which is the highest.",
            evidence_text="South: 67.6 req/s",
        )
        assert result.score < 0.5

    def test_weights_sum_behavior(self):
        result = faithfulness_score("a b c", "a b c", "a b c", weights=(1.0, 0.0))
        # With weight entirely on numeric and no numbers anywhere, numeric scores 1.0 (vacuous match)
        assert result.score == result.numeric.score
