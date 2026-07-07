"""Evaluation metrics for chart/document VLM outputs.

Implements lightweight, dependency-minimal proxies for the metrics
frontier VLM evaluation harnesses use in 2026: grounding (does the
answer correspond to something actually present in the image/data),
faithfulness (RAGAS-style — is the answer supported by the source,
adapted here for vision rather than retrieved text), and numeric
accuracy (charts/documents are dominated by numeric answers, so exact
text-match is the wrong metric).

These are intentionally simple, auditable implementations rather than
black-box scorers, so users can inspect exactly what is being
measured and swap in stronger components (e.g. an auxiliary VLM
judge) where needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?")


def _extract_numbers(text: str) -> list[float]:
    out = []
    for match in _NUMBER_RE.findall(text):
        cleaned = match.replace(",", "").rstrip("%")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


@dataclass
class NumericAccuracyResult:
    score: float
    matched: int
    total_reference: int
    tolerance: float


def numeric_accuracy(prediction: str, reference: str, tolerance: float = 0.02) -> NumericAccuracyResult:
    """Fraction of reference numbers recoverable from the prediction within relative tolerance.

    Chart/document QA answers are usually numeric (e.g. "$4.2M",
    "37%"). Exact string match is too strict (formatting differences)
    and BLEU/ROUGE are not meaningful for numbers, so this extracts
    numeric tokens from both strings and matches with a relative
    tolerance (default 2%), which is the convention used in chart-VQA
    benchmarks such as ChartQA-style numeric evaluation.
    """
    ref_numbers = _extract_numbers(reference)
    pred_numbers = _extract_numbers(prediction)

    if not ref_numbers:
        return NumericAccuracyResult(
            score=1.0 if not pred_numbers else 0.0, matched=0, total_reference=0, tolerance=tolerance
        )

    remaining_pred = list(pred_numbers)
    matched = 0
    for ref_val in ref_numbers:
        for i, pred_val in enumerate(remaining_pred):
            denom = max(abs(ref_val), 1e-9)
            if abs(pred_val - ref_val) / denom <= tolerance:
                matched += 1
                remaining_pred.pop(i)
                break

    score = matched / len(ref_numbers)
    return NumericAccuracyResult(score=score, matched=matched, total_reference=len(ref_numbers), tolerance=tolerance)


@dataclass
class GroundingResult:
    score: float
    grounded_terms: int
    total_terms: int


def grounding_score(prediction: str, evidence_text: str, min_token_len: int = 3) -> GroundingResult:
    """Fraction of content words in the prediction that also appear in the source evidence.

    A coarse but transparent proxy for "is this answer actually
    referencing things present in the chart/document," used as a
    cheap pre-filter before any LLM-judge-based grounding check. Stop
    words and very short tokens are excluded to avoid inflating the
    score with function words.
    """
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "but",
        "not",
        "what",
        "which",
        "value",
        "data",
        "chart",
        "shows",
        "show",
        "according",
    }

    def tokens(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return {w for w in words if len(w) >= min_token_len and w not in stop_words}

    pred_tokens = tokens(prediction)
    evidence_tokens = tokens(evidence_text)

    if not pred_tokens:
        return GroundingResult(score=0.0, grounded_terms=0, total_terms=0)

    grounded = pred_tokens & evidence_tokens
    return GroundingResult(
        score=len(grounded) / len(pred_tokens), grounded_terms=len(grounded), total_terms=len(pred_tokens)
    )


@dataclass
class FaithfulnessResult:
    score: float
    numeric: NumericAccuracyResult
    grounding: GroundingResult


def faithfulness_score(
    prediction: str,
    reference: str,
    evidence_text: str | None = None,
    weights: tuple[float, float] = (0.6, 0.4),
) -> FaithfulnessResult:
    """Composite faithfulness score combining numeric accuracy and grounding.

    Inspired by RAGAS-style faithfulness (is the generated answer
    supported by the source) but adapted for vision/chart QA, where
    "the source" is the chart's underlying data/evidence text rather
    than retrieved passages. Numeric accuracy is weighted higher by
    default since chart/document answers are predominantly numeric.
    """
    numeric = numeric_accuracy(prediction, reference)
    grounding = grounding_score(prediction, evidence_text if evidence_text is not None else reference)
    w_num, w_ground = weights
    composite = w_num * numeric.score + w_ground * grounding.score
    return FaithfulnessResult(score=composite, numeric=numeric, grounding=grounding)
