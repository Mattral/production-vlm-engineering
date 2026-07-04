"""Hallucination/grounding guard: turns eval metrics into an actionable accept/flag/reject decision.

`production_vlm.eval.grounding_score` and `faithfulness_score` already
measure whether a generated answer is supported by its source
evidence. This module is the thin policy layer on top: given those
scores and configurable thresholds, decide what a serving system
should actually *do* -- pass the answer through, flag it for human
review, or refuse to return it and fall back to a safe default. This
is the "wrap VLM inference with guard-style checks" integration
pattern from the original transformation plan, made concrete.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from production_vlm.eval import FaithfulnessResult, faithfulness_score


class GuardDecision(str, Enum):
    PASS = "pass"
    FLAG = "flag"
    REJECT = "reject"


@dataclass
class GuardConfig:
    """Thresholds on `faithfulness_score`'s composite score for each decision tier.

    Default thresholds (0.6 / 0.3) are reasonable starting points, not
    universal constants -- calibrate against your own labeled
    hallucination examples (see `benchmark()` in
    `examples/pipelines/vlm_robustness_guard/run.py` for the
    methodology) before relying on them in production.
    """

    pass_threshold: float = 0.6
    flag_threshold: float = 0.3
    fallback_message: str = "I'm not confident enough in this answer to provide it -- please verify against the source."


@dataclass
class GuardResult:
    decision: GuardDecision
    faithfulness: FaithfulnessResult
    output_text: str
    reasoning: str


class HallucinationGuard:
    """Cross-checks a generated answer against visual evidence before it's returned to a user.

    Usage pattern (the integration this module is meant to enable):

        guard = HallucinationGuard()
        raw_answer = vlm.generate(image, question)          # your VLM call
        result = guard.check(raw_answer, reference_answer, evidence_text)
        return result.output_text                            # safe to return regardless of decision
    """

    def __init__(self, config: GuardConfig | None = None) -> None:
        self.config = config or GuardConfig()

    def check(self, prediction: str, reference: str, evidence_text: str) -> GuardResult:
        faithfulness = faithfulness_score(prediction, reference, evidence_text)
        score = faithfulness.score

        if score >= self.config.pass_threshold:
            return GuardResult(
                decision=GuardDecision.PASS,
                faithfulness=faithfulness,
                output_text=prediction,
                reasoning=f"faithfulness={score:.3f} >= pass_threshold={self.config.pass_threshold}",
            )
        elif score >= self.config.flag_threshold:
            return GuardResult(
                decision=GuardDecision.FLAG,
                faithfulness=faithfulness,
                output_text=prediction,
                reasoning=(
                    f"faithfulness={score:.3f} between flag_threshold={self.config.flag_threshold} "
                    f"and pass_threshold={self.config.pass_threshold} -- returned but flagged for review"
                ),
            )
        else:
            return GuardResult(
                decision=GuardDecision.REJECT,
                faithfulness=faithfulness,
                output_text=self.config.fallback_message,
                reasoning=f"faithfulness={score:.3f} < flag_threshold={self.config.flag_threshold}",
            )

    def check_batch(
        self, predictions: list[str], references: list[str], evidence_texts: list[str]
    ) -> list[GuardResult]:
        if not (len(predictions) == len(references) == len(evidence_texts)):
            raise ValueError("predictions, references, and evidence_texts must have the same length")
        return [self.check(p, r, e) for p, r, e in zip(predictions, references, evidence_texts)]
