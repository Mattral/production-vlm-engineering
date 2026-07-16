# Evaluation Metrics

## Why not BLEU or exact-match?

Chart and document VQA answers are predominantly numeric: "67.6 req/s", "$4.2M", "37%". Standard text metrics handle these poorly:

- **Exact-match** fails on formatting differences ("$4.2M" ≠ "4,200,000")
- **BLEU** averages n-gram overlap, which is not meaningful for numbers ("four point two" and "4.2" share no n-grams despite being the same value)
- **BERTScore** has better semantic capture but is expensive to run and requires a model

`production_vlm.eval` provides three lightweight, dependency-free, auditable metrics instead.

## `numeric_accuracy`

Extracts numeric tokens from prediction and reference using a regex that handles commas, decimals, and percent signs, then matches them pairwise within a relative tolerance:

$$\text{match}(a, b) = \frac{|a - b|}{\max(|a|, \epsilon)} \leq \tau$$

Default tolerance $\tau = 0.02$ (2%), matching ChartQA-style numeric evaluation conventions. A greedy matching prevents the same predicted token from satisfying two reference tokens.

```python
from production_vlm.eval import numeric_accuracy

result = numeric_accuracy("Revenue was $4.21M", "Revenue was $4.2M", tolerance=0.02)
# NumericAccuracyResult(score=1.0, matched=1, total_reference=1, tolerance=0.02)
```

## `grounding_score`

Checks what fraction of content words in the prediction appear in the source evidence text. Stop words and very short tokens are excluded. This catches fluent but ungrounded answers — a model that generates plausible-sounding text with no reference to the actual chart data.

```python
from production_vlm.eval import grounding_score

result = grounding_score(
    prediction="South region had throughput of 67.6",
    evidence_text="South: 67.6 req/s; North: 50.0 req/s",
)
# GroundingResult(score=1.0, grounded_terms=3, total_terms=3)
```

## `faithfulness_score`

A weighted combination of the two above, inspired by RAGAS (Es et al., 2023[^1]) but adapted from retrieved-text to chart/image evidence. The default weighting is 60% numeric, 40% grounding, reflecting that chart QA is predominantly numeric:

$$\text{faithfulness} = w_{\text{num}} \cdot \text{numeric\_accuracy} + w_{\text{ground}} \cdot \text{grounding\_score}$$

```python
from production_vlm.eval import faithfulness_score

result = faithfulness_score(
    prediction="South has throughput of 67.6 req/s, which is highest.",
    reference="South has throughput of 67.6 req/s, which is the highest.",
    evidence_text="South: 67.6; North: 50.0",
)
# FaithfulnessResult(score=0.80, numeric=..., grounding=...)
```

## Known limitation: numeric faithfulness without entity linking

`numeric_accuracy` checks whether a predicted number appears in the reference, without verifying which entity it's attributed to. A prediction "LATAM is 77.8" scores 1.0 on numeric accuracy if "77.8" appears anywhere in the evidence, even if that value belongs to a different category. This was discovered as a real failure mode during the hallucination guard evaluation — the guard injection was initially using a value from another bar, which happened to match the evidence numerically. The fix was to use fabricated numbers (3× the maximum) that cannot appear in the evidence by construction. See the `vlm_robustness_guard` example and its `run_guard_evaluation()` function for the full explanation.

## Why this matters for 2027

Static, one-time benchmarks (run a model against a fixed test set, publish a leaderboard score) are already a poor proxy for how a VLM behaves once it's deployed against a live, drifting distribution of real user inputs — and as more VLM output feeds directly into financial reports, scientific data extraction, and other domains where a wrong number is not just an inconvenience but a liability, the bar for evaluation moves from "does it read charts well on average" to "can we cite exactly why we trust this specific answer." That requires metrics that are auditable by a human reviewing a single case, not just aggregatable across a large sample — which is precisely why this repo favors transparent, decomposable scores (`numeric_accuracy`, `grounding_score`) over an opaque single "does-it-look-right" LLM-judge call: an auditor can point at the exact number that didn't match, not just a score that went down. Expect continuous, in-production evaluation against exactly this kind of decomposable, cite-able metric to become the norm for any VLM deployment with real financial or regulatory exposure.

[^1]: Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv:2309.15217*. [Link](https://arxiv.org/abs/2309.15217)
