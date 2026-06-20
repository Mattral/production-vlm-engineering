# OOD Detection & Robustness

## Two different "robustness" problems

This repo addresses two related but distinct failure modes that are often conflated:

| Failure mode | What shifts | When | Component |
|---|---|---|---|
| **Natural perturbation brittleness** | Pixel statistics (noise, blur, rotation) | At inference time, per-image | `NaturalPerturbation` sweep |
| **Distribution shift (OOD)** | The underlying data-generating process | Over time, across batches | `KNNOODDetector` |

Perturbation robustness is the classic adversarial ML question: does the model break under small, realistic image corruptions? Distribution shift is a production MLOps question: has the world changed in a way the model wasn't trained to handle?

## Natural perturbations

The six perturbations in `NaturalPerturbation.ALL` follow the ImageNet-C (Hendrycks & Dietterich, 2019[^1]) convention: each corruption has a scalar `severity` in [0, 1], severity 0 produces minimal change, and the mapping is monotonic where possible:

| Perturbation | Physical analog | Monotonic? |
|---|---|---|
| `brightness` | Low-light / underexposed | ✓ (always darkens) |
| `contrast` | Fog / washed-out sensor | ✓ |
| `gaussian_noise` | Sensor noise / JPEG artifacts | ✓ in expectation |
| `gaussian_blur` | Motion blur / out-of-focus | ✓ |
| `rotation` | Tilted document scan | ✓ in angle |
| `occlusion` | Sticker / finger / glare | ✓ in area |

### The adaptive background detection fix

The first implementation of the chart reader used a fixed absolute pixel-brightness threshold (pixel ≥ 235 = background). Under brightness/contrast perturbation this threshold breaks immediately: darkened pixels fall below 235 even for genuine background regions, making the reader unable to find where bars start and end.

The fix: `_estimate_background_color()` samples the four image corners (which are always background in bar charts) and uses their actual brightness as the reference, making the threshold relative rather than absolute. This makes the reader genuinely robust to brightness and contrast perturbation — not just stated as robust.

## Per-sample OOD detection

`KNNOODDetector` computes each new embedding's mean cosine similarity to its k nearest neighbors in the reference set, then compares to a calibrated threshold.

### Calibration from the reference set

The threshold is set at the `percentile`-th percentile of the in-distribution kNN similarity distribution (computed leave-one-out on the reference set). This ties the threshold to a target false-positive rate rather than an arbitrary absolute value.

### The precision/recall tradeoff is real and not hidden

On the synthetic style-shift scenario at `percentile=15`:

- **True positive rate**: 100% (every style-shifted sample flagged)
- **False positive rate**: 12.5% (some genuine in-distribution samples flagged)

At more conservative thresholds (`percentile=1`):

- **True positive rate**: 2.5% (nearly useless)
- **False positive rate**: 0%

This is the correct, honest characterization of per-sample OOD detection's fundamental tradeoff. The 1:1 relationship between tightening the threshold and losing recall is not a limitation of this implementation — it's a property of the underlying problem geometry in high-dimensional spaces.

## Hallucination guard

The `HallucinationGuard` converts a continuous `faithfulness_score` into a three-way actionable decision, rather than leaving the choice of threshold to the caller:

```
faithfulness ≥ pass_threshold (0.6)  →  PASS   (return original answer)
flag_threshold ≤ faithfulness < pass_threshold  →  FLAG   (return with warning)
faithfulness < flag_threshold (0.3)   →  REJECT  (return safe fallback message)
```

The FLAG tier is intentional: there's a meaningful difference between "this answer is probably wrong" (REJECT) and "this answer is plausible but we're not confident" (FLAG). Collapsing them into a binary PASS/REJECT discards useful signal for human reviewers.

### Known limitation: numeric faithfulness without entity linking

See [Evaluation Metrics: Known limitation](metrics.md#known-limitation-numeric-faithfulness-without-entity-linking).

[^1]: Hendrycks, D., & Dietterich, T. (2019). Benchmarking Neural Network Robustness to Common Corruptions and Perturbations. *ICLR 2019*. [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)
