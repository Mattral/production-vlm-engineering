# vlm_robustness_guard

Four safety and robustness components for production VLM systems, run as a single evaluation pipeline.

## What it demonstrates

| Component | Failure mode addressed | Metric |
|---|---|---|
| Perturbation robustness sweep | Vision encoder brittleness to natural corruptions | Accuracy vs severity table |
| Adversarial robustness (PGD proxy) | Embedding-space manipulation under a bounded pixel budget | Centroid cosine shift, OOD catch rate |
| `KNNOODDetector` | Anomalous inputs reaching the model | FP rate / TP rate at calibrated threshold |
| `HallucinationGuard` | Ungrounded/fabricated generated text | Precision / recall on injected hallucinations |

## Run it

```bash
production-vlm run-example vlm_robustness_guard
```

All four components run end-to-end in under 10 seconds on CPU.

## Component 1: Perturbation robustness sweep

Applies six ImageNet-C-style perturbations (brightness, contrast, Gaussian noise, Gaussian blur, rotation, occlusion) at five severity levels to a set of synthetic bar charts, and measures what fraction the pixel-based chart reader correctly identifies.

Key design decisions:
- **Adaptive background detection**: the reader samples the actual image corners to estimate background colour rather than using a fixed threshold (255 ≥ pixel ≥ 235 was the original broken design, caught here under contrast reduction).
- **Ground-truth plot bbox**: coordinates come from matplotlib's `ax.get_window_extent()` rather than heuristic spine detection, so perturbation failures reflect genuine chart-reading difficulty, not preprocessing brittleness.
- **Near-tie filtering**: charts where the top two values differ by < 1% of the maximum are excluded from the sweep — they cannot be reliably distinguished at our rendered pixel resolution, and including them would create failures unrelated to the perturbation being tested.

Expected results: brightness/contrast robust at all severities (adaptive bg), blur/rotation genuinely brittle at high severity (pixel structure destroyed), occlusion graceful degradation.

## Component 2: Adversarial robustness (PGD proxy)

Measures how far a bounded pixel perturbation (L∞ budget, default 8/255 matching Madry et al.'s standard attack budget) can shift an embedding away from its reference centroid, and whether the shifted embedding gets caught by the OOD detector.

The CPU-runnable path is a numpy transfer-attack proxy (adds noise in a fixed direction within the epsilon ball, not a real gradient computation) — it answers "how much can a bounded pixel change move this embedding?" without needing a real differentiable model. The real gradient-based PGD (`production_vlm.robustness.perturbations.pgd_attack`) requires `pip install -e ".[ml]"` and a CUDA device; wire in your VLM's vision tower as the `model` argument.

This connects directly to the OOD guard: if an attacker can craft inputs that shift embeddings close to the in-distribution centroid, they can evade OOD-based defenses entirely. The `ood_detector_catch_rate` metric in the results tells you whether your calibrated threshold provides adequate margin against this specific attack budget.

## Component 3: Per-sample OOD detection

`KNNOODDetector` calibrates its flag threshold from the reference set's own leave-one-out kNN similarity distribution, rather than an arbitrary fixed cosine cutoff.

At `percentile=15.0` (the empirically validated default):
- **FP rate ~10–12%** on in-distribution holdout
- **TP rate ~97–100%** on style-shifted (OOD) inputs

Per-sample OOD detection is inherently harder than batch-level drift detection — one sample's kNN similarity has higher variance than a batch mean. See `production_vlm.robustness.KNNOODDetector` docstring for the full precision/recall tradeoff table.

## Component 4: Hallucination guard

`HallucinationGuard` wraps `faithfulness_score` in a three-tier pass/flag/reject decision:

```python
from production_vlm.robustness import HallucinationGuard, GuardConfig

guard = HallucinationGuard(GuardConfig(pass_threshold=0.6, flag_threshold=0.3))
result = guard.check(prediction, reference, evidence_text)
# result.decision: "pass" | "flag" | "reject"
# result.output_text: original answer (pass/flag) or safe fallback (reject)
```

The FLAG tier is intentional — it's meaningful to distinguish "wrong" (reject) from "uncertain" (flag), and collapsing them to binary discards signal for human reviewers.

**Known limitation**: `faithfulness_score` checks numbers without entity linking. A prediction "LATAM: 77.8" can score high if 77.8 appears anywhere in the evidence, even if it belongs to a different category (US: 77.8). The injection in this example uses numbers 3× the true maximum to avoid this — documented in `run_guard_evaluation()` in `run.py`.

## Files

- `run.py` — four components, `main()` entry point
- `../../../configs/vlm_robustness_guard.yaml` — thresholds, sample counts, injection rate
- `../../../src/production_vlm/robustness/` — the underlying modules (perturbations, ood, guard, chart_reader)
