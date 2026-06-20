# Robustness & Safety Guard

**Pipeline:** `examples/pipelines/vlm_robustness_guard/`  
**Config:** `configs/vlm_robustness_guard.yaml`  
**P-level:** P1-02

## What it demonstrates

Three failure modes that matter in production VLM deployments, each with a concrete detection or mitigation:

| Failure mode | Component | Detection approach |
|---|---|---|
| Input distribution shift (batch) | `CosineDriftDetector` in P0-04 | KS test on similarity distributions |
| Input distribution shift (per-sample) | `KNNOODDetector` | kNN cosine similarity vs reference set |
| Ungrounded / hallucinated output | `HallucinationGuard` | `faithfulness_score` with pass/flag/reject |

Plus a **perturbation robustness sweep** that measures how a pixel-based chart reader degrades under six ImageNet-C-style perturbations at five severity levels — a proxy for asking "how fragile is the vision component to realistic input degradation?"

## Component 1: Perturbation robustness sweep

Six perturbation types, each with a `severity` parameter in [0, 1]:

```python
from production_vlm.robustness import apply_perturbation

perturbed = apply_perturbation(image, kind="gaussian_noise", severity=0.5, seed=42)
print(perturbed.params)  # {"sigma": 0.125}
```

The sweep table from the latest run:

| Perturbation | sev=0.0 | sev=0.2 | sev=0.4 | sev=0.6 | sev=0.8 | sev=1.0 |
|---|---|---|---|---|---|---|
| brightness | 100% | 100% | 100% | 100% | 100% | 100% |
| contrast | 100% | 100% | 100% | 100% | 100% | 100% |
| gaussian_blur | 100% | 33% | 33% | 33% | 33% | 33% |
| gaussian_noise | 100% | 100% | 13% | 33% | 33% | 33% |
| occlusion | 100% | 100% | 100% | 100% | 100% | 100% |
| rotation | 100% | 33% | 60% | 53% | 60% | 47% |

Brightness and contrast are fully robust because the reader uses adaptive background-color estimation (sampling actual image corners rather than a fixed threshold). Blur and rotation genuinely destroy the pixel signal — the correct, honest result.

## Component 2: OOD detection

```python
from production_vlm.robustness import KNNOODDetector

detector = KNNOODDetector(reference_embeddings, k=5, percentile=15.0)
result = detector.score(new_embedding)
print(result.is_ood, result.ood_score)
```

The `percentile` parameter sets the calibrated operating point:

| `percentile` | FP rate | TP rate |
|---|---|---|
| 1 | 0.0% | 2.5% |
| 5 | 2.5% | 17.5% |
| **15** | **12.5%** | **100%** |
| 20 | 17.5% | 100% |

Default `percentile=15.0` — 100% TP at 12.5% FP on the synthetic style-shift scenario.

!!! note "Per-sample vs batch-level"
    Per-sample OOD detection is inherently harder than batch-level drift detection. The kNN similarity of a single embedding has high variance; a batch's mean similarity averages that noise away. Calibrate `percentile` against your own reference set and known-shift data rather than relying on these numbers for a different embedding space.

## Component 3: Hallucination guard

```python
from production_vlm.robustness import HallucinationGuard, GuardConfig

guard = HallucinationGuard(GuardConfig(pass_threshold=0.6, flag_threshold=0.3))
result = guard.check(prediction, reference, evidence_text)

if result.decision == "pass":
    return result.output_text      # original prediction, cleared
elif result.decision == "flag":
    return result.output_text      # original prediction, with a warning
else:  # "reject"
    return result.output_text      # safe fallback message
```

Latest benchmark: precision 1.0, recall 1.0 on the injected hallucination test set (30% injection rate, fabricated numbers 3× the chart maximum so they can't coincidentally match real evidence values).

## Run it

```bash
cv-playbook run-example vlm_robustness_guard
```

All three components run in ~6s on CPU.
