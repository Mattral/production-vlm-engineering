# Examples Overview

All four examples share the same CLI interface:

```bash
cv-playbook list-examples
cv-playbook run-example <name>
cv-playbook run-example <name> --config path/to/custom.yaml
cv-playbook benchmark <name>    # where implemented
```

## At a glance

| Example | P-level | Core technique | Key output metric | CPU time |
|---|---|---|---|---|
| [`vlm_chart_finetune`](vlm_chart_finetune.md) | P0-02 | LoRA (vision tower + LM) | Faithfulness Δ vs zero-shot | ~5s |
| [`embedding_drift_active_learning`](embedding_drift.md) | P0-04 | KS-test + EWMA SPC | Detection delay (batches) | <1s |
| [`vlm_edge_inference`](edge_inference.md) | P0-03 | ONNX + dynamic INT8 | Speedup vs fp32 | ~2s |
| [`vlm_robustness_guard`](robustness_guard.md) | P1-02 | Perturbation sweep, kNN OOD, faithfulness guard | Guard precision/recall | ~6s |

## Shared data

All examples use the same zero-download synthetic chart generator (`production_vlm.utils.synthetic_charts`) as their default dataset. This means:

- No external data dependency for the smoke-test paths
- Reproducible: same seed → same chart every time
- Swappable: replace with ChartQA, DocVQA, or your own data by swapping the loader

The synthetic generator produces bar, line, and pie charts with coherent category groups (regions, quarters, markets), ground-truth values, and a derived QA pair — enough for the evaluation harness to produce real, non-trivial numbers.

## How results.json works

Every example writes:

```json
{
  "config_name": "...",
  "ran_with_real_ml_stack": false,
  "headline_metric_1": ...,
  ...
}
```

The `ran_with_real_ml_stack` (or `ran_with_real_export_stack`) flag makes the fallback path unambiguous in downstream tooling, not just in console output. `benchmarks/run_all.py` uses this to mark ⚠️ in the generated Markdown report automatically.
