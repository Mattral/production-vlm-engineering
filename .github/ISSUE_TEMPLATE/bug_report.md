---
name: Bug report
about: Something in the code is wrong or produces unexpected output
title: "[BUG] "
labels: bug
assignees: ''
---

## What example / module

Which example or module is affected?

- [ ] `vlm_chart_finetune`
- [ ] `embedding_drift_active_learning`
- [ ] `vlm_edge_inference`
- [ ] `vlm_robustness_guard`
- [ ] `production_vlm.drift`
- [ ] `production_vlm.eval`
- [ ] `production_vlm.robustness`
- [ ] `production_vlm.utils`
- [ ] Benchmark runner (`benchmarks/run_all.py`)
- [ ] Other: ___

## Environment

```
Python version:
OS:
GPU (if relevant): none / <model>
ML stack installed: yes (pip install -e ".[ml]") / no (CPU only)
ONNX stack installed: yes / no
```

## Reproduction

```bash
# Exact command that triggers the bug
```

## Expected vs actual

**Expected:** 

**Actual:**

## Console output / traceback

```
<paste here>
```

## `results.json` if available

```json
```

## Additional context

Note: if `ran_with_real_ml_stack` is `false` in your `results.json`, this is the CPU smoke-test path. Please confirm whether you observed the bug on the real ML stack or the CPU fallback.
