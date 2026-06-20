---
name: Wrong or stale result
about: A benchmark number, metric, or claim seems incorrect or outdated
title: "[RESULT] "
labels: correctness
assignees: ''
---

## Which result

Where did you see the claim that seems wrong?

- [ ] README benchmark table
- [ ] `benchmarks/reports/benchmark_report.md`
- [ ] A `results.json` output
- [ ] A docstring or inline comment
- [ ] A concept page in the docs
- [ ] Other: ___

## What seems wrong

What's the claim, and what do you think the correct value is?

## Your reproduction

```bash
# How you ran the example or benchmark
```

```
# Your results.json or console output
```

## Context

- Was `ran_with_real_ml_stack` true or false?
- What checkpoint did you use (if different from the default)?
- What's your hardware?

This is especially valuable for numbers marked ⚠️ (CPU smoke-test) being reported as if they were real GPU benchmarks, or for checkpoint-pinned results that have become stale.
