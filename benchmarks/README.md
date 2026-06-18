# benchmarks/

Reserved for cross-example, longer-running benchmark suites (e.g. a full ChartQA/DocVQA
evaluation run, a multi-checkpoint LoRA rank sweep, or a multi-hardware-target edge inference
comparison) that go beyond what each example's own `benchmark()` entry point covers.

Each example's lightweight, fast sensitivity sweep lives next to its own code instead (see
`cv-playbook benchmark <name>`, e.g. `examples/pipelines/embedding_drift_active_learning/run.py`'s
`benchmark()` function) since those are designed to run in seconds as part of normal
development, not as a separate heavyweight suite.

Nothing is checked in here yet -- see `ROADMAP.md` (P1-03: Advanced Evaluation &
Benchmarking Harness) for what's planned.
