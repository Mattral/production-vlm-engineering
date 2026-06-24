# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `production_vlm.utils.observability` — structured JSONL event log (`ObservabilityLogger`,
  schema-versioned, thread-safe, immediate flush) and optional Prometheus metrics server
  (`PrometheusMetricsServer`, graceful no-op stub when `prometheus_client` not installed).
  Emits `drift_check`, `ood_check`, and `guard_check` events from the drift pipeline.
  Satisfies the P0-04 roadmap requirement "log metrics, optional Prometheus exposition."

- `production_vlm.utils.retraining` — `RetrainingTrigger` closes the drift → active-learning →
  retrain feedback loop. Maintains a priority queue of `QueuedSample` objects (indexed by
  novelty score), fires a configurable callback when `queue_threshold` accumulates, respects a
  `cooldown_s` between runs. `enqueue_batch()` fires multiple times if the threshold is crossed
  multiple times in one call (fixed bug: initial implementation coalesced all samples into one
  oversized batch). Satisfies P0-04 "trigger retraining simulation on drifted data."

- `vlm_video_temporal` example pipeline (P1-04) — three frame-sampling strategies (uniform,
  keyframe via L1 pixel diff, adaptive via highest-motion budget), temporal grounding metric
  extending `faithfulness_score` to multi-frame evidence, structured JSON answer schema,
  scene-change detection via `CosineDriftDetector`. Explicit `next_steps` in `results.json`.

- `_CHART_JSON_SCHEMA` + `_extract_structured_json()` + `_structured_extraction_accuracy()`
  in `vlm_chart_finetune` — structured JSON chart extraction evaluation with schema validity
  rate, numeric MAPE, and category coverage. Before/after: zero-shot 40% MAPE → fine-tuned 0%.
  Satisfies P0-02 roadmap requirement "structured output (JSON for chart values)."

- `configs/vlm_video_temporal.yaml` — video temporal example now config-driven (consistent
  with the four existing examples).

- `docs/examples/vlm_video_temporal.md` — full docs page for the video temporal example.

- Three GitHub issue templates (bug, feature/new example, wrong result) and a PR template
  enforcing the repo's correctness standards (requires running the example, not just reading
  the diff).

- `benchmarks/run_all.py` now covers all five examples, including `vlm_video_temporal`.

- CI: `vlm_robustness_guard` and `vlm_video_temporal` added to smoke-run job; benchmark
  runner exercised; docs build added; `scripts/verify_no_pytest.py` runs in CI.

- 56 checks in `scripts/verify_no_pytest.py` (up from 40), covering observability,
  retraining trigger, and structured JSON extraction.

### Fixed

- `RetrainingTrigger.enqueue_batch()` coalesced all samples into one oversized batch
  instead of firing once per threshold-worth of samples. Fixed by looping `_fire()` until
  the queue drops below threshold, and draining exactly `queue_threshold` items per call
  rather than the full queue.

- `vlm_video_temporal` frame sampling parameters hardcoded in `main()` rather than read
  from config. Fixed by wiring all sampling/evaluation parameters to the YAML config.

- All five `examples/pipelines/*/\_\_init\_\_.py` files were empty — they now expose
  `main()` for direct import, consistent with the CLI's importlib dispatch pattern.

### Changed

- README section renamed from "The Four Examples" to "The Five Examples" to include
  `vlm_video_temporal`.
- ROADMAP.md updated: P1-03, P1-04, P1-05, P2-01 all marked Done; P2-02 marked N/A
  (out of scope for a repo transformation).


## [0.1.0] — 2026-06-01

Initial release of the transformed repository.

### Added

- `production_vlm.config`: stdlib dataclass schemas for experiment configs.
- `production_vlm.drift`: `CosineDriftDetector` and `EWMADriftDetector`.
- `production_vlm.eval`: `numeric_accuracy`, `grounding_score`, `faithfulness_score`.
- `production_vlm.utils`: synthetic chart generator, vision encoder abstraction,
  dynamic batching queue, console helper, run logger.
- `vlm_chart_finetune` example (P0-02): LoRA fine-tuning on chart-QA.
- `embedding_drift_active_learning` example (P0-04): KS-test + EWMA drift detection
  with label-free active-learning triage.
- `vlm_edge_inference` example (P0-03): ONNX export, INT8 quantization, FastAPI serving.
- Shared `pyproject.toml` with tiered optional extras.
- GitHub Actions CI: lint, test, smoke-run all examples, docker build.
- CPU and GPU Dockerfiles.
- stdlib-only fallback verifier (`scripts/verify_no_pytest.py`).

[Unreleased]: https://github.com/Mattral/production-vlm-engineering/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Mattral/production-vlm-engineering/releases/tag/v0.1.0
