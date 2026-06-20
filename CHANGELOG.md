# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- `production_vlm.robustness` module (P1-02): six ImageNet-C-style natural perturbations,
  calibrated kNN OOD detector, three-tier hallucination guard (`HallucinationGuard`).
- `vlm_robustness_guard` example pipeline: perturbation sweep table, OOD FP/TP benchmark,
  and hallucination guard precision/recall report.
- `benchmarks/run_all.py`: unified runner that collects all four examples' results and
  generates a single Markdown + JSON comparative benchmark report.
- Full MkDocs Material documentation site: architecture guide, per-example deep-dives,
  concept pages (LoRA, drift detection, metrics, robustness), full API reference.
- `SyntheticChart.plot_bbox` field: matplotlib axes pixel coordinates passed through from
  the chart generator to the robustness chart reader, eliminating fragile pixel-heuristic
  spine detection that broke under blur/contrast perturbation.

### Fixed

- **`SyntheticEmbeddingProxy` shift direction regenerated per sample** instead of being fixed
  once at construction time, causing the injected OOD signal to cancel out across batches
  (each sample shifted in a different random direction). Fixed by computing `_shift_direction`
  once in `__init__`.
- **`EWMADriftDetector` variance contamination**: a continuously-adapting variance estimate
  is self-defeating under a step-change (the jump inflates the estimate and widens the
  control limits). Fixed by freezing `_baseline_std` from the calibration period only.
- **Chart reader assumed bars span full figure width**: matplotlib adds y-axis label/tick
  margins, so bars live within a subset of the figure canvas. Fixed by using
  `ax.get_window_extent()` to get the true axes bounding box.
- **Chart reader background threshold absolute, not adaptive**: a fixed `pixel ≥ 235`
  threshold breaks under brightness/contrast perturbation. Fixed by sampling the actual
  image corners to estimate the background color.
- **Top-axis spine caught as "bar start"**: after `y_top` is correctly placed at the axes'
  inner edge, the spine itself sits at `y_top + 1`. Fixed by skipping the first 3 rows
  of each column before scanning for bar content.
- **Hallucination injection used a real value from another bar**: the injected "wrong"
  answer could accidentally match a genuine evidence value, scoring high faithfulness
  despite being incorrect. Fixed by using a fabricated number (3× the chart maximum).

### Changed

- Repository renamed from `computer-vision-playbook` / `cv_playbook` to
  `production-vlm-engineering` / `production_vlm`.
- `KNNOODDetector` default `percentile` changed from `1.0` to `15.0` after empirical
  calibration showing `percentile=1` gives only 2.5% TP rate (useless as a guard) while
  `percentile=15` gives 100% TP at 12.5% FP.
- `HallucinationGuard.check_batch()` now validates that all three lists have equal length
  and raises `ValueError` on mismatch.
- Perturbation sweep now filters near-tie charts (where the top-2 values differ by less
  than 1% of the maximum) to avoid pixel-resolution-limit failures that are unrelated to
  the perturbation being tested.

---

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
