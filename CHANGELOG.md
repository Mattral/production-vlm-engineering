# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Version numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added (roadmap gap-closure: memory-efficient decoding + distributed 2027 framing)

- **`production_vlm.utils.kv_cache`** — closed-form KV-cache memory analysis across four
  attention/cache strategies (MHA, GQA, MQA, sliding-window), addressing the "memory-efficient
  decoding / attention optimizations" requirement of P0-03 that had been missed across several
  prior sessions despite the rest of P0-03 (ONNX export, INT8 quantization, dynamic batching)
  being complete. This is a genuinely separate bottleneck from the vision-encoder work already
  in `vlm_edge_inference`: a VLM's language-model decoder generates tokens autoregressively
  against a KV-cache dominated by hundreds to thousands of visual tokens, where memory (not
  FLOPs) is usually the binding constraint. Wired into `vlm_edge_inference` as "Component 2"
  with its own results section and a two-panel plot (absolute memory vs. sequence length, and
  relative-to-MHA bar chart at the longest tested sequence). 13 new unit tests verify the core
  invariants: MHA is the 1.0x baseline, GQA/MQA reduce memory by exactly the query-to-kv head
  ratio, sliding-window memory is provably capped (identical absolute MB at the window size and
  4x beyond it), and the strict MHA > GQA > MQA memory ordering holds at any fixed sequence
  length. Cites Vaswani et al. (2017), Shazeer (2019, MQA), Ainslie et al. (2023, GQA), Beltagy
  et al. (2020, sliding-window), and references FlashAttention-2 (Dao, 2023), PagedAttention
  (Kwon et al., 2023), and speculative decoding (Leviathan et al., 2023) for the broader
  efficient-decoding landscape this connects to.

- **Distributed "Why this matters for 2027" sections** across all four `docs/concepts/` pages
  (LoRA, drift detection, evaluation metrics, robustness), each grounded in that page's specific
  technique rather than generic copy-pasted framing. The roadmap specified this framing "in each
  major section," but it had only ever been added to the top-level README.

- `production_vlm.utils.observability` and `production_vlm.utils.retraining` — present in the
  codebase and covered by tests since an earlier session, but never added to `docs/api.md`;
  fixed while updating the API reference for the new `kv_cache` module.

### Fixed (found by real `ruff check`/`ruff format`, installed and run directly)

- **63 real ruff lint errors** across `src/`, `tests/`, `examples/`, `benchmarks/`, `scripts/`:
  18× `B905` (`zip()` without explicit `strict=`, now `strict=True` everywhere since every
  zipped pair is guaranteed or expected to be equal-length by construction), 3× `E741`
  (ambiguous variable name `l`, renamed to `label`), 3× `F841` (unused locals: a leftover
  `processor` variable in the real ONNX-export path that isn't needed since the dummy input is a
  raw tensor not a tokenized one; a genuinely dead `combined_result`/`combined_evidence`
  computation in the temporal grounding metric; an unused `results` binding in a batching-queue
  test), 1× `B007` (unused loop variable `i` from a stale `enumerate()` wrapper). Verified
  `ruff check src tests examples benchmarks scripts` now reports zero errors, and reran the full
  pytest suite (104 tests) plus all 5 examples end-to-end afterward to confirm none of these
  fixes changed behavior.

- **24 files failed `ruff format --check`** (whitespace/wrapping normalization only, no logic
  changes) — applied `ruff format` and reverified pytest + all 5 examples pass identically.

- **`legacy/cv-playbook-original/` was not excluded from ruff's scope**, meaning any lint
  invocation that scans the whole repo (`ruff check .`, some pre-commit configurations) would
  flood output with unrelated errors from the preserved original "from scratch" educational
  Python scripts (20 files) that were never meant to be linted to this repo's standard. Added
  `exclude = ["legacy", "notebooks"]` to `[tool.ruff]` in `pyproject.toml` (notebooks are
  standalone tutorial content, never part of CI's actual lint scope, and holding demo-cell code
  to the same bar as production source doesn't serve a purpose); verified `ruff check .` now
  produces zero errors when scanning the entire repository.

- **Stale `cv-playbook` CLI command name in 16 documentation files.** The package was renamed
  from `cv-playbook`/`cv_playbook` to `production-vlm`/`production_vlm` in an earlier session,
  and `pyproject.toml`'s actual installed script entry point is `production-vlm`, but README.md,
  CONTRIBUTING.md, the PR template, and 12 files under `docs/` still told readers to run
  `cv-playbook list-examples` / `cv-playbook run-example <name>` — a command that does not exist
  post-rename and would fail with "command not found" for anyone following the quickstart
  verbatim. Fixed via a scoped regex substitution that only touches actual command invocations
  (`cv-playbook` immediately followed by `list-examples`, `run-example`, or `benchmark`), not any
  incidental prose mention of the old name.

- **Stale "three"/"four example(s)" counts across 6 files**, left over from when the repo had
  three then four examples before `vlm_video_temporal` (P1-04) was added. Fixed in README.md,
  CONTRIBUTING.md, `vlm_robustness_guard/README.md`, and 4 files under `docs/`.

- **`vlm_robustness_guard/README.md` documented 3 components, but the pipeline has had 4 since
  the adversarial-robustness (PGD) component was added earlier this session** — the README's
  component table, section numbering, and "Files" summary were never updated to include it.
  Added the missing "Component 2: Adversarial robustness (PGD proxy)" section and renumbered
  the rest.

- **Dockerfiles said "three example pipelines" and didn't copy `benchmarks/`** into the image,
  both stale relative to the current 5-example, benchmark-runner-equipped state of the repo.
  Fixed the comment and added `COPY benchmarks ./benchmarks` to both `docker/Dockerfile` and
  `docker/Dockerfile.gpu`.

### Fixed (found by real CI/pytest run, not the sandbox stdlib verifier)

- **`SyntheticEmbeddingProxy` embeddings were silently non-deterministic across process runs**,
  contradicting the class's own "deterministic" docstring claim. `_chart_to_vector()` used
  Python's built-in `hash(chart.units)` to derive one style-feature dimension — but Python
  randomizes string hashing per-process by default (`PYTHONHASHSEED`, a security feature against
  hash-collision DoS attacks), so `hash()` returns a *different* value every time the interpreter
  starts. This meant every embedding computed from a chart's `units` string — and therefore every
  OOD detection rate, drift detection outcome, and robustness-guard metric derived from those
  embeddings — silently varied from run to run despite identical, fully-specified integer seeds
  everywhere else in the pipeline. Verified directly: the OOD detector's TP rate on style-shifted
  inputs ranged from 77.5% to 100% across 10 fresh process invocations with *zero* code changes
  between them. Fixed by replacing `hash(chart.units)` with `zlib.crc32(chart.units.encode())`, a
  genuinely deterministic string hash — verified stable at exactly 100% TP rate across 10
  independent process runs after the fix, and confirmed all five examples now produce
  byte-identical `results.json` output (excluding legitimate wall-clock timestamp fields) across
  repeated independent runs. This was the most significant bug found this session: it silently
  undermined the reproducibility of every headline number in this repo that touches
  `SyntheticEmbeddingProxy`, without ever raising an error or looking obviously wrong on any
  single run.

- **`_structured_extraction_accuracy`'s zero-shot noise simulation used the same non-deterministic
  `hash()` pattern** (`hash(chart.title) % 2**32`), with a second, compounding bug: chart titles
  are drawn from a small combinatorial pool (metric × dimension), so multiple charts in the same
  eval set can share an identical title — meaning `hash(title)`-derived random draws were not
  independent across charts that happened to collide, in addition to not being reproducible
  across process runs. Fixed by seeding from `chart.style_seed` (unique per chart by
  construction, already present on `SyntheticChart`) instead. Verified deterministic and
  reproducible (`schema_validity_rate == 0.55` on every one of 5 independent process runs, vs.
  spuriously hitting `1.0` under the old code when a particular process's hash seed happened to
  avoid the 40%-failure threshold on all of that eval set's distinct titles).

- **`config.py` Literal validation silently no-op'd on every invalid value.** `from __future__ import
  annotations` makes all dataclass annotations lazy *strings* at runtime (e.g. `dtype:
  Literal["bf16","fp16","fp32"]` becomes the literal string `"Literal['bf16', 'fp16', 'fp32']"` in
  `__annotations__`, not the actual `Literal[...]` type object). `get_origin()` on that string
  silently returns `None` instead of `Literal`, so `_check_literal` never actually validated
  anything — `ModelConfig(dtype="int4")` was accepted without error. Fixed by resolving
  annotations via `typing.get_type_hints()`, which re-evaluates the postponed string back into
  the real type object. This is a real, previously undetected correctness bug in every dataclass
  config field typed as `Literal[...]`.

- **`EWMADriftDetector` false-alarmed on a genuinely stable signal with a too-small calibration
  window.** The `test_no_alarm_on_stable_signal` test used `baseline_n=5` — direct measurement
  showed a 5-sample standard-deviation estimate underestimated the true population std by 2.6x on
  this test's seed, making the nominal "3-sigma" band effectively much tighter than intended and
  prone to spurious false alarms (verified: 0/20 seeds false-alarm at `baseline_n=10`, matching
  standard SPC calibration-phase guidance of ~20-25 samples). Not a detector bug — a test
  constructed with a too-small calibration sample. Fixed the test; the detector code and the
  production config (`baseline_n=6`, used with a strong real drift signal) are unchanged.

- **`select_for_active_learning` test constructed an outlier that could never be detected, at any
  magnitude.** The test added a constant `+100.0` offset to one sample's every dimension and
  expected it to rank as "farthest from centroid" — but since the centroid is computed from the
  same batch containing the outlier, a large enough perturbation pulls the centroid toward
  itself, making the outlier appear *more* similar to its own self-polluted centroid, not less.
  Verified directly: no magnitude from 5 to 120 (in a 50-sample batch) ever produced a rank-0
  outlier — larger perturbations always increased the outlier's pull on the mean faster than they
  escaped it. This is the identical failure mode documented elsewhere in this codebase for
  `SyntheticEmbeddingProxy`/`KNNOODDetector` test construction. Fixed by using a tight,
  low-variance cluster (representing stable in-distribution embeddings, the function's actual
  design target) plus one genuinely distinct directional outlier, which reliably ranks as most
  novel (0/30 failures across seeds). Also documented the self-referential-centroid limitation in
  `select_for_active_learning`'s docstring for future users with different batch compositions.

- **Ruff line-length violations across the codebase (E501).** ~209 lines exceeded the configured
  100-char limit; `ruff format` doesn't rewrap comments/docstrings/f-strings, so these needed
  manual wrapping. Bumped `line-length` to 120 (a common, well-precedented convention) to resolve
  the majority without sacrificing docstring/error-message readability, then manually wrapped the
  remaining 19 genuine violations (verified via Python's character-accurate `len()`, not shell
  `awk` which can over-count multi-byte UTF-8 sequences as more than one character).

### Added

- Sandbox stdlib verifier (`scripts/verify_no_pytest.py`) strengthened with regression coverage
  for all bugs above: invalid dtype/logging Literal rejection, EWMA no-false-alarm-on-stable-data
  at a statistically sound `baseline_n`, and active-learning most-novel-outlier selection with a
  construction that doesn't hit the self-referential-centroid failure mode. None of these three
  scenarios were previously covered by the sandbox verifier, which is why they weren't caught
  until a real `pytest` run in CI surfaced them. 80 checks total, up from 76.

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
