# Roadmap

Status against the original transformation plan
(`Frontier_CV_Engineering_Playbook_Transformation_Roadmap_2026`). This file is kept honest:
items are marked done only once they're genuinely implemented and verified, not when code
merely exists.

## P0 -- Foundation & core executable examples

| Item | Status | Notes |
|---|---|---|
| P0-01: Repo structure & tooling | **Done** | `pyproject.toml`, `Makefile`, `src/production_vlm/`, `examples/pipelines/`, `configs/`, `tests/`, `docker/`, `.github/workflows/`. CPU-only core deps (numpy/scipy/pyyaml/matplotlib/pillow); `ml`/`cli`/`onnx`/`serving`/`demo` as optional extras. |
| P0-02: VLM chart/document fine-tuning example | **Done** | LoRA on vision tower + LM, zero-download synthetic chart-QA data, three-metric eval harness (numeric/grounding/faithfulness), honest CPU-fallback labeling. Runs in ~5s on CPU. |
| P0-03: Inference optimization & edge deployment example | **Done** | ONNX export + dynamic INT8 quantization path; honest synthetic-backbone fallback (with a documented numpy float16-is-slower gotcha avoided); FastAPI serving stub with a real, unit-tested dynamic batching queue; **KV-cache memory-efficient decoding analysis** (`production_vlm.utils.kv_cache`) comparing MHA/GQA/MQA/sliding-window attention strategies via closed-form memory arithmetic — addresses the roadmap's separate "memory-efficient decoding/attention optimizations" requirement, distinct from the vision-encoder ONNX work above. TensorRT/OpenVINO documented as extension paths with code stubs and Jetson-class FPS targets. Runs in ~2s on CPU fallback. |
| P0-04: Embedding drift detection & active learning | **Done** | KS-test cosine drift detector + frozen-baseline EWMA detector (a real self-defeating-variance bug was found and fixed during development -- see `src/production_vlm/drift/__init__.py` docstrings), label-free active-learning triage, honest sensitivity-sweep benchmark. Runs in <1s on CPU. |

## P1 -- Utility library, differentiators, documentation

| Item | Status | Notes |
|---|---|---|
| P1-01: Shared utility library | **Done** (folded into P0-01) | `production_vlm.config`, `production_vlm.drift`, `production_vlm.eval`, `production_vlm.utils.*` -- built alongside the P0 examples rather than as a separate pass, since the examples needed it immediately. |
| P1-02: Robustness & safety layer (adversarial/OOD/grounding guard) | **Done** | `production_vlm.robustness`: six ImageNet-C-style natural perturbations, calibrated kNN OOD detector (validated FP/TP tradeoff documented in class docstring), hallucination guard with three-tier pass/flag/reject decision. Fourth example pipeline `vlm_robustness_guard` ties all three together. Full test coverage in `tests/test_robustness.py` and `scripts/verify_no_pytest.py`. |
| P1-03: Advanced evaluation & benchmarking harness | **Done** | `benchmarks/run_all.py` unified runner generates Markdown + JSON comparative report across all five examples. Synthetic perturbation generator exists (`production_vlm.robustness.NaturalPerturbation`: six ImageNet-C-style types, severity sweep, result table). Structured JSON chart extraction added to `vlm_chart_finetune` (P0-02 requirement). |
| P1-04: Light video/3D extension | **Done (minimal template)** | `vlm_video_temporal` example: three frame-sampling strategies (uniform, keyframe L1-diff, adaptive), temporal grounding metric, structured JSON answer schema, scene-change detection via `CosineDriftDetector`. Explicit `next_steps` in results.json for swapping in real video loaders and VLMs. |
| P1-05: Comprehensive documentation & citations | **Done** | Full MkDocs Material site: quickstart, architecture (Mermaid diagram), per-example deep-dives, concept pages (LoRA, drift, metrics, robustness) each with a technique-specific "Why this matters for 2027" section, full API reference. Benchmark report embedded. 3 interactive Jupyter notebooks with pre-executed cells. All techniques cited inline at the point of implementation plus a consolidated `docs/citations.md`. |

## P2 -- Polish, releases, promotion

| Item | Status | Notes |
|---|---|---|
| P2-01: Releases, CHANGELOG, CONTRIBUTING, issue templates | **Done** | `CHANGELOG.md` with full history and documented bug record. 3 GitHub issue templates (bug, feature, wrong result) + PR template enforcing correctness standards. `CONTRIBUTING.md`. CI now covers all 5 examples + benchmark runner + docs build. |
| P2-02: Promotion & measurement plan | **N/A — out of scope** | Outside the scope of what a repository transformation can do on its own. The work is visible through the repo's structure and CI; promotion cadence and metrics are the owner's call, not automatable. |

## Design decisions worth recording

- **Core dependencies are CPU-only and minimal** (numpy/scipy/pyyaml/matplotlib/pillow).
  `pydantic`, `rich`, and `typer` were deliberately *not* made hard dependencies of
  `production_vlm` itself -- they're nice-to-haves gated behind the `cli` extra, with the core
  library degrading gracefully (plain dataclasses instead of pydantic, plain `print` instead
  of `rich`) when absent. This was a direct response to verifying the repo in a genuinely
  offline, no-extra-packages environment during development and finding the original
  pydantic/rich/typer-everywhere design simply didn't run there.
- **Every example has an honest, labeled CPU/no-network fallback path.** This was a hard
  requirement discovered while building, not an afterthought: a benchmark repo whose claims
  can't be checked by someone who clones it without a GPU and network access isn't
  trustworthy. Every fallback path runs real code (real data generation, real metric
  computation, real differential timing) and is unambiguously labeled as a smoke test rather
  than a benchmark, both in console output and in each `results.json`.
- **Two real bugs were found and fixed by actually running the code**, not just writing it:
  a shift-direction regeneration bug in `SyntheticEmbeddingProxy` that silently canceled out
  the injected drift signal across a batch, and a variance-contamination bug in the original
  EWMA drift detector that made it unable to detect the very step-change that inflated its
  own control limits. Both are documented in-place rather than silently fixed, since the
  failure modes are instructive.

- **Two more real bugs found in `vlm_video_temporal` by an external audit that noticed all
  three frame-sampling strategies scored an identical 0.698**, which shouldn't happen if the
  strategies are actually being compared: (1) `temporal_grounding_score` was called with
  `faithfulness_score(prediction, prediction, ev)`, comparing the prediction against itself
  rather than the real ground-truth `sample.answer`, making the numeric-accuracy half of the
  composite score a no-op that's always perfect; (2) evaluation was scored against
  `sample.per_frame_evidence` for the *entire* clip regardless of which frames a strategy
  actually sampled, and the mock model answered correctly even when the frame it depended on
  was dropped -- so no strategy could ever be penalized for discarding the key frame, which
  defeats the purpose of the comparison. Fixed by threading the true reference through, scoring
  each strategy only against the evidence for the frames it kept, and making the mock model
  fall back to a lower-confidence guess when the key frame isn't in its sample. Scores now
  genuinely differ across strategies (0.387 / 0.266 / 0.485 on a representative run).
- **One real packaging bug: the documented `production-vlm run-example <name>` command
  (the installed console-script entry point) crashed with `ModuleNotFoundError: No module
  named 'examples'` for every example.** `examples/` lives at the repo root and isn't part of
  the installed wheel; `python -m production_vlm.cli` implicitly adds the caller's working
  directory to `sys.path` and so happens to find it, but the installed script does not, since
  its own location becomes `sys.path[0]` instead. This means `docs/contributing.md`'s explicit
  instruction to contributors -- run the example via that exact command -- didn't work as
  written. Fixed in `cli.py` by inserting the repo root onto `sys.path` before importing an
  example module, mirroring the `sys.path` insertion each example's `run.py` already does in
  the opposite direction (adding `src/` so it can find `production_vlm`). Verified against the
  actual installed entry point, not just `python -m`, for all 5 examples.
- **Running tally: seven real bugs found and fixed by actually running the code** across this
  project's history (two in P0-04, two more in P1-02, two in `vlm_video_temporal`, one in the
  CLI packaging), none caught by writing or reading the code alone.

- **Repository renamed** from `computer-vision-playbook` / `cv_playbook` to `production-vlm-engineering` / `production_vlm` to accurately reflect the project's focus and production-engineering scope.
- **Two more real bugs found and fixed in P1-02** beyond those already documented above
  (the `SyntheticEmbeddingProxy` shift-direction bug is the same fix noted earlier in this
  file, not a separate occurrence — it is not re-counted here): (1) `_find_plot_area_bounds`
  used an absolute spine-darkness threshold that broke under contrast and blur perturbation;
  (2) chart reader assumed bars spanned the full figure width rather than the matplotlib axes
  area (a critical misunderstanding of figure layout that caused every bar to report identical
  height). Both documented in-place.
- **Observability module** (`production_vlm.utils.observability`): structured JSONL event log (zero dependencies, always emitted) plus optional Prometheus metrics server (graceful no-op when `prometheus_client` not installed). All drift, OOD, and guard events are versioned with `schema_version` for forward-compatible log consumption. This was a specific P0-04 roadmap requirement ("log metrics, optional Prometheus exposition") that had been deferred until this session.

- **Retraining trigger** (`production_vlm.utils.retraining`): closes the drift → active-learning → retrain feedback loop explicitly called out in P0-04 ("integrate with training example, trigger retraining simulation on drifted data"). Key design: `_fire()` drains exactly `queue_threshold` items per invocation so a large `enqueue_batch()` call fires multiple times correctly rather than coalescing all samples into one oversized batch. This was a real bug found during testing (9 items at threshold=3 fired once with 9 instead of three times with 3 each) and fixed.

- **Five examples, not four**: `vlm_video_temporal` added as the P1-04 "minimal runnable template" the roadmap explicitly accepts. It demonstrates real frame-sampling algorithms (L1 scene-change detection, adaptive motion-based sampling) and is intentionally forward-pointing rather than feature-complete.

- **Structured JSON extraction** added to `vlm_chart_finetune` (P0-02 roadmap line 101: "structured output (JSON for chart values)"): `_CHART_JSON_SCHEMA`, `_extract_structured_json`, `_structured_extraction_accuracy` with schema validity rate, numeric MAPE, and category coverage — a real before/after table showing 40% → 0% MAPE improvement.

- **A genuinely non-deterministic bug shipped silently for multiple sessions before a real `pytest`
  CI run caught it.** `SyntheticEmbeddingProxy._chart_to_vector()` used Python's built-in `hash()`
  on a string, which is randomized per-process by default (`PYTHONHASHSEED`). Every embedding
  computed from chart metadata — and therefore every OOD detection rate, drift detection outcome,
  and robustness metric — silently varied run-to-run despite every other seed in the pipeline
  being fully deterministic. This never surfaced in this sandbox's own `verify_no_pytest.py` runs
  because that script happened to only ever run once per invocation with whatever hash seed that
  process got, and every single run "passed" on its own terms. It only surfaced when a *real* CI
  environment ran the test suite and, on that particular invocation's hash seed, an assertion
  landed on the wrong side of a threshold. Fixed with `zlib.crc32` (a genuinely deterministic hash)
  and verified stable across 10 independent process invocations. The broader lesson: a stdlib-only
  fallback verifier that can't install pytest is still valuable, but running the *same* fixed-seed
  test exactly once per session is not the same guarantee as running it across many independent
  process invocations — non-determinism bugs specifically hide from single-invocation testing.

- **A systematic, evidence-based re-audit against the roadmap's exact text (not memory of prior
  audits) found two genuine remaining gaps** after several sessions of "everything is done"
  claims: (1) P0-03's "memory-efficient decoding / attention optimizations" requirement was
  never actually implemented — the existing `vlm_edge_inference` work covered ONNX/quantization
  for the *vision encoder* only, a different bottleneck than the *language-model decoder's*
  KV-cache memory during autoregressive generation, which the roadmap names as a separate
  concern; (2) the roadmap's "'Why this matters for 2027' framing in each major section" was
  present only in the top-level README, not distributed across the individual concept pages as
  specified. Both are now fixed: `production_vlm.utils.kv_cache` implements real, tested,
  closed-form MHA/GQA/MQA/sliding-window memory comparisons (13 new unit tests, wired into
  `vlm_edge_inference` as Component 2 with its own plot and results section), and each of the
  four concept pages (`lora.md`, `drift.md`, `metrics.md`, `robustness.md`) now has its own
  technique-specific 2027 framing grounded in that page's actual content, not generic filler
  copy-pasted across pages. The lesson: periodically re-deriving the gap list from the source
  document's literal text, rather than trusting an accumulated mental model of "what's already
  done," catches drift between claimed and actual completion.
