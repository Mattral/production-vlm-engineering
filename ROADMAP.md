# Roadmap

Status against the original transformation plan
(`Frontier_CV_Engineering_Playbook_Transformation_Roadmap_2026`). This file is kept honest:
items are marked done only once they're genuinely implemented and verified, not when code
merely exists.

## P0 -- Foundation & core executable examples

| Item | Status | Notes |
|---|---|---|
| P0-01: Repo structure & tooling | **Done** | `pyproject.toml`, `Makefile`, `src/cv_playbook/`, `examples/pipelines/`, `configs/`, `tests/`, `docker/`, `.github/workflows/`. CPU-only core deps (numpy/scipy/pyyaml/matplotlib/pillow); `ml`/`cli`/`onnx`/`serving`/`demo` as optional extras. |
| P0-02: VLM chart/document fine-tuning example | **Done** | LoRA on vision tower + LM, zero-download synthetic chart-QA data, three-metric eval harness (numeric/grounding/faithfulness), honest CPU-fallback labeling. Runs in ~5s on CPU. |
| P0-03: Inference optimization & edge deployment example | **Done** | ONNX export + dynamic INT8 quantization path; honest synthetic-backbone fallback (with a documented numpy float16-is-slower gotcha avoided); FastAPI serving stub with a real, unit-tested dynamic batching queue. Runs in ~2s on CPU fallback. |
| P0-04: Embedding drift detection & active learning | **Done** | KS-test cosine drift detector + frozen-baseline EWMA detector (a real self-defeating-variance bug was found and fixed during development -- see `src/cv_playbook/drift/__init__.py` docstrings), label-free active-learning triage, honest sensitivity-sweep benchmark. Runs in <1s on CPU. |

## P1 -- Utility library, differentiators, documentation

| Item | Status | Notes |
|---|---|---|
| P1-01: Shared utility library | **Done** (folded into P0-01) | `cv_playbook.config`, `cv_playbook.drift`, `cv_playbook.eval`, `cv_playbook.utils.*` -- built alongside the P0 examples rather than as a separate pass, since the examples needed it immediately. |
| P1-02: Robustness & safety layer (adversarial/OOD/grounding guard) | **Not started** | Scoped but not yet implemented. The grounding/faithfulness metrics in `cv_playbook.eval` are a partial building block; a dedicated adversarial test harness and a "wrap VLM inference with guard checks" pattern are still open. |
| P1-03: Advanced evaluation & benchmarking harness | **Partially done** | Numeric accuracy + grounding + faithfulness exist; the embedding-drift benchmark sweep covers sensitivity analysis. Not yet built: a synthetic perturbation generator (lighting/angle/noise/style-shift beyond the existing chart style-shift) and a unified comparative-report generator across all three examples. |
| P1-04: Light video/3D extension | **Not started** | Optional/stretch per the original plan; deferred. |
| P1-05: Comprehensive documentation & citations | **Partially done** | Inline citations exist in every module/example docstring (LoRA, RAGAS, KS-test, SPC/EWMA, dynamic quantization, etc.), and a minimal `mkdocs.yml` + `docs/` skeleton exists. The full polished MkDocs Material site with rendered example outputs, benchmark tables, and architecture diagrams is **not** built yet. |

## P2 -- Polish, releases, promotion

| Item | Status | Notes |
|---|---|---|
| P2-01: Releases, CHANGELOG, CONTRIBUTING, issue templates | **Not started** | |
| P2-02: Promotion & measurement plan | **Not started** | Outside the scope of what a repository transformation can do on its own. |

## Design decisions worth recording

- **Core dependencies are CPU-only and minimal** (numpy/scipy/pyyaml/matplotlib/pillow).
  `pydantic`, `rich`, and `typer` were deliberately *not* made hard dependencies of
  `cv_playbook` itself -- they're nice-to-haves gated behind the `cli` extra, with the core
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
