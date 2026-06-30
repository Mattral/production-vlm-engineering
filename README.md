<div align="center">

# Production VLM Engineering

**Reproducible, production-grade pipelines for modern multimodal vision systems.**

*Efficient VLM adaptation · Embedding-space drift detection · Edge inference · Robustness & safety · Video/temporal reasoning*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/production-vlm-engineering/blob/main/notebooks/colab/01_evaluation_metrics_colab.ipynb)

</div>

---

Most "CV best practices" repos are static markdown — educational, but nothing you can run, benchmark against, or build on. This repo takes the opposite stance: **five runnable pipelines** that implement current (June 2026) frontier techniques with production constraints built in from the start.

Every example ships with:
- A **CPU-only fallback path** so you can verify mechanics before spending GPU hours
- **Honest benchmark tables** that clearly label CPU smoke-test vs. real GPU numbers
- **Before/after metrics** with working code, not slides
- **Inline citations** to the specific 2025–2026 papers behind each technique

> *"This is the kind of repo I wish existed when I was trying to productionize VLMs."*

---

## Benchmark Results (CPU smoke-test paths)

Five examples run end-to-end in under 25 seconds on CPU. Numbers marked ⚠️ are smoke-test
proxies; GPU path notes are in the rightmost column.

| Example | Headline metric | Value | Notes |
|---|---|---|---|
| `vlm_chart_finetune` | Faithfulness Δ (LoRA vs zero-shot) | **+0.71** | ⚠️ Proxy; real LoRA trains on GPU |
| `vlm_chart_finetune` | Structured extraction MAPE (fine-tuned) | **0%** | ⚠️ Ground-truth JSON used as proxy |
| `embedding_drift_active_learning` | Detection delay (batches) | **0** | ✅ Real KS-test, numpy/scipy |
| `embedding_drift_active_learning` | Retraining runs triggered | **2** | ✅ Real threshold-based feedback loop |
| `vlm_edge_inference` | INT8 speedup vs fp32 | **3.99×** | ⚠️ Synthetic backbone; real ONNX needs `[onnx]` extra |
| `vlm_robustness_guard` | Hallucination guard precision / recall | **1.0 / 1.0** | ✅ Real faithfulness harness |
| `vlm_robustness_guard` | OOD detection TP rate | **85–100%** | ✅ Real kNN calibration |
| `vlm_video_temporal` | Frame sampling faithfulness | **0.70** | ✅ Real temporal grounding metric |

Full tables (per-severity perturbation breakdown, OOD ROC sweep, benchmark timestamps):
[`benchmarks/reports/benchmark_report.md`](benchmarks/reports/benchmark_report.md) — regenerate with `python benchmarks/run_all.py`.

---

## Architecture

```
production_vlm/           Shared library — CPU-only, zero hard ML deps
├── config.py             Fail-fast dataclass config schemas (stdlib dataclasses, no pydantic)
├── drift/                KS-test CosineDriftDetector + frozen-baseline EWMA SPC
├── eval/                 Numeric accuracy, grounding, faithfulness (RAGAS-inspired)
├── robustness/           ImageNet-C perturbations, kNN OOD detection, hallucination guard
└── utils/                Synthetic charts, vision encoder, batching queue,
                          observability (JSONL + Prometheus), retraining trigger

examples/pipelines/
├── vlm_chart_finetune/              LoRA (vision tower + LM) + structured JSON extraction
├── embedding_drift_active_learning/ KS-test + EWMA drift, AL triage, retraining loop
├── vlm_edge_inference/              ONNX export, INT8 quantization, FastAPI dynamic batching
├── vlm_robustness_guard/            Perturbation sweep, OOD, hallucination guard
└── vlm_video_temporal/              Frame sampling, temporal grounding, scene-change detection

notebooks/                3 interactive notebooks with pre-executed output cells
benchmarks/run_all.py     Unified runner → Markdown + JSON comparative report
tests/                    pytest suite (56 checks via verify_no_pytest.py)
scripts/verify_no_pytest.py  stdlib-only verifier (no pytest needed, runs in CI)
```

---

## Quickstart

```bash
git clone https://github.com/Mattral/production-vlm-engineering
cd production-vlm-engineering
make setup                # ~15s, CPU-only install
cv-playbook list-examples # see all four examples
```

Run all examples and generate a unified benchmark report:

```bash
python benchmarks/run_all.py
# → benchmarks/reports/benchmark_report.md  (Markdown, paste into PRs / docs)
# → benchmarks/reports/benchmark_report.json (machine-readable)
```

For real GPU fine-tuning and ONNX export:

```bash
make setup-gpu            # adds torch / transformers / peft / onnxruntime
cv-playbook run-example vlm_chart_finetune
cv-playbook run-example vlm_edge_inference
```

---

## The Five Examples

### 1 · VLM Chart Fine-Tuning ([`vlm_chart_finetune`](examples/pipelines/vlm_chart_finetune/))

LoRA adapts **both** the vision tower and language model projections (the 2025–2026 convention for multimodal LoRA — language-only adapters leave the visual representation unchanged, which limits improvement on tasks that require reading numeric values off a chart). Default checkpoint: `Qwen2-VL-2B-Instruct`, swappable via config YAML.

Evaluation uses three metrics purpose-built for chart/document QA rather than BLEU or exact-match:
- **Numeric accuracy** — relative-tolerance matching on extracted numeric tokens (2% tolerance, matching ChartQA evaluation convention)
- **Grounding score** — fraction of content words in the prediction that appear in the source evidence
- **Faithfulness score** — weighted composite (60% numeric, 40% grounding), inspired by RAGAS adapted for image evidence

References: Hu et al. (2021) LoRA · Wang et al. (2024) Qwen2-VL · Es et al. (2023) RAGAS

### 2 · Embedding Drift Detection & Active Learning ([`embedding_drift_active_learning`](examples/pipelines/embedding_drift_active_learning/))

Addresses the **#1 cited cause of silent production CV failure** in 2026 enterprise deployment reports: the model keeps serving while the input distribution has shifted (new camera, new rendering pipeline, new document format) and accuracy decays with no error signal.

Two complementary detectors with different semantics:
- **`CosineDriftDetector`** — two-sample KS test on cosine-similarity distributions. Persistent: fires every batch where drift is present. Right for monitoring dashboards.
- **`EWMADriftDetector`** — EWMA-SPC with a *frozen baseline standard deviation*. Onset-detecting: fires when the shift begins, then re-centers. Right for alerting systems.

When drift is flagged, `select_for_active_learning()` ranks the batch by distance from the reference centroid — a free, label-free novelty proxy — and queues the most novel samples for human labeling first.

References: Massey (1951) KS test · Montgomery (2020) SPC · Settles (2009) active learning survey

### 3 · Edge Inference & Serving ([`vlm_edge_inference`](examples/pipelines/vlm_edge_inference/))

ONNX export → dynamic INT8 quantization → before/after benchmark table (latency p50/p95, throughput, peak memory, accuracy retention) across multiple image sizes and batch sizes. Target: 3–5× speedup with <2% accuracy drop.

The FastAPI serving stub implements the **dynamic batching pattern** used by Triton Inference Server and TorchServe: requests queue and flush on `max_batch_size` or `max_batch_wait_ms`, whichever comes first. The batching queue (`production_vlm.utils.batching_queue.BatchingQueue`) is dependency-free and unit-tested independently — drop it into any asyncio serving layer.

References: Jacob et al. (2018) integer-arithmetic-only inference

### 4 · Robustness & Safety Guard ([`vlm_robustness_guard`](examples/pipelines/vlm_robustness_guard/))

Three production failure modes, each with a concrete detection or mitigation:

**Perturbation robustness**: Six ImageNet-C-style corruptions (brightness, contrast, noise, blur, rotation, occlusion) at five severity levels. Brightness and contrast are fully robust via adaptive background-color estimation. Blur and rotation genuinely degrade the reader at high severity — the honest result for destructive perturbations.

**OOD detection**: `KNNOODDetector` calibrates its threshold from the reference set's own leave-one-out similarity distribution (targeting a specific false-positive rate rather than an arbitrary cosine cutoff). Validated operating point: 100% TP at 12.5% FP on the style-shift scenario.

**Hallucination guard**: `HallucinationGuard` converts `faithfulness_score` into a three-tier pass/flag/reject decision, returning a safe fallback message on reject rather than surfacing an ungrounded answer to the user.

References: Hendrycks & Dietterich (2019) ImageNet-C

### 5 · Video / Temporal Reasoning ([`vlm_video_temporal`](examples/pipelines/vlm_video_temporal/))

Minimal but genuinely runnable template for multi-frame VLM reasoning (P1-04). Three frame sampling strategies compared on a synthetic clip dataset, a temporal grounding metric that extends `faithfulness_score` to multi-frame evidence, and structured JSON answer output matching a versioned schema.

The key architectural connection: the same `CosineDriftDetector` from example 2 detects scene changes by flagging frames whose embeddings drift from the preceding window — no scene-change-specific training needed. This means a production video pipeline can reuse the same drift monitoring infrastructure as a batch monitoring system.

Replace `_synthetic_frame_sequence()` with a real video loader (decord/torchvision) and `_mock_vlm_temporal()` with your VLM call (Video-LLaVA, VITA, InternVL2-Video). The sampling, grounding, and scene-detection code is unchanged.

References: Lin et al. (2023) Video-LLaVA · Fu et al. (2024) VITA · CVPR 2026 temporal VLM track

---

## Honesty About Fallback Paths

Every example detects at runtime whether the real ML stack (torch/transformers/peft/bitsandbytes, or onnx/onnxruntime) is installed and a CUDA device is available. If not, it runs a CPU-only path that exercises real data generation, real config validation, and the real evaluation harness — but uses simulated model outputs or a compute-equivalent proxy backbone rather than actual model weights.

This distinction is recorded in every `results.json`:

```json
{ "ran_with_real_ml_stack": false, ... }
```

And printed unambiguously in console output. CPU smoke-test numbers and real GPU numbers are never presented interchangeably — in this repo or in the generated benchmark report.

---

## 📓 Notebooks (Open in Google Colab)

Interactive walkthroughs of the core techniques — **no GPU required, no local setup**. Click any badge to launch directly in Colab; the package installs automatically at the top of each notebook.

| # | Notebook | Covers | Launch |
|---|---|---|---|
| 01 | Evaluation Metrics | `numeric_accuracy`, `grounding_score`, `faithfulness_score` — why BLEU fails on chart answers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/production-vlm-engineering/blob/main/notebooks/colab/01_evaluation_metrics_colab.ipynb) |
| 02 | Drift Detection & Active Learning | `CosineDriftDetector`, `EWMADriftDetector` (frozen-baseline SPC), label-free active learning triage | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/production-vlm-engineering/blob/main/notebooks/colab/02_drift_detection_colab.ipynb) |
| 03 | Robustness & Safety Guard | Perturbation sweep, `KNNOODDetector`, `HallucinationGuard`, production wrapper pattern | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/production-vlm-engineering/blob/main/notebooks/colab/03_robustness_guard_colab.ipynb) |

Each Colab notebook runs end-to-end in 1–3 minutes and includes a "try it yourself" cell at the end for experimenting with your own inputs. See [`notebooks/colab/README.md`](notebooks/colab/README.md) for the full index.

**Prefer to read without running anything?** Pre-executed versions with all output cells populated render directly on GitHub:
[`01_evaluation_metrics.ipynb`](notebooks/01_evaluation_metrics.ipynb) · [`02_drift_detection_active_learning.ipynb`](notebooks/02_drift_detection_active_learning.ipynb) · [`03_robustness_safety_guard.ipynb`](notebooks/03_robustness_safety_guard.ipynb)

---

## Testing

```bash
make test                          # pytest, 40 tests, requires pip install -e ".[dev]"
python scripts/verify_no_pytest.py # stdlib-only fallback, no pytest needed
```

---

## Related Work

This repo is part of a set of production ML engineering resources:

- **[GuardRail-Studio](https://github.com/Mattral/GuardRail-Studio)** — LLM/VLM safety and guardrail patterns (the hallucination guard here follows GuardRail-Studio conventions)
- **[FlashSpec](https://github.com/Mattral/FlashSpec)** — Memory-efficient speculative decoding (complements the edge inference example)
- **[Multimodal RAG](https://github.com/Mattral)** — Vision + retrieval pipelines (the chart-QA fine-tuning here is a natural upstream complement)

The patterns here — efficient adaptation, embedding-space monitoring, faithfulness evaluation, inference optimization — are designed to compose with those repos rather than duplicate them.

---

## Why This Matters for 2027

The 2026 → 2027 trajectory is clear: VLMs become the default perception layer for agentic systems, on-device efficient models become viable for real-time use, and **MLOps and robustness patterns become standardized requirements** for any serious VLM deployment — analogous to how LLMOps matured in 2024-2025.

The adoption gap right now is that practitioners understand the high-level ideas (VLMs, LoRA, drift detection, guardrails) but struggle with reproducible end-to-end implementations that handle real data variation and integrate safety/observability. This repo closes that gap for the vision/multimodal stack specifically.

Concretely:
- **The LoRA pipeline** prepares you for the next generation of chart/document VLMs and structured visual reasoning tasks that will be central to agentic systems requiring visual grounding
- **The drift detector** is the monitoring primitive that every production VLM deployment will need once models are serving real traffic
- **The edge inference patterns** generalize directly to the on-device efficient VLMs that will proliferate in 2027 on Jetson-class and similar hardware
- **The robustness guard** aligns with the emerging standardization of safety layers for multimodal models

---

## Citations & References

Every technique is cited inline in the docstring of the function/class that implements it. A consolidated bibliography (LoRA, RAGAS, ChartQA, KS test, SPC, PGD, ImageNet-C, ONNX quantization, Triton batching, Video-LLaVA/VITA) lives in [`docs/citations.md`](docs/citations.md).

---

## License

MIT. See [LICENSE](LICENSE).

---

<div align="center">
<sub>Built with production constraints in mind, not just research plausibility.</sub>
</div>
