# Production VLM Engineering

Reproducible, production-grade pipelines for modern multimodal vision systems: efficient
VLM fine-tuning on charts/documents, embedding-space drift detection with active learning,
and inference optimization for edge deployment. Built to current (2026) practice, with
runnable code, honest benchmarks, and clear before/after numbers rather than slide-style
best-practice notes.

This repository is a ground-up transformation of an earlier collection of computer vision
study notes. That original material is preserved under [`legacy/production-vlm-original/`](legacy/production-vlm-original/)
for anyone who finds the conceptual walkthroughs useful, but it is no longer the focus of
this repo.

## Why this exists

Most public "CV best practices" repos are static markdown: useful as notes, but nothing you
can run, benchmark, or build on. This repo takes the opposite approach -- three runnable
pipelines, each addressing a documented 2026 production pain point, each with a CPU-only
fallback path so you can verify the mechanics before spending GPU hours:

| Example | Addresses | Runtime (CPU fallback) |
|---|---|---|
| [`vlm_chart_finetune`](examples/pipelines/vlm_chart_finetune/) | Efficient VLM adaptation for chart/document QA via LoRA (vision tower + LM) | ~5s |
| [`embedding_drift_active_learning`](examples/pipelines/embedding_drift_active_learning/) | Silent production drift -- the #1 cited gap in 2026 enterprise CV deployments | <1s |
| [`vlm_edge_inference`](examples/pipelines/vlm_edge_inference/) | ONNX export, INT8 quantization, and real-time edge serving with dynamic batching | ~2s |

Every example honestly labels which numbers come from a real model/GPU run versus a
CPU-only pipeline-mechanics smoke test (see [Honesty about fallback paths](#honesty-about-fallback-paths)
below) -- nothing here pretends to be a real benchmark when it isn't.

## Quickstart

```bash
git clone https://github.com/Mattral/production-vlm-engineering
cd computer-vision-playbook
make setup                       # CPU-only install (numpy/scipy/pyyaml/matplotlib/pillow + cli/dev extras)
make run-example NAME=embedding_drift_active_learning
```

For real GPU fine-tuning, ONNX export, and quantization (requires CUDA + network access to
pull checkpoints):

```bash
make setup-gpu                   # adds torch/transformers/peft/bitsandbytes/onnxruntime
make run-example NAME=vlm_chart_finetune
```

Or via the CLI directly once installed:

```bash
production-vlm list-examples
production-vlm run-example vlm_edge_inference
production-vlm benchmark embedding_drift_active_learning
```

## Repository layout

```
src/production_vlm/          Shared library code (config, drift detection, eval metrics, utils)
examples/pipelines/       The three runnable P0 examples, each with its own run.py
configs/                  YAML configs, one per example, with pinned checkpoints/dates
tests/                    pytest suite covering the shared library
scripts/                  Stdlib-only verification script (no pytest required)
docker/                   CPU and GPU Dockerfiles
legacy/                   The original markdown study notes, preserved as-is
```

## The three examples

### 1. VLM chart/document fine-tuning (`vlm_chart_finetune`)

LoRA fine-tunes a vision-language model (default: Qwen2-VL-2B, swappable) on chart
visual-question-answering, adapting both the vision tower and language model projections
rather than language-only LoRA, following 2025-2026 convention for multimodal adapters.
Training data is a zero-download synthetic chart generator (bar/line/pie charts with
ground-truth values), so the full pipeline runs with no external dataset dependency.

Evaluation uses three purpose-built metrics in `production_vlm.eval` rather than exact-match
or BLEU, which are the wrong tools for numeric chart answers: numeric accuracy (relative
tolerance matching on extracted numbers), grounding (does the answer reference terms
actually present in the chart), and a composite faithfulness score inspired by RAGAS,
adapted from retrieved-text faithfulness to chart/image evidence.

```bash
production-vlm run-example vlm_chart_finetune
```

Without a CUDA device + the `ml` extra installed, this runs a CPU-only smoke test: real
synthetic data generation and a real run through the evaluation harness, with simulated
(clearly labeled) model outputs standing in for actual generations. Install
`pip install -e ".[ml]"` and run on a GPU host for genuine fine-tuning numbers.

### 2. Embedding drift detection & active learning (`embedding_drift_active_learning`)

Implements the missing piece enterprise CV reports repeatedly flag: a model can keep
serving fine on paper (latency nominal, no errors) while the input distribution has
quietly shifted -- new camera, new lighting, a new upstream rendering pipeline -- and
accuracy degrades with no signal until someone notices downstream.

Two complementary detectors, both implemented from scratch in `production_vlm.drift`:

- **`CosineDriftDetector`** -- a two-sample Kolmogorov-Smirnov test on the distribution of
  cosine similarities between incoming embeddings and a reference centroid. Distribution-free,
  robust to the non-Gaussian shape real embedding spaces actually have.
- **`EWMADriftDetector`** -- an online, alertable statistical-process-control signal with a
  **frozen baseline standard deviation**, not a continuously-adapting one. (A naive
  continuously-adapting variance estimate is self-defeating under a real step-change: the
  jump itself inflates the variance estimate and widens the control limits just when they
  need to stay tight. This was a real bug caught and fixed during development -- see the
  detector's docstring for the full explanation.)

When drift is flagged, a simple active-learning triage ranks the batch by distance from the
reference centroid (a free, label-free novelty proxy) and queues the most novel samples for
labeling/retraining.

```bash
production-vlm run-example embedding_drift_active_learning
production-vlm benchmark embedding_drift_active_learning   # sensitivity sweep over drift magnitude
```

This example needs only numpy/scipy/matplotlib/pillow -- no GPU, no network -- via a
`SyntheticEmbeddingProxy` that derives embeddings from chart metadata rather than hashing
pixels, calibrated so injected style shifts produce a real, validated separation in
embedding space. Swap in `production_vlm.utils.vision_encoder.RealVisionEncoder` for a genuine
DINOv3/SigLIP-2/CLIP embedding space once you have GPU + network access; the
detection/active-learning code is unchanged.

The benchmark sweep is deliberately honest about sensitivity limits: at low injected-shift
magnitudes the detector reliably *fails* to trigger, and the sweep table shows exactly where
the detection boundary sits rather than picking a magnitude that always "works."

### 3. Edge inference optimization (`vlm_edge_inference`)

Exports a vision backbone to ONNX, applies ONNX Runtime dynamic INT8 quantization, and
produces a before/after latency/throughput/memory/accuracy table -- the standard "should I
ship fp32 or quantized" decision artifact. Includes a FastAPI serving stub
(`examples/pipelines/vlm_edge_inference/serve.py`) implementing the same dynamic-batching
pattern Triton/TorchServe use: requests queue and flush as a batch on
size-or-timeout-whichever-first, trading a little per-request latency for much higher
throughput under concurrent load.

```bash
production-vlm run-example vlm_edge_inference
uvicorn examples.pipelines.vlm_edge_inference.serve:app --port 8000   # requires the `serving` extra
```

Without `onnx`/`onnxruntime`/`torch`/`transformers` + network access, this benchmarks a
synthetic compute-equivalent backbone instead -- real differential timing on real matrix
multiplications, just not a real exported model. The dynamic-batching queue
(`production_vlm.utils.batching_queue.BatchingQueue`) has zero hard dependency on FastAPI and
is unit-tested directly with asyncio.

## Honesty about fallback paths

Every example is designed to run end-to-end without GPU access or network egress, because
that's the only way to keep a benchmark repo's claims checkable by anyone who clones it.
Each example detects whether the real ML stack (torch/transformers/peft, or
onnx/onnxruntime, plus a CUDA device where relevant) is actually available:

- **If yes**: it runs the real path and reports real numbers.
- **If no**: it runs a CPU-only pipeline-mechanics smoke test -- real data generation, real
  config validation, real evaluation-metric computation, real differential timing on a
  comparable-cost synthetic workload -- and prints an explicit, unmissable warning that the
  headline numbers are a smoke test, not a benchmark.

This distinction is also recorded in each example's `results.json` (`ran_with_real_ml_stack`
/ `ran_with_real_export_stack`), so downstream tooling can tell the difference programmatically.

## Testing

```bash
make test                              # pytest, requires `pip install -e ".[dev]"`
python scripts/verify_no_pytest.py     # stdlib-only fallback verifier, no pytest required
```

The stdlib-only verifier exists because some CI/sandbox environments lack network access to
install pytest; it re-checks the same core invariants (drift detection correctness, config
validation, metric correctness, batching-queue semantics) using only the standard library
plus the project's own runtime dependencies.

## Citations & reference techniques

Key techniques implemented here are documented inline with their source in each module's
docstring rather than collected in one bibliography, since the attribution is most useful
right next to the code it justifies. Notable references: LoRA (Hu et al., 2021) adapted to
both vision and language modalities; RAGAS-style faithfulness (Es et al., 2023) adapted from
retrieved-text to chart/image evidence; the two-sample Kolmogorov-Smirnov test (Massey,
1951) for distribution-free drift detection; Shewhart/EWMA statistical process control for
online drift alerting; post-training dynamic quantization (Jacob et al., 2018) for edge
inference.

## License

MIT. See [LICENSE](LICENSE).
