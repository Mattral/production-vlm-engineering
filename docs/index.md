# Production VLM Engineering

**Reproducible, production-grade pipelines for multimodal vision systems.**

Four runnable examples covering the full lifecycle of a production VLM deployment —
fine-tuning, edge inference, drift monitoring, and safety/robustness testing — each with a
CPU-only fallback path so any claim in this repo can be checked without GPU access.

## Quickstart

```bash
git clone https://github.com/Mattral/production-vlm-engineering
cd production-vlm-engineering
make setup                              # CPU-only install (~15 seconds)
production-vlm list-examples               # list all five examples
production-vlm run-example embedding_drift_active_learning   # fastest (<1s)
python benchmarks/run_all.py            # run all five, generate unified benchmark report
```

## Four examples

| Example | What it shows | CPU runtime |
|---|---|---|
| `vlm_chart_finetune` | LoRA fine-tuning (vision tower + LM), zero-download synthetic chart-QA data, grounding/faithfulness metrics | ~5s |
| `embedding_drift_active_learning` | KS-test + EWMA drift detection, label-free active learning triage, sensitivity sweep | <1s |
| `vlm_edge_inference` | ONNX export, INT8 quantization, real-time serving with dynamic batching queue | ~2s |
| `vlm_robustness_guard` | ImageNet-C-style perturbation sweep, kNN OOD detection, hallucination guard | ~6s |

## Latest benchmark results

See [Benchmark Report](benchmark_report.md) for headline numbers from the most recent
`python benchmarks/run_all.py` run. Numbers marked ⚠️ are CPU smoke-test values;
see individual example READMEs for GPU reproduction instructions.

## Repository layout

```
src/production_vlm/        Shared library: config, drift, eval, robustness, utils
examples/pipelines/        Four runnable examples, one directory each
configs/                   YAML configs with pinned checkpoints and dates
tests/                     pytest suite (40 tests)
scripts/                   stdlib-only fallback verifier (no pytest needed)
benchmarks/                Unified runner + generated reports
docker/                    CPU and GPU Dockerfiles
docs/                      This site
```

## Honesty policy

Every fallback path that runs without a GPU or network is clearly labeled — in console
output, in each `results.json` (`ran_with_real_ml_stack`), and in this documentation.
CPU smoke-test numbers and real GPU numbers are never presented interchangeably.

Six real bugs were found and fixed by running the code rather than only reading it.
All are documented in-place. See [ROADMAP](../ROADMAP.md) for the full list.
