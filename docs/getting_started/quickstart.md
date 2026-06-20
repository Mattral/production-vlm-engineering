# Quickstart

Get your first result in under two minutes.

## Prerequisites

- Python 3.10+
- No GPU required for the smoke-test paths

## Install

```bash
git clone https://github.com/Mattral/production-vlm-engineering
cd production-vlm-engineering
make setup
```

`make setup` creates a virtual environment and installs the CPU-only core plus the CLI and dev extras. The full ML stack (torch, transformers, peft, bitsandbytes) is optional — install it separately with `make setup-gpu` when you have a CUDA device.

## Run your first example

```bash
# Fastest example: embedding drift detection (<1 second)
cv-playbook run-example embedding_drift_active_learning

# See what else is available
cv-playbook list-examples
```

## Run all examples and generate a benchmark report

```bash
python benchmarks/run_all.py
# → benchmarks/reports/benchmark_report.md
# → benchmarks/reports/benchmark_report.json
```

## What you'll see

Each example prints a clear table showing whether it ran the real GPU path or the CPU smoke-test fallback, then writes `outputs/<example_name>/results.json`. The benchmark runner collects all four `results.json` files into a single Markdown report.

!!! note "CPU vs GPU numbers"
    Numbers from the CPU smoke-test path are labeled ⚠️ in both console output and `results.json`. They verify the pipeline mechanics (data generation, config validation, evaluation harness, timing harness) but are not representative of what you'd see on a GPU with real model weights. See [Installation](installation.md) for the GPU setup path.

## Next steps

- [Architecture overview](architecture.md) — how the four examples and shared library fit together
- [Example deep-dives](../examples/overview.md) — what each example demonstrates and how to adapt it
- [Benchmark results](../benchmark_report.md) — headline numbers from the latest run
