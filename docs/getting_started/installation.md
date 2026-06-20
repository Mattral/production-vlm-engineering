# Installation

## Dependency tiers

The package is split into tiers so you only install what you need. The core library (`numpy`, `scipy`, `pyyaml`, `matplotlib`, `pillow`) has no hard dependency on PyTorch, transformers, or any ML framework — this was a deliberate decision that lets the drift detection, evaluation metrics, and robustness modules run in any environment, including lightweight CI runners and edge devices.

| Tier | Command | What it enables |
|---|---|---|
| **Core** (always) | `pip install -e .` | All shared library code, all four CPU smoke-test paths |
| **CLI** | `pip install -e ".[cli]"` | `rich` table output, `typer` CLI (gracefully degrades without this) |
| **ML** (GPU) | `pip install -e ".[ml]"` | Real LoRA fine-tuning, real vision encoder inference |
| **ONNX** | `pip install -e ".[onnx]"` | Real ONNX export and quantization |
| **Serving** | `pip install -e ".[serving]"` | FastAPI serving stub |
| **Dev** | `pip install -e ".[dev]"` | pytest, ruff, pre-commit, mkdocs |

## Recommended setups

=== "CPU / CI / offline"
    ```bash
    make setup
    # → installs core + cli + dev extras
    # → all four examples run in smoke-test mode
    # → 40 tests pass, verify_no_pytest.py passes
    ```

=== "GPU fine-tuning"
    ```bash
    make setup-gpu
    # → installs core + cli + ml + onnx + serving + dev
    # → real LoRA fine-tuning in vlm_chart_finetune
    # → real ONNX export + quantization in vlm_edge_inference
    # Requires: CUDA 12.1+, ≥12GB VRAM for the default 2B checkpoint
    ```

=== "Docker (CPU)"
    ```bash
    make docker-build
    docker run --rm cv-playbook:latest
    # Runs embedding_drift_active_learning smoke test by default
    ```

=== "Docker (GPU)"
    ```bash
    make docker-build-gpu
    docker run --gpus all --rm cv-playbook:gpu
    # Requires NVIDIA Container Toolkit
    ```

## Verifying your install

```bash
# Check all 40 tests pass
make test

# Or without pytest (offline/CI fallback)
python scripts/verify_no_pytest.py

# Check the CLI works
cv-playbook list-examples
```

## Model checkpoints

The default VLM checkpoint (`Qwen/Qwen2-VL-2B-Instruct`, pinned `2026-03-01`) is only pulled when running the real GPU path. The CPU smoke-test paths generate their own synthetic data and use a proxy backbone — no network egress required. If you switch checkpoints, update `checkpoint_pinned_date` in the relevant config YAML so the date of verification is recorded alongside the model name.

## Python version support

Python 3.10, 3.11, and 3.12 are tested in CI (see `.github/workflows/ci.yml`). Python 3.9 is not supported (`match`/`case` syntax is not used but some type-annotation syntax requires 3.10+).
