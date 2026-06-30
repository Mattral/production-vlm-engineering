# Colab Notebooks

Interactive notebooks designed for Google Colab — no local setup required. Each notebook installs the `production-vlm-engineering` package automatically at startup and runs end-to-end in 1–3 minutes.

| Notebook | Topic | Open |
|---|---|---|
| 01 · Evaluation Metrics | `numeric_accuracy`, `grounding_score`, `faithfulness_score` — why BLEU fails on chart answers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/production-vlm-engineering/blob/main/notebooks/colab/01_evaluation_metrics_colab.ipynb) |
| 02 · Drift Detection | `CosineDriftDetector`, `EWMADriftDetector` (frozen-baseline SPC), active learning triage | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/production-vlm-engineering/blob/main/notebooks/colab/02_drift_detection_colab.ipynb) |
| 03 · Robustness & Safety | Perturbation sweep, kNN OOD detection, hallucination guard, production wrapper | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/production-vlm-engineering/blob/main/notebooks/colab/03_robustness_guard_colab.ipynb) |

## No GPU required

All three notebooks use only the CPU-only core dependencies. The package is installed directly from GitHub at the start of each notebook.

## Local notebooks (pre-executed)

The [`../`](../) directory contains the same three notebooks in a pre-executed form — all output cells are populated so they render correctly on GitHub without running anything:

- [`01_evaluation_metrics.ipynb`](../01_evaluation_metrics.ipynb)
- [`02_drift_detection_active_learning.ipynb`](../02_drift_detection_active_learning.ipynb)
- [`03_robustness_safety_guard.ipynb`](../03_robustness_safety_guard.ipynb)
