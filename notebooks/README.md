# Notebooks

Interactive walkthroughs of the core techniques in this repo. All notebooks use only the CPU-only core dependencies (numpy/scipy/matplotlib/pillow) — no GPU or ML stack required unless noted.

Notebooks include pre-executed output cells so they render correctly on GitHub without needing to run first.

## Contents

| Notebook | What it covers | Dependencies |
|---|---|---|
| [`01_evaluation_metrics.ipynb`](01_evaluation_metrics.ipynb) | `numeric_accuracy`, `grounding_score`, `faithfulness_score` — why BLEU/exact-match fail on chart answers and how these alternatives work | CPU core only |
| [`02_drift_detection_active_learning.ipynb`](02_drift_detection_active_learning.ipynb) | `CosineDriftDetector`, `EWMADriftDetector` with frozen baseline SPC, `select_for_active_learning` | CPU core only |
| [`03_robustness_safety_guard.ipynb`](03_robustness_safety_guard.ipynb) | Perturbation sweep, `KNNOODDetector` calibration, `HallucinationGuard` three-tier decisions, production wrapper pattern | CPU core only |

## Running

```bash
# Install Jupyter if not already present
pip install jupyter

# Launch
jupyter notebook notebooks/

# Or in VS Code: just open the .ipynb file
```

All paths use `sys.path.insert(0, '../src')` — run them from the `notebooks/` directory or adjust the path for your setup.
