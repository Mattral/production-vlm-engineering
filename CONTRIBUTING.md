# Contributing

Thanks for considering a contribution. This repo prioritizes runnable correctness over
breadth -- a small set of examples that genuinely work, are tested, and are honest about
their limitations beats a large set of untested ones.

## Setup

```bash
git clone https://github.com/Mattral/production-vlm-engineering
cd production-vlm-engineering
make setup            # CPU-only core; add `make setup-gpu` for the full ML stack
```

## Before opening a PR

```bash
make lint
make test                              # requires `pip install -e ".[dev]"`
python scripts/verify_no_pytest.py     # stdlib-only fallback check
```

If you're changing one of the three example pipelines, actually run it (`cv-playbook
run-example <name>`) and check the printed table/`results.json` make sense -- don't just
read the diff. Several real bugs in this repo's own history (see `ROADMAP.md`'s "Design
decisions worth recording" section) were only caught by running the code, not by reading it.

## Adding a new example

1. Create `examples/pipelines/<name>/` with `run.py` (a `main(config_path=None)` entry point)
   and a matching `configs/<name>.yaml`.
2. Register it in `_EXAMPLES` in `src/cv_playbook/cli.py`.
3. If it needs a GPU/network-dependent library, gate the real path behind a runtime
   availability check (see `_has_real_ml_stack()` in `vlm_chart_finetune/run.py` for the
   pattern) and provide an honest, clearly-labeled CPU fallback that still exercises real
   code paths (data generation, config validation, metric computation) rather than just
   printing a "not implemented" message.
4. Add a `README.md` alongside `run.py` documenting what it demonstrates, how to run it, and
   what the fallback path does differently from the real path.
5. Add tests under `tests/` for any new shared library code in `src/cv_playbook/`.

## Code style

- `ruff` for linting/formatting (config in `pyproject.toml`); pre-commit hooks are provided
  in `.pre-commit-config.yaml`.
- Keep `src/cv_playbook`'s core import surface CPU-only and dependency-light (numpy/scipy/
  pyyaml/matplotlib/pillow). Anything requiring torch/transformers/onnx/fastapi goes behind
  an optional extra in `pyproject.toml` and a runtime availability check, not a hard import
  at module load time.
- Cite techniques inline in the docstring of the function/class that implements them, not in
  a separate bibliography file -- attribution is most useful right next to the code it
  justifies.

## Reporting issues

Please include whether you're running with the `ml`/`onnx`/`serving` extras installed and a
GPU available, since several behaviors (real vs. fallback path) depend on it.
