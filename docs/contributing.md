# Contributing

See [`CONTRIBUTING.md`](https://github.com/Mattral/production-vlm-engineering/blob/main/CONTRIBUTING.md)
in the repo root for setup instructions and the full contribution guide.

## Key principles for contributors

**Run the code.** Every real bug in this repo was found by running the code, not by reading it. Before opening a PR for a new example or a change to an existing pipeline, actually run it (`production-vlm run-example <name>`) and check the output makes sense.

**Document bugs in-place.** When you find and fix a bug, leave a clear comment explaining the failure mode alongside the fix. The failure modes here are instructive (variance contamination in EWMA, absolute-threshold brittleness under contrast perturbation, matplotlib axes-vs-figure coordinate confusion) and worth preserving rather than quietly rewriting.

**Keep the CPU-only fallback honest.** If an example runs differently without a GPU or ML stack, that difference must be unambiguously labeled — in console output, in `results.json`, and in the docs. The whole trust model of this repo depends on never presenting smoke-test numbers as benchmark numbers.

**Calibrate thresholds empirically, then document the sweep.** Don't pick a threshold by intuition. Sweep it, measure FP/TP, pick the right operating point for the use case, and document the sweep in the code (like `KNNOODDetector`'s docstring) and in the relevant concept page.

## Adding a new example

1. Create `examples/pipelines/<name>/` with `run.py` and `README.md`
2. Add a YAML config under `configs/<name>.yaml` with pinned checkpoint dates
3. Register in `src/production_vlm/cli.py` and in `benchmarks/run_all.py` (add an `ExampleSpec`)
4. Add tests under `tests/`
5. Add a doc page under `docs/examples/<name>.md` and register it in `mkdocs.yml`
6. Run `python benchmarks/run_all.py` and confirm the new example appears in the report
