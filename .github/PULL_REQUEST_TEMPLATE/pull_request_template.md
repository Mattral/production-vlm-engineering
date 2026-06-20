# Pull Request

## What this changes

Brief description of the change. If it adds a new example, state the headline metric and CPU runtime.

## Checklist

### Required for all PRs
- [ ] `python scripts/verify_no_pytest.py` passes (40 checks, no pytest needed)
- [ ] `make test` passes (if pytest is available in your environment)
- [ ] `make lint` passes (ruff check + format)
- [ ] `cv-playbook list-examples` still lists all expected examples
- [ ] All existing `results.json` outputs look sane after change

### Required for new or changed examples
- [ ] I actually ran the example (`cv-playbook run-example <name>`) and checked the output
- [ ] CPU-fallback behavior is clearly labeled in console output and in `results.json`
- [ ] Any new threshold or parameter was calibrated empirically, not chosen by intuition
- [ ] The calibration sweep is documented in the code (docstring or inline comment)
- [ ] `README.md` benchmark table updated if headline metrics changed
- [ ] Example registered in `src/production_vlm/cli.py` and `benchmarks/run_all.py`
- [ ] Per-example `README.md` written or updated
- [ ] Docs page added/updated under `docs/examples/`

### Required for new techniques / modules
- [ ] Inline citation to the specific paper in the docstring of the implementing function/class
- [ ] Added to `docs/concepts/` if it introduces a new technique

### For bugs / correctness fixes
- [ ] Description explains the failure mode, not just the fix (these are instructive for future contributors)
- [ ] Regression test added that would have caught the bug

## How I tested this

```bash
# Commands you actually ran
```

## Before / after metrics (if relevant)

| Metric | Before | After |
|---|---|---|
| | | |

## Notes for reviewer

Anything that needs explanation or where you made a judgment call.
