"""Pytest configuration: make `src/` and the repo root importable without installation.

This lets `pytest` run directly against the source tree (e.g. in CI
before `pip install -e .` has happened, or in this repo's own
sandboxed verification) without requiring the package to be built.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

for path in (str(REPO_ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)
