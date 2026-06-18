"""Seeding and lightweight run-logging helpers shared across examples."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if available) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class RunLogger:
    """Append-only JSONL run logger — the 'local' backend for TrainConfig.logging.

    Deliberately simple: one JSON object per line, no server, no
    external dependency. Swappable for a W&B logger behind the same
    interface (``log(dict)``) when ``logging: wandb`` is set.
    """

    def __init__(self, output_dir: str | Path, run_name: str = "run") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / f"{run_name}.jsonl"
        self._start = time.time()

    def log(self, record: dict) -> None:
        record = {"elapsed_s": round(time.time() - self._start, 3), **record}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]


@contextmanager
def timer(label: str = ""):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"[timer] {label}: {elapsed:.3f}s" if label else f"[timer] {elapsed:.3f}s")
