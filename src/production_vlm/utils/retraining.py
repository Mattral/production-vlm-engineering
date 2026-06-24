"""Drift-triggered retraining feedback loop.

Implements the "integrate with a training example (trigger retraining
simulation on drifted data)" requirement from P0-04 of the roadmap.

``RetrainingTrigger`` maintains a queue of samples flagged by the drift
detector or active-learning selector, and fires a retraining callback
when the queue reaches a configured threshold — simulating the human-
in-the-loop or pseudo-labeling retraining pattern that production
ML systems use to close the distribution shift → model update loop.

Design pattern: the trigger is stateful and thread-safe, intended to
be shared between the drift monitoring thread (or async task) and a
retraining worker (or scheduled job). The callback is a plain callable
rather than a specific training framework API so the trigger composes
with any fine-tuning backend — the P0-02 `train_real()` function, a
HuggingFace Trainer, or a distributed training cluster.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class QueuedSample:
    """A sample flagged for labeling/retraining, with provenance metadata."""
    embedding_index: int
    batch_idx: int
    novelty_score: float           # distance-from-centroid proxy for labeling priority
    flagged_by: str                # "drift_ks" | "drift_ewma" | "ood" | "active_learning"
    timestamp_utc: float = field(default_factory=time.time)
    pseudo_label: Any = None       # set by a labeling callback when available


@dataclass
class RetrainingRun:
    """Record of one retraining trigger invocation."""
    trigger_id: int
    n_samples: int
    triggered_at_utc: float
    trigger_reason: str
    success: bool
    error: str = ""
    duration_s: float = 0.0


class RetrainingTrigger:
    """Queues drifted/OOD samples and triggers retraining when the queue fills.

    Typical usage in a drift monitoring loop::

        from production_vlm.utils.retraining import RetrainingTrigger, QueuedSample

        def my_retraining_callback(samples: list[QueuedSample]) -> None:
            # In production: kick off a fine-tuning job with these samples
            # Here: just log that it would have happened
            print(f"Would retrain on {len(samples)} samples")

        trigger = RetrainingTrigger(
            queue_threshold=20,
            callback=my_retraining_callback,
            cooldown_s=300,      # minimum seconds between retraining runs
        )

        # In the drift loop:
        if drift_result.is_drift:
            novel_indices = select_for_active_learning(...)
            for idx in novel_indices:
                trigger.enqueue(QueuedSample(
                    embedding_index=idx, batch_idx=batch_idx,
                    novelty_score=..., flagged_by="drift_ks",
                ))
    """

    def __init__(
        self,
        queue_threshold: int = 20,
        callback: Callable[[list[QueuedSample]], None] | None = None,
        cooldown_s: float = 300.0,
        max_queue_size: int = 1000,
    ) -> None:
        if queue_threshold < 1:
            raise ValueError("queue_threshold must be >= 1")
        self.queue_threshold = queue_threshold
        self.callback = callback or _default_callback
        self.cooldown_s = cooldown_s
        self._queue: deque[QueuedSample] = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()
        self._last_trigger_at: float = 0.0
        self._trigger_count: int = 0
        self._history: list[RetrainingRun] = []

    def enqueue(self, sample: QueuedSample) -> bool:
        """Add a sample to the retraining queue. Returns True if a retraining run was triggered."""
        with self._lock:
            self._queue.append(sample)
            if self._should_trigger():
                self._fire()
                return True
        return False

    def enqueue_batch(self, samples: list[QueuedSample]) -> bool:
        """Batch-enqueue multiple samples. Fires as many times as the threshold is crossed.

        Unlike a single ``enqueue()`` call, a large batch can cross the
        queue threshold multiple times -- e.g., 9 samples at threshold=3
        fires 3 times with 3 samples each. This keeps batch-enqueue
        semantically equivalent to calling ``enqueue()`` in a loop.
        """
        fired = False
        with self._lock:
            for s in samples:
                self._queue.append(s)
            # Drain and fire as many full batches as the queue allows.
            while self._should_trigger():
                self._fire()
                fired = True
        return fired

    def _should_trigger(self) -> bool:
        if len(self._queue) < self.queue_threshold:
            return False
        if time.time() - self._last_trigger_at < self.cooldown_s:
            return False
        return True

    def _fire(self) -> None:
        """Called with _lock held. Drains exactly `queue_threshold` items and invokes the callback."""
        samples = [self._queue.popleft() for _ in range(min(self.queue_threshold, len(self._queue)))]
        self._last_trigger_at = time.time()
        self._trigger_count += 1
        trigger_id = self._trigger_count

        t0 = time.time()
        try:
            self.callback(samples)
            self._history.append(RetrainingRun(
                trigger_id=trigger_id,
                n_samples=len(samples),
                triggered_at_utc=t0,
                trigger_reason=f"queue_threshold={self.queue_threshold} reached",
                success=True,
                duration_s=time.time() - t0,
            ))
        except Exception as e:
            self._history.append(RetrainingRun(
                trigger_id=trigger_id,
                n_samples=len(samples),
                triggered_at_utc=t0,
                trigger_reason=f"queue_threshold={self.queue_threshold} reached",
                success=False,
                error=str(e),
                duration_s=time.time() - t0,
            ))

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def trigger_count(self) -> int:
        """Total number of retraining runs fired."""
        with self._lock:
            return self._trigger_count

    @property
    def trigger_history(self) -> list[RetrainingRun]:
        with self._lock:
            return list(self._history)

    def summary(self) -> dict:
        with self._lock:
            return {
                "trigger_count": self._trigger_count,
                "queue_size": len(self._queue),
                "last_trigger_utc": self._last_trigger_at or None,
                "runs": [
                    {
                        "trigger_id": r.trigger_id,
                        "n_samples": r.n_samples,
                        "success": r.success,
                        "duration_s": round(r.duration_s, 3),
                    }
                    for r in self._history
                ],
            }


def _default_callback(samples: list[QueuedSample]) -> None:
    """Default no-op callback: logs what a real callback would do.

    Replace with a real fine-tuning trigger in production — for example,
    calling the `train_real()` function in `vlm_chart_finetune/run.py`
    with these samples, or submitting a training job to a cluster.
    """
    print(
        f"[RetrainingTrigger] Would trigger retraining on {len(samples)} samples "
        f"(sorted by novelty: {sorted(s.novelty_score for s in samples)[:3]}...). "
        f"Wire a real callback to this trigger for production use."
    )
