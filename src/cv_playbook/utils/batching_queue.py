"""Dynamic batching queue for serving layers: stdlib asyncio + numpy only.

Extracted from the FastAPI serving stub so the core batching logic
(flush-on-size-or-timeout) is unit-testable without pulling in
fastapi/uvicorn/pydantic, and reusable by any serving frontend.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import numpy as np


class BatchingQueue:
    """Accumulates single-item requests into batches, flushed by size or time.

    Each call to `submit()` enqueues one item and awaits its own
    future; a single background task drains the queue, building
    batches up to `max_batch_size` or waiting at most
    `max_batch_wait_ms`, runs one model call per batch, and resolves
    each item's future with its slice of the batched output. This
    mirrors the core pattern Triton's dynamic batcher and TorchServe's
    batch predictor use, simplified to single-process asyncio.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        max_batch_size: int,
        max_batch_wait_ms: float,
        poll_interval_s: float = 0.001,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_batch_wait_ms < 0:
            raise ValueError("max_batch_wait_ms must be >= 0")
        self._predict_fn = predict_fn
        self.max_batch_size = max_batch_size
        self.max_batch_wait_ms = max_batch_wait_ms
        self.poll_interval_s = poll_interval_s
        self._queue: list[tuple[np.ndarray, asyncio.Future, float]] = []
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None
        self.batches_served = 0
        self.items_served = 0

    def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, array: np.ndarray) -> tuple[np.ndarray, int, float, float]:
        """Enqueue one item; returns (output, batch_size_served_with, queue_wait_ms, inference_ms)."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        submit_time = time.perf_counter()
        async with self._lock:
            self._queue.append((array, future, submit_time))
        return await future

    async def _worker_loop(self) -> None:
        while True:
            await asyncio.sleep(self.poll_interval_s)
            async with self._lock:
                if not self._queue:
                    continue
                oldest_wait_ms = (time.perf_counter() - self._queue[0][2]) * 1000
                should_flush = len(self._queue) >= self.max_batch_size or oldest_wait_ms >= self.max_batch_wait_ms
                if not should_flush:
                    continue
                batch_items = self._queue[: self.max_batch_size]
                self._queue = self._queue[self.max_batch_size :]

            arrays = np.stack([item[0] for item in batch_items])
            t0 = time.perf_counter()
            outputs = self._predict_fn(arrays)
            inference_ms = (time.perf_counter() - t0) * 1000

            self.batches_served += 1
            self.items_served += len(batch_items)

            for i, (_, future, submit_time) in enumerate(batch_items):
                queue_wait_ms = (t0 - submit_time) * 1000
                if not future.done():
                    future.set_result((outputs[i], len(batch_items), queue_wait_ms, inference_ms))
