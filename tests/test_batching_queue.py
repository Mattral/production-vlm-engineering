"""Unit tests for cv_playbook.utils.batching_queue.BatchingQueue."""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from cv_playbook.utils.batching_queue import BatchingQueue


def _sum_predict_fn(batch: np.ndarray) -> np.ndarray:
    return batch.sum(axis=1, keepdims=True) * np.ones((1, 4))


class TestBatchingQueue:
    def test_rejects_invalid_max_batch_size(self):
        with pytest.raises(ValueError):
            BatchingQueue(_sum_predict_fn, max_batch_size=0, max_batch_wait_ms=10)

    def test_rejects_negative_wait_ms(self):
        with pytest.raises(ValueError):
            BatchingQueue(_sum_predict_fn, max_batch_size=4, max_batch_wait_ms=-1)

    @pytest.mark.asyncio
    async def test_flushes_on_max_batch_size(self):
        queue = BatchingQueue(_sum_predict_fn, max_batch_size=4, max_batch_wait_ms=1000)
        queue.start()

        async def submit_one(i):
            return await queue.submit(np.full((3,), i, dtype=np.float32))

        results = await asyncio.gather(*[submit_one(i) for i in range(4)])
        await queue.stop()

        for _, batch_size, *_ in results:
            assert batch_size == 4
        assert queue.batches_served == 1
        assert queue.items_served == 4

    @pytest.mark.asyncio
    async def test_flushes_on_timeout_with_partial_batch(self):
        queue = BatchingQueue(_sum_predict_fn, max_batch_size=10, max_batch_wait_ms=30)
        queue.start()

        start = time.perf_counter()
        output, batch_size, queue_wait_ms, inference_ms = await queue.submit(np.full((3,), 1.0, dtype=np.float32))
        elapsed_ms = (time.perf_counter() - start) * 1000

        await queue.stop()

        assert batch_size == 1
        assert elapsed_ms >= 25  # should have waited roughly max_batch_wait_ms before flushing alone

    @pytest.mark.asyncio
    async def test_multiple_batches_formed_from_burst(self):
        queue = BatchingQueue(_sum_predict_fn, max_batch_size=3, max_batch_wait_ms=200)
        queue.start()

        async def submit_one(i):
            return await queue.submit(np.full((2,), i, dtype=np.float32))

        results = await asyncio.gather(*[submit_one(i) for i in range(7)])
        await queue.stop()

        assert queue.items_served == 7
        assert queue.batches_served == 3

    @pytest.mark.asyncio
    async def test_output_correctness_per_item(self):
        queue = BatchingQueue(_sum_predict_fn, max_batch_size=4, max_batch_wait_ms=50)
        queue.start()

        async def submit_one(value):
            arr = np.full((5,), value, dtype=np.float32)
            output, *_ = await queue.submit(arr)
            return value, output

        results = await asyncio.gather(*[submit_one(v) for v in [1.0, 2.0, 3.0]])
        await queue.stop()

        for value, output in results:
            expected_sum = value * 5
            assert np.allclose(output, expected_sum)

    @pytest.mark.asyncio
    async def test_stop_is_idempotent_safe(self):
        queue = BatchingQueue(_sum_predict_fn, max_batch_size=4, max_batch_wait_ms=10)
        queue.start()
        await queue.stop()
        # Calling stop a second time without a running task should not raise
        await queue.stop()
