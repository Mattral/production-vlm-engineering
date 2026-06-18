# vlm_edge_inference

ONNX export, INT8 quantization, latency/throughput/memory benchmarking, and a FastAPI
serving stub with dynamic batching.

## What this demonstrates

- Export to ONNX and post-training **dynamic INT8 quantization** via ONNX Runtime
  (Jacob et al., 2018-style integer-arithmetic inference).
- A real before/after benchmark table: latency (mean/p50/p95), throughput, peak memory, and
  accuracy retention, across multiple image sizes and batch sizes.
- A serving layer (`serve.py`) implementing the same **dynamic batching** pattern Triton
  Inference Server and TorchServe use: requests queue and flush as a batch when either
  `max_batch_size` is reached or `max_batch_wait_ms` elapses, whichever comes first.

## Run it

```bash
cv-playbook run-example vlm_edge_inference
```

Serve it (requires the `serving` extra: `pip install -e ".[serving]"`):

```bash
uvicorn examples.pipelines.vlm_edge_inference.serve:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
    -d "{\"image_base64\": \"$(base64 -w0 some_image.png)\"}"
```

## What you'll see

Without `onnx`/`onnxruntime`/`torch`/`transformers` installed (the `ml` + `onnx` extras)
and network access to pull `model.checkpoint`, the benchmark runs against a synthetic
compute-equivalent backbone instead: real differential timing on real matrix multiplications
sized to approximate a small transformer encoder's per-token cost, not a real exported
model. This is clearly labeled in the table title and in `results.json`
(`ran_with_real_export_stack: false`).

A note on the synthetic fallback's quantization model: numpy has no native INT8 GEMM kernel,
and naively casting to `float16` is actually *slower* than `float32` on CPU (no accelerated
BLAS path), which would produce a backwards, misleading benchmark. Instead the synthetic
"quantized" variant reduces the effective matmul width by a fixed factor calibrated to the
real-world ballpark for ONNX Runtime dynamic INT8 speedups on CPU (commonly 2-4x for
transformer encoders, hardware/op-mix dependent). Treat the resulting ratio as illustrative
of the *harness*, not as a substitute for the real `benchmark_onnx_session` measurement.

With the real stack installed, `export_real_model_to_onnx` and `quantize_real_model` in
`run.py` produce a genuine `.onnx` file pair and `benchmark_onnx_session` measures both with
ONNX Runtime's `CPUExecutionProvider`.

## The dynamic batching queue

`cv_playbook.utils.batching_queue.BatchingQueue` has zero hard dependency on FastAPI and is
unit-tested directly with asyncio (`tests/test_batching_queue.py`) -- import it into any
other asyncio-based serving layer.

## Files

- `run.py` -- export, quantization, benchmark harness, synthetic fallback backbone.
- `serve.py` -- FastAPI app using `BatchingQueue` for request batching.
- `../../../src/cv_playbook/utils/batching_queue.py` -- the reusable batching primitive.
- `../../../configs/vlm_edge_inference.yaml` -- model/export/quantization/benchmark/serving config.
