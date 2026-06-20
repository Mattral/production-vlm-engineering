# Edge Inference & Serving

**Pipeline:** `examples/pipelines/vlm_edge_inference/`  
**Config:** `configs/vlm_edge_inference.yaml`  
**P-level:** P0-03

## What it demonstrates

Three things in one pipeline:

1. **ONNX export + dynamic INT8 quantization** — the standard "should I ship fp32 or quantized?" decision artifact
2. **Before/after benchmark table** — latency (mean/p50/p95), throughput, peak memory, accuracy retention across multiple image sizes and batch sizes
3. **A production-grade serving stub** with dynamic batching

## Why these matter together

ONNX export and quantization are well-documented individually. What's less covered is the *decision workflow*: export both variants, run the same benchmark harness on both, read the table, decide. This pipeline makes that workflow repeatable and scriptable.

## The benchmark table

```
Variant        Image  Batch  Latency (ms)  Throughput  Peak Mem  Accuracy
fp32           224px  1      3.1ms         323 img/s   0.57MB    100.0%
dynamic_int8   224px  1      0.9ms         1135 img/s  0.14MB    98.4%
...
```

!!! note "Synthetic fallback numbers"
    Without `onnx`/`onnxruntime`/`torch`/`transformers` installed, the benchmark runs against a synthetic compute-equivalent backbone. The INT8 "speedup" in the fallback is modeled by reducing the effective matmul width (since numpy has no native INT8 GEMM kernel — using float16 is actually *slower*, a real gotcha that bit the first implementation here). Treat fallback numbers as illustrative of the harness, not as production benchmarks.

## Dynamic batching queue

The FastAPI serving stub (`serve.py`) uses `production_vlm.utils.batching_queue.BatchingQueue` — a dependency-free asyncio primitive tested independently:

```python
from production_vlm.utils.batching_queue import BatchingQueue

queue = BatchingQueue(predict_fn, max_batch_size=8, max_batch_wait_ms=50)
queue.start()

# From any async handler:
output, batch_size, wait_ms, inf_ms = await queue.submit(image_array)
```

Requests queue and flush as a batch when either `max_batch_size` is reached or `max_batch_wait_ms` elapses, whichever comes first. A single-item request at low traffic waits at most `max_batch_wait_ms` before being served alone — verified in `tests/test_batching_queue.py`.

## Run the server

```bash
pip install -e ".[serving]"
uvicorn examples.pipelines.vlm_edge_inference.serve:app --port 8000

# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"$(base64 -w0 chart.png)\"}"
```

## Run it

```bash
cv-playbook run-example vlm_edge_inference
```

For real ONNX export and quantization (requires the `ml` + `onnx` extras and network access):

```bash
pip install -e ".[ml,onnx]"
cv-playbook run-example vlm_edge_inference
```
