# Edge Inference & Serving

**Pipeline:** `examples/pipelines/vlm_edge_inference/`  
**Config:** `configs/vlm_edge_inference.yaml`  
**P-level:** P0-03

## What it demonstrates

Two components, addressing two genuinely different bottlenecks in VLM inference:

**Component 1 — vision encoder throughput** (three things in one benchmark):

1. **ONNX export + dynamic INT8 quantization** — the standard "should I ship fp32 or quantized?" decision artifact
2. **Before/after benchmark table** — latency (mean/p50/p95), throughput, peak memory, accuracy retention across multiple image sizes and batch sizes
3. **A production-grade serving stub** with dynamic batching

**Component 2 — language-model decoder memory** (the "memory-efficient decoding / attention optimizations" the roadmap calls out separately from the vision-encoder work above):

4. **KV-cache memory comparison** across four attention strategies (MHA / GQA / MQA / sliding-window) for the LM decoder's autoregressive generation step

## Why these matter together — and why they're different problems

ONNX export and quantization speed up the *vision encoder's* single forward pass. But a VLM also has a *language-model decoder* that generates tokens autoregressively, and for that step the binding constraint is usually **KV-cache memory**, not FLOPs — a modern VLM commonly encodes 500–1500+ visual tokens per image before generating a single word, so the KV-cache for those visual tokens dominates memory for the entire generation. Optimizing the vision encoder's export format does nothing for this; it needs a different technique entirely (attention/cache strategy), which is what Component 2 addresses.

## Component 1: The benchmark table

```
Variant        Image  Batch  Latency (ms)  Throughput  Peak Mem  Accuracy
fp32           224px  1      3.1ms         323 img/s   0.57MB    100.0%
dynamic_int8   224px  1      0.9ms         1135 img/s  0.14MB    98.4%
...
```

!!! note "Synthetic fallback numbers"
    Without `onnx`/`onnxruntime`/`torch`/`transformers` installed, the benchmark runs against a synthetic compute-equivalent backbone. The INT8 "speedup" in the fallback is modeled by reducing the effective matmul width (since numpy has no native INT8 GEMM kernel — using float16 is actually *slower*, a real gotcha that bit the first implementation here). Treat fallback numbers as illustrative of the harness, not as production benchmarks.

## Component 2: KV-cache memory-efficient decoding

`production_vlm.utils.kv_cache` computes closed-form memory footprints (pure arithmetic, no model weights needed, so this runs identically on any machine) for four attention/cache strategies:

| Strategy | Technique | Memory vs MHA baseline |
|---|---|---|
| **MHA** | Standard multi-head attention (Vaswani et al., 2017) | 1.00× (baseline) |
| **GQA** | Grouped-query attention (Ainslie et al., 2023) — query heads share K/V heads | ~0.14× (7× smaller, for a 28→4 head ratio) |
| **MQA** | Multi-query attention (Shazeer, 2019) — single shared K/V head | ~0.04× (28× smaller) |
| **Sliding window** | Bounded local context (Beltagy et al., 2020-style) | O(1) in sequence length — no benefit until the true sequence exceeds the window, then increasingly dominant |

The sliding-window result is the most instructive: at `seq_len` below the window size it provides **zero** savings (matches MHA exactly), then its relative advantage grows without bound as the sequence continues to grow, since its absolute memory is capped while MHA's keeps scaling linearly. Verified directly: identical absolute memory at `seq_len=512` and `seq_len=2000` for a `sliding_window_size=512` configuration.

```python
from production_vlm.utils.kv_cache import ModelDecoderConfig, AttentionStrategy, compute_kv_cache_memory

cfg = ModelDecoderConfig(n_layers=28, n_query_heads=28, n_kv_heads_gqa=4)
result = compute_kv_cache_memory(cfg, AttentionStrategy.GQA, seq_len=1576)
print(f"{result.kv_cache_mb:.1f} MB ({result.relative_to_mha:.1%} of MHA)")
```

This connects to the broader efficient-decoding landscape via FlashAttention-2 (Dao, 2023 — compute/memory-efficient *exact* attention, orthogonal to cache strategy and applicable to all four above), PagedAttention (Kwon et al., 2023, the vLLM paper — efficient memory *management* of whichever strategy is chosen), and speculative decoding (Leviathan et al., 2023; Chen et al., 2023 — reducing forward passes per token, a complementary axis to cache-size reduction).

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
production-vlm run-example vlm_edge_inference
```

For real ONNX export and quantization (requires the `ml` + `onnx` extras and network access):

```bash
pip install -e ".[ml,onnx]"
production-vlm run-example vlm_edge_inference
```
