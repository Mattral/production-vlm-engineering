#!/usr/bin/env python
"""Minimal FastAPI serving layer with dynamic batching for the exported vision model.

Implements the "simple serving (FastAPI or Triton-like) with batching"
requirement of P0-03. Follows the standard dynamic-batching pattern
used by Triton Inference Server / TorchServe: incoming requests are
queued and flushed as a batch either when `max_batch_size` requests
have accumulated or `max_batch_wait_ms` has elapsed, whichever comes
first -- this trades a small amount of added latency per request for
much higher throughput under concurrent load, which is the right
tradeoff for most real-time CV serving workloads.

Run:
    pip install -e ".[ml,onnx,serving]"
    uvicorn examples.pipelines.vlm_edge_inference.serve:app --host 0.0.0.0 --port 8000

Then:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \\
        -d '{"image_base64": "<base64-encoded image bytes>"}'

This module intentionally has a hard runtime dependency on `fastapi`
+ `uvicorn` (the `serving` extra) since it is a server entry point,
not a library import other examples might pull in -- it is not
imported by `production_vlm.cli` and therefore never breaks the
no-network-stack fallback paths used by the other examples' smoke
tests.
"""

from __future__ import annotations

import base64
import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from production_vlm.utils.batching_queue import BatchingQueue  # noqa: E402
from production_vlm.utils.console import Console  # noqa: E402

console = Console()
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "vlm_edge_inference.yaml"


class PredictRequest(BaseModel):
    image_base64: str


class PredictResponse(BaseModel):
    embedding_preview: list[float]
    batch_size_served_with: int
    queue_wait_ms: float
    inference_ms: float


def _load_predict_fn():
    """Returns a `(np.ndarray batch) -> np.ndarray` callable.

    Uses the real ONNX Runtime session if an exported model is found
    on disk (produced by `run.py`'s real export path); otherwise falls
    back to the same synthetic backbone used by the benchmark, so this
    serving stub is exercisable end-to-end without GPU/network too.
    """
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    onnx_dir = Path(cfg["export"]["output_dir"])
    fp32_path = onnx_dir / "model_fp32.onnx"

    if fp32_path.exists():
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name

            def predict(batch: np.ndarray) -> np.ndarray:
                out = session.run(None, {input_name: batch})[0]
                return out.reshape(out.shape[0], -1)

            console.print(f"[green]Serving real ONNX model from {fp32_path}[/green]")
            return predict
        except ImportError:
            pass

    console.print(
        "[yellow]No exported ONNX model / onnxruntime found -- "
        "serving with the synthetic benchmark backbone.[/yellow]"
    )
    from examples.pipelines.vlm_edge_inference.run import _SyntheticBackbone

    model = _SyntheticBackbone(quantized=False)

    def predict(batch: np.ndarray) -> np.ndarray:
        out = model.run(batch)
        return out.reshape(out.shape[0], -1)

    return predict


_predict_fn = None
_batching_queue: BatchingQueue | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predict_fn, _batching_queue
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    _predict_fn = _load_predict_fn()
    _batching_queue = BatchingQueue(
        _predict_fn,
        max_batch_size=cfg["serving"]["max_batch_size"],
        max_batch_wait_ms=cfg["serving"]["max_batch_wait_ms"],
    )
    _batching_queue.start()
    yield
    if _batching_queue:
        await _batching_queue.stop()


app = FastAPI(title="production-vlm-engineering edge inference server", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    try:
        raw = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}") from e

    array = (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)

    if _batching_queue is None:
        raise HTTPException(status_code=503, detail="Server not ready")

    embedding, batch_size_served, queue_wait_ms, inference_ms = await _batching_queue.submit(array)

    return PredictResponse(
        embedding_preview=[float(v) for v in embedding[:8]],
        batch_size_served_with=batch_size_served,
        queue_wait_ms=round(queue_wait_ms, 3),
        inference_ms=round(inference_ms, 3),
    )
