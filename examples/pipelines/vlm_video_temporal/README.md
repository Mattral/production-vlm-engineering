# vlm_video_temporal

**P1-04 (forward-looking)**: minimal runnable template for video/temporal VLM reasoning.

## What it demonstrates

1. **Three frame sampling strategies** — uniform, key-frame (L1 scene-change proxy), and adaptive (highest-motion frames within a budget)
2. **Structured temporal prompt formatting** — how to build multi-frame VLM prompts with a typed JSON answer schema
3. **Temporal grounding metric** — extends `faithfulness_score` to multi-frame evidence: the answer must ground in *at least one* frame, not every frame
4. **Drift-as-scene-detection** — the same `CosineDriftDetector` from P0-04 can detect scene boundaries by flagging frames whose embeddings drift from the preceding window

## Status

This is an explicit minimal template, not a full training pipeline. The roadmap marks P1-04 as "or pointers + minimal runnable template" — this satisfies that requirement with:

- Real frame sampling logic (the L1 and embedding-drift scene-change detectors are genuine)
- Real temporal grounding metrics (computed via the actual `faithfulness_score` harness)
- Clear `next_steps` in `results.json` for what to swap in for production use

## Run it

```bash
production-vlm run-example vlm_video_temporal
```

Runs in ~3s on CPU. No GPU, no video file needed — uses synthetic chart sequences as proxy frames.

## Extending to real video

```python
# 1. Real frame loader (replace _synthetic_frame_sequence)
import decord
vr = decord.VideoReader("your_video.mp4")
frames = [VideoFrame(i, vr[i].asnumpy(), ...) for i in range(len(vr))]

# 2. Real VLM call (replace _mock_vlm_temporal)
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("your-video-vlm")
prediction = model.generate(
    prompt=_format_temporal_prompt(sample, sampled_frames),
    images=[f.image for f in sampled_frames],
)

# 3. Real vision encoder for scene-change detection (replace SyntheticEmbeddingProxy)
from production_vlm.utils.vision_encoder import RealVisionEncoder
encoder = RealVisionEncoder("facebook/dinov2-base")
```

Requires `pip install -e ".[ml]"` for items 2 and 3, and `pip install decord` for item 1.

## Connection to the rest of the repo

Frame embeddings go through the exact same `CosineDriftDetector` used in `embedding_drift_active_learning`. This means:

- A production video pipeline can monitor for scene changes, camera switches, or format changes using the same drift infrastructure already in place for batch monitoring
- The same active-learning queue can be used to flag unusual frames for human review
- The same `HallucinationGuard` applies directly to temporal VQA answers

The temporal prompt format uses a JSON answer schema (`STRUCTURED_ANSWER_SCHEMA` in `run.py`), matching the P0-02 structured extraction pattern and making temporal answers programmatically consumable downstream.

## Files

- `run.py` — frame sampling, temporal evaluation, scene-change detection, `main()` entry point
- `STRUCTURED_ANSWER_SCHEMA` in `run.py` — the typed JSON output schema for temporal answers
