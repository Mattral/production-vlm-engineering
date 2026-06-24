# VLM Video / Temporal Reasoning

**Pipeline:** `examples/pipelines/vlm_video_temporal/`  
**Config:** `configs/vlm_video_temporal.yaml`  
**P-level:** P1-04 (minimal runnable template)

## What it demonstrates

Multi-frame VLM reasoning is the natural extension of the chart-QA work: instead of one image, the model receives a sequence of frames and must reason across time. This example shows the engineering patterns without pretending to be a full video training pipeline.

Three things, all genuinely runnable on CPU:

1. **Frame sampling strategies** — three algorithms for selecting a budget of frames from a longer clip, with measured faithfulness for each:

    | Strategy | Description |
    |---|---|
    | `uniform` | Evenly-spaced frames — baseline, no motion awareness |
    | `keyframe` | Frames where L1 pixel diff from previous exceeds a threshold — naive scene-change proxy |
    | `adaptive` | First + last frame always kept; remaining budget filled by highest-motion frames |

2. **Temporal grounding metric** — `temporal_grounding_score()` extends `faithfulness_score` to multi-frame evidence: the answer must ground in *at least one* frame, not necessarily all frames. This correctly models temporal QA where the relevant information may only appear in certain frames.

3. **Scene-change detection via embedding drift** — directly connects P1-04 to P0-04: the same `CosineDriftDetector` that monitors production batch drift can detect scene boundaries by flagging frames whose embeddings drift from a preceding window.

## Run it

```bash
cv-playbook run-example vlm_video_temporal
```

Runs in ~3s on CPU. No real video files needed.

## Connecting to the rest of the repo

```
Synthetic frames → SyntheticEmbeddingProxy → CosineDriftDetector (scene boundaries)
                                                     ↑
                              Same detector as embedding_drift_active_learning

Temporal QA answer → temporal_grounding_score → faithfulness_score (per-frame evidence)
                                                           ↑
                              Same metric as vlm_chart_finetune

JSON answer schema → structured output → STRUCTURED_ANSWER_SCHEMA
                                                    ↑
                              Same pattern as vlm_chart_finetune structured extraction
```

## Extending to real video

Swap two functions in `run.py`:

```python
# 1. Real frame loader (replace _synthetic_frame_sequence)
import decord
def real_frame_sequence(video_path: str, n_frames: int) -> list[VideoFrame]:
    vr = decord.VideoReader(video_path)
    indices = [int(i * len(vr) / n_frames) for i in range(n_frames)]
    return [VideoFrame(i, vr[idx].asnumpy(), timestamp_s=float(idx) / vr.get_avg_fps())
            for i, idx in enumerate(indices)]

# 2. Real VLM call (replace _mock_vlm_temporal)
def real_vlm_temporal(sample: TemporalSample, sampled_frames: list[VideoFrame]) -> str:
    # Interleave frame images with prompt tokens, per your VLM's API
    ...
```

See `results.json` `next_steps` for the full checklist. Requires `pip install -e ".[ml]" decord`.

## Status vs a production video pipeline

This template is intentionally scoped. What it does **not** include:

- Temporal position embeddings or timestamp token injection (VLM-specific)
- Temporal grounding by segment (IoU over time intervals, not just frame references)  
- Audio stream handling (VITA-style multimodal)
- Optical flow for motion estimation (better than L1 diff for keyframe detection)

These are tracked as `next_steps` in `results.json` and in `ROADMAP.md`.
