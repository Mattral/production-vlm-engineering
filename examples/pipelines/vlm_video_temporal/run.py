#!/usr/bin/env python
"""P1-04: Video / temporal reasoning template for VLMs.

A minimal but genuinely runnable starting point for temporal VLM
reasoning — the "light video/3D extension" called out as a forward-
looking P1 item in the transformation roadmap. It doesn't pretend to
be a full video training pipeline; it demonstrates the pattern that
generalises from our chart-QA work to multi-frame inputs.

What it shows:
    1. Frame sampling strategies: uniform, key-frame (scene-change
       proxy via L1 pixel diff), and adaptive (high-motion frames).
    2. Temporal context formatting: how to build a VLM prompt that
       spans multiple frames ("Frame 1: ... Frame 3: ...") with a
       structured JSON answer schema.
    3. Temporal grounding metric: extends `production_vlm.eval.grounding_score`
       to multi-frame evidence (each frame gets its own evidence string;
       the answer must ground in *at least one* frame's evidence).
    4. A concrete pointer to where this connects to the rest of the repo:
       frames are embedded by `SyntheticEmbeddingProxy` (or `RealVisionEncoder`
       in production), so the same drift detector that monitors chart
       batches can monitor video frame-by-frame for distribution shift —
       e.g., lighting changes between scenes, sensor switches in a
       multi-camera setup, or format changes in a document stream.

GPU / real model notes:
    Without `pip install -e ".[ml]"` and a CUDA device, this runs a
    CPU-only smoke test: synthetic multi-frame "video" data (sequences
    of perturbed synthetic charts), frame sampling, temporal grounding
    evaluation, and an embedding-drift pass over the frame sequence.
    With the real stack, replace `_synthetic_frame_sequence` with a
    real video loader and `_mock_vlm_temporal` with your VLM call.

References:
    - Video-LLaVA (Lin et al., 2023) — early multi-frame VLM interleaving
    - VITA (Fu et al., 2024) — video + audio instruction following
    - Temporal Grounding in Videos (arXiv survey, 2025) — benchmark landscape
    - CVPR 2026 temporal VLM sessions — key-frame selection and efficient
      temporal attention for long-form video understanding
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from production_vlm.eval import faithfulness_score  # noqa: E402
from production_vlm.robustness.perturbations import apply_perturbation  # noqa: E402
from production_vlm.utils import set_seed, timer  # noqa: E402
from production_vlm.utils.console import Console  # noqa: E402
from production_vlm.utils.synthetic_charts import generate_synthetic_chart  # noqa: E402
from production_vlm.utils.vision_encoder import SyntheticEmbeddingProxy  # noqa: E402

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VideoFrame:
    frame_idx: int
    image: object  # PIL.Image.Image | None
    chart_metadata: object  # SyntheticChart (proxy for real frame content)
    timestamp_s: float = 0.0


@dataclass
class TemporalSample:
    """A multi-frame clip with a temporal question and per-frame evidence."""

    frames: list[VideoFrame]
    question: str
    answer: str
    per_frame_evidence: list[str]
    structured_answer: dict = field(default_factory=dict)  # JSON schema output


@dataclass
class TemporalEvalResult:
    temporal_faithfulness: float
    grounded_in_any_frame: bool
    n_frames: int
    structured_output_valid: bool


# ---------------------------------------------------------------------------
# Synthetic video data (proxy for real video frames)
# ---------------------------------------------------------------------------

STRUCTURED_ANSWER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["answer", "frame_references", "confidence"],
    "properties": {
        "answer": {"type": "string"},
        "frame_references": {"type": "array", "items": {"type": "integer"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


def _synthetic_frame_sequence(seed: int, n_frames: int = 5, add_scene_change: bool = True) -> list[VideoFrame]:
    """Generate a synthetic multi-frame sequence (proxy for a real video clip).

    Simulates a document/chart stream where frames are mostly stable but
    occasionally show a different chart (scene change) — the temporal
    analog of a document switching pages, a dashboard refreshing, or a
    camera angle changing in a real-world deployment.
    """
    frames = []
    base_chart = generate_synthetic_chart(seed=seed, render_image=True)
    for i in range(n_frames):
        if add_scene_change and i == n_frames // 2:
            # Scene change: inject a clearly different chart type/content
            chart = generate_synthetic_chart(seed=seed + 1000, chart_type="pie", render_image=True)
        else:
            # Mild temporal variation (slight noise, simulating camera jitter)
            pert = apply_perturbation(base_chart.image, "gaussian_noise", severity=0.1 * (i % 3), seed=i)
            chart = generate_synthetic_chart(seed=seed + i, render_image=False)
            chart = type(chart)(
                image=pert.perturbed_image,
                **{
                    k: getattr(chart, k)
                    for k in [
                        "chart_type",
                        "title",
                        "categories",
                        "values",
                        "units",
                        "question",
                        "answer",
                        "evidence_text",
                        "style_seed",
                        "plot_bbox",
                    ]
                },
            )
        frames.append(VideoFrame(frame_idx=i, image=chart.image, chart_metadata=chart, timestamp_s=i * 0.5))
    return frames


def _build_temporal_sample(frames: list[VideoFrame]) -> TemporalSample:
    """Build a temporal QA sample: question spans the whole clip, evidence is per-frame."""
    first_chart = frames[0].chart_metadata
    question = (
        f"Across the video clip, which frame shows the highest '{first_chart.categories[0]}' value and what is it?"
    )
    max_val_frame = max(frames, key=lambda f: f.chart_metadata.values[0] if f.chart_metadata.values else 0)
    answer = (
        f"Frame {max_val_frame.frame_idx} shows the highest value of "
        f"{max_val_frame.chart_metadata.values[0]:.1f} {max_val_frame.chart_metadata.units}."
    )
    per_frame_evidence = [f"Frame {f.frame_idx}: {f.chart_metadata.evidence_text}" for f in frames]
    structured_answer = {
        "answer": answer,
        "frame_references": [max_val_frame.frame_idx],
        "confidence": 0.9,
    }
    return TemporalSample(
        frames=frames,
        question=question,
        answer=answer,
        per_frame_evidence=per_frame_evidence,
        structured_answer=structured_answer,
    )


# ---------------------------------------------------------------------------
# Frame sampling strategies
# ---------------------------------------------------------------------------


def sample_uniform(frames: list[VideoFrame], n: int) -> list[VideoFrame]:
    """Select n evenly-spaced frames."""
    if n >= len(frames):
        return frames
    indices = [int(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
    return [frames[i] for i in indices]


def sample_keyframe(frames: list[VideoFrame], threshold: float = 0.15) -> list[VideoFrame]:
    """Select frames where L1 pixel difference from previous exceeds threshold.

    In production: replace the L1 diff with a proper scene-change detector
    (histogram intersection, optical flow magnitude, or a learned boundary
    detector). Here, the L1 diff is computed directly on the numpy arrays
    of our synthetic PIL images — real, not simulated.
    """
    if not frames:
        return frames
    selected = [frames[0]]
    prev_arr = np.asarray(frames[0].image, dtype=np.float32) / 255.0
    for frame in frames[1:]:
        curr_arr = np.asarray(frame.image, dtype=np.float32) / 255.0
        l1_diff = float(np.abs(curr_arr - prev_arr).mean())
        if l1_diff > threshold:
            selected.append(frame)
        prev_arr = curr_arr
    return selected


def sample_adaptive(frames: list[VideoFrame], n_budget: int) -> list[VideoFrame]:
    """Adaptive sampling: always include first/last, fill budget with highest-motion frames."""
    if len(frames) <= n_budget:
        return frames
    diffs = [0.0]
    for i in range(1, len(frames)):
        prev = np.asarray(frames[i - 1].image, dtype=np.float32) / 255.0
        curr = np.asarray(frames[i].image, dtype=np.float32) / 255.0
        diffs.append(float(np.abs(curr - prev).mean()))
    sorted_by_motion = sorted(range(len(frames)), key=lambda i: -diffs[i])
    keep = sorted(set(sorted_by_motion[: n_budget - 2] + [0, len(frames) - 1]))
    return [frames[i] for i in keep]


# ---------------------------------------------------------------------------
# Temporal evaluation metric
# ---------------------------------------------------------------------------


def temporal_grounding_score(prediction: str, per_frame_evidence: list[str]) -> TemporalEvalResult:
    """Multi-frame faithfulness: score against each frame's evidence, take the max.

    A temporally-grounded answer should be supported by *at least one*
    frame's evidence — the model shouldn't be penalised for not grounding
    in frames where the relevant content doesn't appear. This is the
    natural generalisation of `faithfulness_score` to temporal inputs.
    """
    if not per_frame_evidence:
        return TemporalEvalResult(0.0, False, 0, False)

    # Score each frame's evidence separately; the answer only needs to
    # ground in the single most relevant frame, not the concatenation of all of them.
    per_frame_scores = [faithfulness_score(prediction, prediction, ev).score for ev in per_frame_evidence]
    max_frame_score = max(per_frame_scores)
    grounded = max_frame_score > 0.3

    # Validate structured output schema (minimal check)
    structured_valid = False
    try:
        import json as _json

        start = prediction.find("{")
        end = prediction.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = _json.loads(prediction[start:end])
            structured_valid = all(k in parsed for k in ["answer", "frame_references", "confidence"])
    except Exception:
        pass

    return TemporalEvalResult(
        temporal_faithfulness=max_frame_score,
        grounded_in_any_frame=grounded,
        n_frames=len(per_frame_evidence),
        structured_output_valid=structured_valid,
    )


# ---------------------------------------------------------------------------
# Mock VLM (CPU fallback) + format helpers
# ---------------------------------------------------------------------------


def _mock_vlm_temporal(sample: TemporalSample, sampled_frames: list[VideoFrame]) -> str:
    """CPU fallback: generates a structured JSON answer using ground-truth metadata.

    Replace with a real VLM call that receives the sampled frame images
    interleaved with the question prompt:
        'Frame 1: <image> Frame 3: <image> ... Question: ...'
    """
    answer_obj = {
        "answer": sample.answer,
        "frame_references": sample.structured_answer["frame_references"],
        "confidence": 0.9,
    }
    return json.dumps(answer_obj)


def _format_temporal_prompt(sample: TemporalSample, sampled_frames: list[VideoFrame]) -> str:
    """Build the multi-frame prompt string a real VLM would receive."""
    lines = []
    for frame in sampled_frames:
        lines.append(f"Frame {frame.frame_idx} (t={frame.timestamp_s:.1f}s): [IMAGE]")
    lines.append(f"\nQuestion: {sample.question}")
    lines.append("\nPlease answer in JSON format matching this schema:")
    lines.append('{"answer": "<string>", "frame_references": [<list of int>], "confidence": <0-1>}')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Drift detection over frame embeddings (P0-04 integration)
# ---------------------------------------------------------------------------


def detect_scene_changes_via_embedding(frames: list[VideoFrame], encoder: SyntheticEmbeddingProxy) -> list[int]:
    """Use embedding-space cosine distance to detect scene changes in the frame sequence.

    This directly connects P1-04 (video) to P0-04 (drift detection):
    the same `CosineDriftDetector` machinery that monitors production
    batch drift can detect scene boundaries in a video by flagging
    frames where the embedding drifts significantly from the preceding
    window, without any scene-change-specific training.
    """
    from production_vlm.drift import CosineDriftDetector

    charts = [f.chart_metadata for f in frames]
    embeddings = encoder.encode_charts(charts, style_shift_flags=[False] * len(charts))

    if len(embeddings) < 3:
        return []

    # Use the first half of the clip as the "reference distribution"
    half = max(1, len(embeddings) // 2)
    detector = CosineDriftDetector(embeddings[:half], alpha=0.05)
    scene_changes = []
    for i in range(half, len(embeddings)):
        batch = embeddings[i : i + 1]
        result = detector.score_batch(batch)
        if result.is_drift:
            scene_changes.append(i)
    return scene_changes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(config_path: str | None = None) -> dict:
    default_cfg_path = Path(__file__).resolve().parents[3] / "configs" / "vlm_video_temporal.yaml"
    cfg_path = Path(config_path) if config_path else default_cfg_path
    import yaml

    cfg = yaml.safe_load(cfg_path.read_text())

    set_seed(cfg["video"]["seed"])
    console.rule("[bold cyan]VLM Video / Temporal Reasoning Template (P1-04)[/bold cyan]")
    console.print(
        "Minimal runnable template for multi-frame VLM reasoning. "
        "Replace `_synthetic_frame_sequence` with a real video loader "
        "and `_mock_vlm_temporal` with your VLM call."
    )
    console.print("")

    n_clips = cfg["video"]["n_clips"]
    n_frames_per_clip = cfg["video"]["n_frames_per_clip"]
    results_per_strategy: dict[str, list[float]] = {
        "uniform_4": [],
        "keyframe": [],
        "adaptive_4": [],
    }

    encoder = SyntheticEmbeddingProxy(
        embedding_dim=cfg["embedding"]["dim"],
        seed=cfg["embedding"]["seed"],
        shift_magnitude=cfg["embedding"]["shift_magnitude"],
    )

    with timer("temporal evaluation"):
        for clip_idx in range(n_clips):
            frames = _synthetic_frame_sequence(seed=clip_idx * 10, n_frames=n_frames_per_clip)
            sample = _build_temporal_sample(frames)

            strategies = {
                "uniform_4": sample_uniform(frames, cfg["sampling"]["uniform_n"]),
                "keyframe": sample_keyframe(frames, threshold=cfg["sampling"]["keyframe_l1_threshold"]),
                "adaptive_4": sample_adaptive(frames, cfg["sampling"]["adaptive_n"]),
            }

            for strategy_name, sampled in strategies.items():
                prediction = _mock_vlm_temporal(sample, sampled)
                eval_result = temporal_grounding_score(prediction, sample.per_frame_evidence)
                results_per_strategy[strategy_name].append(eval_result.temporal_faithfulness)

            # Detect scene changes via embedding drift (P0-04 integration demo)
            scene_changes = detect_scene_changes_via_embedding(frames, encoder)

    console.table(
        title="Frame Sampling Strategy Comparison (temporal faithfulness)",
        columns=["Strategy", "Mean Faithfulness", "Frames Used (avg)"],
        rows=[
            ["uniform_4 frames", f"{np.mean(results_per_strategy['uniform_4']):.3f}", "4"],
            ["keyframe (L1 thresh=0.12)", f"{np.mean(results_per_strategy['keyframe']):.3f}", "~3-5"],
            ["adaptive_4 frames", f"{np.mean(results_per_strategy['adaptive_4']):.3f}", "4"],
        ],
    )

    console.print("")
    console.print("[dim]Scene change detection via embedding drift (last clip):[/dim]")
    scene_msg = scene_changes if scene_changes else "(none above threshold)"
    console.print(f"  Frame indices flagged as scene changes: {scene_msg}")
    console.print("  → The same CosineDriftDetector from P0-04 detects scene boundaries.")

    output_dir = Path("outputs/vlm_video_temporal")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config_name": "vlm_video_temporal_demo",
        "n_clips": n_clips,
        "n_frames_per_clip": n_frames_per_clip,
        "strategy_results": {k: float(np.mean(v)) for k, v in results_per_strategy.items()},
        "structured_output_schema": STRUCTURED_ANSWER_SCHEMA,
        "p1_04_status": "minimal_runnable_template",
        "next_steps": [
            "Replace _synthetic_frame_sequence() with decord/av/torchvision video loader",
            "Replace _mock_vlm_temporal() with real VLM call (Video-LLaVA, VITA, etc.)",
            "Swap SyntheticEmbeddingProxy for RealVisionEncoder in detect_scene_changes_via_embedding()",
            "Add temporal position embeddings / timestamp tokens if your VLM supports them",
            "Extend temporal_grounding_score() with IoU over temporal segments for grounding",
        ],
    }
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    console.print(f"[bold green]Results written to {out_path}[/bold green]")
    return results


if __name__ == "__main__":
    main()
