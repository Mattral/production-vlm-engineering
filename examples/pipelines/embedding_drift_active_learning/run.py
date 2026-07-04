#!/usr/bin/env python
"""Vision embedding drift detection & active learning loop (Production MLOps core).

Implements P0-04 of the Production VLM Engineering roadmap.
Enterprise reports from early 2026 repeatedly cite the absence of
embedding-space drift detection as the leading cause of silent
production CV failure: a model keeps running, latency looks fine,
but the input distribution has quietly shifted (new camera, new
lighting, new data source, a new chart-rendering pipeline upstream)
and accuracy degrades without anyone noticing until it's a customer
complaint. This example builds the missing piece directly:

    1. Build a reference embedding set from "known good" data.
    2. Stream batches through the system; for each batch, compute
       both a one-shot distributional test (Kolmogorov-Smirnov on
       cosine similarity to the reference centroid) and an online,
       alertable SPC/EWMA signal.
    3. When drift is flagged, run a simple active-learning triage:
       rank the batch's samples by novelty (distance from the
       reference centroid) and select the top-k for human labeling /
       retraining-queue insertion — the cheapest "what do I do about
       this drift" loop that still meaningfully reduces labeling cost
       versus random sampling.

Reference techniques:
    - Statistical process control / EWMA control limits — classic
      manufacturing QA, applied here to ML embedding monitoring per
      2026 production-MLOps convention (alibi-detect / evidently-style
      patterns, reimplemented here dependency-light for vision).
    - Two-sample Kolmogorov-Smirnov test (Massey, 1951) as a
      distribution-free drift test, robust to the non-Gaussian shape
      typical of real cosine-similarity distributions.
    - Uncertainty/novelty sampling for active learning (Settles, 2009
      survey) — here approximated via distance-from-centroid as a
      cheap proxy that needs no labels and no extra model calls.

Run:
    python -m examples.pipelines.embedding_drift_active_learning.run
    # or: production-vlm run-example embedding_drift_active_learning

This example needs only numpy/scipy/matplotlib/pillow — no GPU, no
network — by default, via SyntheticEmbeddingProxy. Swap in
RealVisionEncoder (production_vlm.utils.vision_encoder) for a genuine
DINOv3/SigLIP-2/CLIP embedding space once you have GPU + network access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from production_vlm.drift import CosineDriftDetector, EWMADriftDetector, select_for_active_learning  # noqa: E402
from production_vlm.utils import set_seed, timer  # noqa: E402
from production_vlm.utils.console import Console  # noqa: E402
from production_vlm.utils.observability import ObservabilityLogger, PrometheusMetricsServer  # noqa: E402
from production_vlm.utils.retraining import QueuedSample, RetrainingTrigger  # noqa: E402
from production_vlm.utils.visualization import plot_drift_timeline  # noqa: E402
from production_vlm.utils.synthetic_charts import generate_synthetic_chart  # noqa: E402
from production_vlm.utils.vision_encoder import SyntheticEmbeddingProxy  # noqa: E402

console = Console()

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "embedding_drift_active_learning.yaml"


def _load_config(config_path: str | None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    return yaml.safe_load(path.read_text())


def _build_reference_set(encoder: SyntheticEmbeddingProxy, n: int, seed: int) -> np.ndarray:
    charts = [generate_synthetic_chart(seed=seed + i, render_image=False) for i in range(n)]
    return encoder.encode_charts(charts, style_shift_flags=[False] * n)


def _build_stream_batch(
    encoder: SyntheticEmbeddingProxy, batch_idx: int, batch_size: int, seed: int, is_drifted: bool
) -> tuple[np.ndarray, list]:
    base_seed = seed + batch_idx * batch_size
    charts = [
        generate_synthetic_chart(seed=base_seed + i, style_shift=is_drifted, render_image=False)
        for i in range(batch_size)
    ]
    embeddings = encoder.encode_charts(charts, style_shift_flags=[is_drifted] * batch_size)
    return embeddings, charts


def main(config_path: str | None = None) -> dict:
    cfg = _load_config(config_path)
    set_seed(cfg["reference"]["seed"])

    console.rule(f"[bold cyan]Embedding Drift Detection & Active Learning: {cfg['name']}[/bold cyan]")

    encoder = SyntheticEmbeddingProxy(
        embedding_dim=cfg["embedding_dim"],
        seed=cfg["reference"]["seed"],
        shift_magnitude=cfg.get("shift_magnitude", 12.0),
    )
    console.print(
        f"Embedding source: [bold]{cfg['embedding_source']}[/bold] (dim={cfg['embedding_dim']}). "
        "Swap for production_vlm.utils.vision_encoder.RealVisionEncoder for a genuine DINOv3/SigLIP-2 space."
    )

    with timer("reference set construction"):
        reference_embeddings = _build_reference_set(encoder, cfg["reference"]["n_samples"], cfg["reference"]["seed"])

    cosine_detector = CosineDriftDetector(reference_embeddings, alpha=cfg["detector"]["cosine_alpha"])
    ewma_detector = EWMADriftDetector(
        lam=cfg["detector"]["ewma_lambda"],
        n_sigma=cfg["detector"]["ewma_n_sigma"],
        warmup=cfg["detector"]["ewma_warmup"],
        baseline_n=cfg["detector"]["ewma_baseline_n"],
    )

    # Observability: structured JSONL event log (always) + optional Prometheus server
    output_dir = Path(cfg["output_dir"])
    obs_logger = ObservabilityLogger(
        output_dir / "events.jsonl",
        run_id=f"{cfg['name']}_{int(__import__('time').time())}",
    )
    prom_port = cfg.get("observability", {}).get("prometheus_port", 0)
    prom_server = PrometheusMetricsServer(port=prom_port) if prom_port else None
    if prom_server:
        prom_server.start()

    # Retraining trigger: closes the drift → label → retrain feedback loop.
    # When the active-learning queue reaches `retraining_threshold` samples,
    # the callback fires (default: prints what a real job would do; replace
    # with a call to train_real() or a cluster job submission in production).
    retrain_threshold = cfg.get("retraining", {}).get("queue_threshold", 15)
    retrain_trigger = RetrainingTrigger(
        queue_threshold=retrain_threshold,
        cooldown_s=cfg.get("retraining", {}).get("cooldown_s", 0),
    )

    console.print(f"Reference set: {reference_embeddings.shape[0]} samples, centroid established.")
    console.print(
        f"Streaming {cfg['stream']['n_batches']} batches of {cfg['stream']['batch_size']}; "
        f"drift injected from batch {cfg['stream']['drift_starts_at_batch']} onward."
    )
    console.print(f"Observability log → {obs_logger.log_path}")

    rows = []
    al_queue_total = 0
    detection_batch = None

    with timer("streaming + drift detection"):
        for batch_idx in range(cfg["stream"]["n_batches"]):
            is_drifted_batch = batch_idx >= cfg["stream"]["drift_starts_at_batch"]
            embeddings, charts = _build_stream_batch(
                encoder, batch_idx, cfg["stream"]["batch_size"], cfg["stream"]["seed"], is_drifted_batch
            )

            ks_result = cosine_detector.score_batch(embeddings)
            ewma_result = ewma_detector.update(ks_result.batch_mean_similarity)

            flagged = ks_result.is_drift or ewma_result.is_drift
            if flagged and detection_batch is None and is_drifted_batch:
                detection_batch = batch_idx

            al_selected = np.array([], dtype=int)
            if flagged:
                al_selected = select_for_active_learning(
                    [ks_result], embeddings, top_k=cfg["active_learning"]["top_k_per_batch"]
                )
                al_queue_total += len(al_selected)

                # Enqueue into the retraining trigger with novelty scores.
                # The trigger fires a retraining callback when the threshold is reached,
                # closing the drift → active-learning → retrain feedback loop.
                l2_norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
                normalized = embeddings / np.clip(l2_norm, 1e-12, None)
                centroid = cosine_detector.centroid
                sims = normalized @ centroid
                novelty_scores = 1.0 - sims  # higher = more novel / farther from centroid

                queued_samples = [
                    QueuedSample(
                        embedding_index=int(idx),
                        batch_idx=batch_idx,
                        novelty_score=float(novelty_scores[idx]),
                        flagged_by="drift_ks" if ks_result.is_drift else "drift_ewma",
                    )
                    for idx in al_selected
                ]
                retrain_trigger.enqueue_batch(queued_samples)

            # Emit structured observability events
            obs_logger.log_drift_event(
                batch_idx=batch_idx,
                batch_size=cfg["stream"]["batch_size"],
                is_drift_ks=ks_result.is_drift,
                is_drift_ewma=ewma_result.is_drift,
                ks_stat=ks_result.score,
                p_value=ks_result.p_value,
                batch_mean_similarity=ks_result.batch_mean_similarity,
                ewma_mean=ewma_result.reference_mean_similarity,
                ewma_lower_cl=ewma_result.details.get("lower_control_limit", 0.0),
                al_selected_count=len(al_selected),
                extra={"true_drift_injected": is_drifted_batch},
            )
            if prom_server:
                prom_server.record_drift(
                    ks_stat=ks_result.score,
                    is_drift_ks=ks_result.is_drift,
                    is_drift_ewma=ewma_result.is_drift,
                    batch_mean_similarity=ks_result.batch_mean_similarity,
                    al_queued=len(al_selected),
                )

            rows.append(
                [
                    str(batch_idx),
                    "yes" if is_drifted_batch else "no",
                    f"{ks_result.score:.4f}",
                    f"{ks_result.p_value:.4g}" if ks_result.p_value is not None else "-",
                    "DRIFT" if ks_result.is_drift else "ok",
                    "DRIFT" if ewma_result.is_drift else "ok",
                    str(len(al_selected)),
                ]
            )

    console.table(
        title="Drift Monitoring Stream",
        columns=["Batch", "True Drift Injected", "KS Stat", "p-value", "KS Flag", "EWMA Flag", "AL Selected"],
        rows=rows,
    )

    ground_truth_start = cfg["stream"]["drift_starts_at_batch"]
    detection_delay = (detection_batch - ground_truth_start) if detection_batch is not None else None

    console.print("")
    if detection_batch is not None:
        console.print(
            f"[bold green]Drift detected at batch {detection_batch} "
            f"(true shift started at batch {ground_truth_start}, "
            f"detection delay = {detection_delay} batches).[/bold green]"
        )
    else:
        console.print("[red]Drift was injected but never flagged -- check detector thresholds.[/red]")

    console.print(
        f"Active learning queue accumulated {al_queue_total} samples "
        f"flagged for human labeling / retraining."
    )

    obs_summary = obs_logger.summary()
    retrain_summary = retrain_trigger.summary()
    results = {
        "config_name": cfg["name"],
        "n_batches": cfg["stream"]["n_batches"],
        "drift_starts_at_batch": ground_truth_start,
        "drift_detected_at_batch": detection_batch,
        "detection_delay_batches": detection_delay,
        "active_learning_queue_size": al_queue_total,
        "retraining": retrain_summary,
        "observability": {
            "events_log": str(obs_logger.log_path),
            "summary": obs_summary,
        },
        "batches": [
            {
                "batch_idx": i,
                "true_drift": i >= ground_truth_start,
                "ks_stat": float(rows[i][2]),
                "ks_flag": rows[i][4] == "DRIFT",
                "ewma_flag": rows[i][5] == "DRIFT",
            }
            for i in range(len(rows))
        ],
    }

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2))

    # Generate visualization artifact
    try:
        plot_path = plot_drift_timeline(
            batch_results=results["batches"],
            drift_starts_at=ground_truth_start,
            output_path=output_dir / "drift_timeline.png",
        )
        results["plots"] = {"drift_timeline": str(plot_path)}
        out_path.write_text(json.dumps(results, indent=2))
        console.print(f"[bold green]Plot → {plot_path}[/bold green]")
    except Exception as e:
        console.print(f"[yellow]Plot generation skipped: {e}[/yellow]")

    console.print(f"[bold green]Results → {out_path}[/bold green]")
    console.print(
        f"[bold green]Observability events: {obs_logger.log_path} "
        f"({obs_summary['total_events']} events)[/bold green]"
    )
    return results


def benchmark() -> None:
    """Sweep synthetic drift magnitude (subtle -> obvious) and report detection delay per detector.

    Sweeping `cosine_alpha` alone (holding drift magnitude fixed) is
    not very informative once the injected shift is large enough to
    be trivially detectable at every threshold tested -- the
    interesting axis is *how subtle the real-world drift is*, so this
    sweeps `shift_magnitude` instead and reports both detectors' delay
    and the active-learning queue size at each level.
    """
    console.rule("[bold cyan]Drift Detector Benchmark Sweep[/bold cyan]")
    base_cfg = _load_config(None)
    rows = []
    for magnitude in [1.0, 3.0, 6.0, 9.0, 12.0, 18.0]:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["shift_magnitude"] = magnitude
        cfg["output_dir"] = f"outputs/embedding_drift_active_learning/bench_mag_{magnitude}"
        result = main_with_cfg(cfg)
        delay = result["detection_delay_batches"]
        rows.append(
            [
                f"{magnitude}",
                str(result["drift_detected_at_batch"]) if result["drift_detected_at_batch"] is not None else "never",
                str(delay) if delay is not None else "-",
                str(result["active_learning_queue_size"]),
            ]
        )
    console.table(
        title="Sensitivity Sweep: Synthetic Drift Magnitude vs Detection Delay",
        columns=["Shift Magnitude", "Detected At Batch", "Delay (batches)", "AL Queue Size"],
        rows=rows,
    )
    console.print(
        "[dim]Lower shift magnitude approximates subtler real-world drift (e.g. gradual sensor "
        "degradation) and is expected to take longer -- or fail -- to detect at fixed thresholds; "
        "this is the expected, honest sensitivity/specificity tradeoff of any drift monitor.[/dim]"
    )


def main_with_cfg(cfg: dict) -> dict:
    """Variant of main() that accepts an in-memory config dict instead of a path, used by benchmark()."""
    set_seed(cfg["reference"]["seed"])
    encoder = SyntheticEmbeddingProxy(
        embedding_dim=cfg["embedding_dim"],
        seed=cfg["reference"]["seed"],
        shift_magnitude=cfg.get("shift_magnitude", 12.0),
    )
    reference_embeddings = _build_reference_set(encoder, cfg["reference"]["n_samples"], cfg["reference"]["seed"])
    cosine_detector = CosineDriftDetector(reference_embeddings, alpha=cfg["detector"]["cosine_alpha"])
    ewma_detector = EWMADriftDetector(
        lam=cfg["detector"]["ewma_lambda"],
        n_sigma=cfg["detector"]["ewma_n_sigma"],
        warmup=cfg["detector"]["ewma_warmup"],
        baseline_n=cfg["detector"]["ewma_baseline_n"],
    )
    detection_batch = None
    al_queue_total = 0
    ground_truth_start = cfg["stream"]["drift_starts_at_batch"]
    for batch_idx in range(cfg["stream"]["n_batches"]):
        is_drifted_batch = batch_idx >= ground_truth_start
        embeddings, _ = _build_stream_batch(
            encoder, batch_idx, cfg["stream"]["batch_size"], cfg["stream"]["seed"], is_drifted_batch
        )
        ks_result = cosine_detector.score_batch(embeddings)
        ewma_result = ewma_detector.update(ks_result.batch_mean_similarity)
        flagged = ks_result.is_drift or ewma_result.is_drift
        if flagged and detection_batch is None and is_drifted_batch:
            detection_batch = batch_idx
        if flagged:
            al_queue_total += cfg["active_learning"]["top_k_per_batch"]
    detection_delay = (detection_batch - ground_truth_start) if detection_batch is not None else None
    return {
        "drift_detected_at_batch": detection_batch,
        "detection_delay_batches": detection_delay,
        "active_learning_queue_size": al_queue_total,
    }


if __name__ == "__main__":
    main()
