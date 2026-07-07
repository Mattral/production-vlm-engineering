"""Visualization utilities: generate matplotlib plots as output artifacts.

The roadmap explicitly requires "clear metrics and visualization." This
module produces publication-quality plots saved as PNG files alongside
each example's ``results.json``, making results inspectable without
needing to re-run the pipeline or open a notebook.

Design: zero interactive dependencies (no display/GUI required). All
plots are saved to disk via ``matplotlib.use("Agg")`` so they work in
CI, containers, and headless servers.

Usage:
    from production_vlm.utils.visualization import (
        plot_faithfulness_comparison,
        plot_drift_timeline,
        plot_perturbation_sweep,
        plot_benchmark_speedup,
        plot_adversarial_embedding_shift,
    )
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 11,
}

_PALETTE = ["#3949AB", "#E53935", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]


def _apply_style():
    plt.rcParams.update(_STYLE)


# ---------------------------------------------------------------------------
# P0-02: vlm_chart_finetune — before/after faithfulness
# ---------------------------------------------------------------------------


def plot_faithfulness_comparison(
    zero_shot_score: float,
    finetuned_score: float,
    structured_zero_shot_mape: float,
    structured_finetuned_mape: float,
    output_path: str | Path,
) -> Path:
    """Bar chart comparing zero-shot vs LoRA fine-tuned on both metrics."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("VLM Chart Fine-Tuning: Before vs After", fontweight="bold", fontsize=13)

    # Faithfulness
    bars = ax1.bar(
        ["Zero-shot", "LoRA fine-tuned"],
        [zero_shot_score, finetuned_score],
        color=[_PALETTE[1], _PALETTE[0]],
        width=0.45,
        zorder=3,
    )
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Mean Faithfulness Score")
    ax1.set_title("Faithfulness (↑ better)")
    for bar, val in zip(bars, [zero_shot_score, finetuned_score], strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}", ha="center", va="bottom", fontweight="bold"
        )

    # Structured extraction MAPE
    bars2 = ax2.bar(
        ["Zero-shot", "LoRA fine-tuned"],
        [structured_zero_shot_mape * 100, structured_finetuned_mape * 100],
        color=[_PALETTE[1], _PALETTE[0]],
        width=0.45,
        zorder=3,
    )
    ax2.set_ylim(0, max(structured_zero_shot_mape * 100 * 1.3, 5))
    ax2.set_ylabel("Numeric MAPE (%)")
    ax2.set_title("Structured Extraction MAPE (↓ better)")
    for bar, val in zip(bars2, [structured_zero_shot_mape * 100, structured_finetuned_mape * 100], strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontweight="bold"
        )

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# P0-04: embedding_drift_active_learning — drift timeline
# ---------------------------------------------------------------------------


def plot_drift_timeline(
    batch_results: list[dict],
    drift_starts_at: int,
    output_path: str | Path,
) -> Path:
    """Timeline plot showing KS statistic, p-value, and drift flags per batch."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Embedding Drift Detection Timeline", fontweight="bold", fontsize=13)

    batches = [r["batch_idx"] for r in batch_results]
    ks_stats = [r["ks_stat"] for r in batch_results]
    ks_flags = [r["ks_flag"] for r in batch_results]

    # KS statistic
    colors = [_PALETTE[0] if not f else _PALETTE[1] for f in ks_flags]
    ax1.bar(batches, ks_stats, color=colors, zorder=3, width=0.7)
    ax1.axvline(
        drift_starts_at - 0.5,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Drift injected (batch {drift_starts_at})",
    )
    ax1.set_ylabel("KS Statistic")
    ax1.set_title("Kolmogorov-Smirnov Test Statistic per Batch")
    ax1.legend(fontsize=9)

    # Drift flag as binary
    flag_colors = [_PALETTE[1] if f else _PALETTE[2] for f in ks_flags]
    ax2.bar(batches, [1] * len(batches), color=flag_colors, zorder=3, width=0.7)
    ax2.axvline(drift_starts_at - 0.5, color="black", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Drift Flag")
    ax2.set_xlabel("Batch Index")
    ax2.set_title("KS Drift Flag (red = DRIFT, green = ok)")
    ax2.set_yticks([])

    # Legend
    red_patch = mpatches.Patch(color=_PALETTE[1], label="DRIFT flagged")
    grn_patch = mpatches.Patch(color=_PALETTE[2], label="In-distribution")
    ax2.legend(handles=[red_patch, grn_patch], fontsize=9)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# P0-03: vlm_edge_inference — speedup comparison
# ---------------------------------------------------------------------------


def plot_benchmark_speedup(
    details: list[dict],
    output_path: str | Path,
) -> Path:
    """Grouped bar chart: fp32 vs INT8 latency across image size × batch size configs."""
    _apply_style()
    fp32 = [d for d in details if d["variant"] == "fp32"]
    int8 = [d for d in details if d["variant"] == "dynamic_int8"]
    labels = [f"{d['image_size']}px b={d['batch_size']}" for d in fp32]

    x = np.arange(len(labels))
    w = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Edge Inference: fp32 vs Dynamic INT8", fontweight="bold", fontsize=13)

    fp32_lat = [d["mean_latency_ms"] for d in fp32]
    int8_lat = [d["mean_latency_ms"] for d in int8]
    b1 = ax1.bar(x - w / 2, fp32_lat, w, label="fp32", color=_PALETTE[0], zorder=3)
    b2 = ax1.bar(x + w / 2, int8_lat, w, label="INT8", color=_PALETTE[2], zorder=3)
    ax1.set_ylabel("Mean Latency (ms)")
    ax1.set_title("Latency (↓ better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend()
    for bar, val in zip(list(b1) + list(b2), fp32_lat + int8_lat, strict=True):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.1, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    speedups = [f / i for f, i in zip(fp32_lat, int8_lat, strict=True)]
    ax2.bar(x, speedups, color=_PALETTE[3], zorder=3)
    ax2.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("INT8 Speedup vs fp32 (↑ better)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    for bar, val in zip(ax2.patches, speedups, strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.05,
            f"{val:.2f}×",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# P1-02: vlm_robustness_guard — perturbation sweep heatmap
# ---------------------------------------------------------------------------


def plot_perturbation_sweep(
    perturbation_results: dict[str, dict[str, float]],
    output_path: str | Path,
) -> Path:
    """Heatmap of accuracy across perturbation type × severity."""
    _apply_style()
    kinds = sorted(perturbation_results.keys())
    severities = sorted({float(s) for kind_data in perturbation_results.values() for s in kind_data})

    matrix = np.array([[perturbation_results[k].get(str(s), 0.0) for s in severities] for k in kinds])

    fig, ax = plt.subplots(figsize=(8, max(4, len(kinds) * 0.65)))
    fig.suptitle("Perturbation Robustness: Accuracy Heatmap", fontweight="bold", fontsize=13)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Accuracy")
    ax.set_xticks(range(len(severities)))
    ax.set_xticklabels([f"{s:.2f}" for s in severities])
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Perturbation Type")

    for i in range(len(kinds)):
        for j in range(len(severities)):
            val = matrix[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=9, fontweight="bold", color=text_color)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# P1-02: adversarial embedding shift visualisation
# ---------------------------------------------------------------------------


def plot_adversarial_embedding_shift(
    adv_result: dict,
    output_path: str | Path,
) -> Path:
    """Radar / bar chart showing adversarial embedding shift metrics."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(
        f"Adversarial Robustness (ε={adv_result['epsilon_budget_8_255']:.4f}, {adv_result['mode']})",
        fontweight="bold",
        fontsize=12,
    )

    metrics = {
        "Cosine sim\n(clean)": adv_result["centroid_cosine_before"],
        "Cosine sim\n(adversarial)": adv_result["centroid_cosine_after"],
        "Cosine drop": adv_result["centroid_cosine_drop"],
        "OOD catch\nrate": adv_result["ood_detector_catch_rate"],
    }

    colors = [_PALETTE[2], _PALETTE[1], _PALETTE[3], _PALETTE[0]]
    bars = ax.bar(range(len(metrics)), list(metrics.values()), color=colors, zorder=3, width=0.55)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(list(metrics.keys()), fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")

    for bar, val in zip(bars, metrics.values(), strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    ax.annotate(
        "Low OOD catch rate on proxy attack = embedding guard is robust to pixel noise.\n"
        "Real PGD (gradient-based) requires pip install -e '[ml]' + CUDA.",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=8,
        color="grey",
    )

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
