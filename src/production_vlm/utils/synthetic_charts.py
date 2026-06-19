"""Synthetic chart generation for VLM fine-tuning/eval without external datasets.

Produces matplotlib-rendered bar/line/pie charts with a ground-truth
JSON record (title, series, values, units) and a derived natural-
language QA pair. This gives a zero-download path through the
examples (P0-02/P0-04) and a controllable way to inject distribution
shift (style/lighting/color-scheme changes) for drift-detection demos.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


@dataclass
class SyntheticChart:
    image: Image.Image | None
    chart_type: str
    title: str
    categories: list[str]
    values: list[float]
    units: str
    question: str
    answer: str
    evidence_text: str
    style_seed: int = field(default=0)
    # Pixel coordinates of the plot axes area (x_left, x_right, y_top, y_bottom),
    # populated when render_image=True. Used by chart_reader to avoid fragile
    # pixel-heuristic spine detection that breaks under perturbation.
    plot_bbox: tuple[int, int, int, int] | None = field(default=None)


_PALETTES = [
    ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"],
    ["#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E"],
    ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"],
]

_CATEGORY_GROUPS = [
    ["North", "South", "East", "West", "Central"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["EU", "US", "APAC", "LATAM"],
]
_METRIC_POOL = [("Revenue", "USD M"), ("Active Users", "K"), ("Latency", "ms"), ("Conversion Rate", "%"), ("Throughput", "req/s")]


def generate_synthetic_chart(
    seed: int,
    chart_type: str | None = None,
    style_shift: bool = False,
    render_image: bool = True,
) -> SyntheticChart:
    """Generate one synthetic chart + QA pair.

    Args:
        seed: Controls data values and category selection (reproducibility).
        chart_type: One of "bar", "line", "pie". Random if None.
        style_shift: If True, applies an out-of-distribution visual
            style (different palette family, font, background) to
            simulate the kind of drift a production system would
            encounter from a new data source or rendering pipeline.
        render_image: If False, skips the matplotlib rendering pass
            and returns ``image=None`` -- used by callers (e.g. the
            embedding-drift example) that only need chart metadata
            (values/categories/chart_type) and would otherwise pay an
            unnecessary rendering cost across thousands of samples.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    chart_type = chart_type or rng.choice(["bar", "line", "pie"])
    metric_name, units = rng.choice(_METRIC_POOL)
    category_group = rng.choice(_CATEGORY_GROUPS)
    n_categories = min(len(category_group), rng.randint(4, len(category_group)))
    categories = rng.sample(category_group, n_categories)
    dimension_name = "Region" if categories[0] in _CATEGORY_GROUPS[0] else ("Quarter" if categories[0] in _CATEGORY_GROUPS[1] else "Market")
    base = rng.uniform(10, 100)
    values = [max(0.1, base + np_rng.normal(0, base * 0.35)) for _ in categories]

    title = f"{metric_name} by {dimension_name}"

    image = None
    if render_image:
        palette = rng.choice(_PALETTES)
        if style_shift:
            # Inject a visibly different rendering regime: grayscale-ish palette,
            # different font family/size, light background tint.
            palette = ["#222222", "#555555", "#888888", "#aaaaaa", "#cccccc"]

        fig, ax = plt.subplots(figsize=(5, 3.4), dpi=80)
        if style_shift:
            fig.patch.set_facecolor("#f0f0e8")
            ax.set_facecolor("#f0f0e8")

        if chart_type == "bar":
            ax.bar(categories, values, color=palette[: len(categories)] if len(palette) >= len(categories) else palette * 3)
            ax.set_ylabel(f"{metric_name} ({units})")
        elif chart_type == "line":
            ax.plot(categories, values, marker="o", color=palette[0], linewidth=2)
            ax.set_ylabel(f"{metric_name} ({units})")
        else:  # pie
            ax.pie(values, labels=categories, autopct="%1.1f%%", colors=palette[: len(categories)] if len(palette) >= len(categories) else palette * 3)

        ax.set_title(title, fontsize=13, fontweight="bold")
        fig.tight_layout()

        # Extract the axes' pixel coordinates *after* tight_layout has finalised
        # the layout, so the bbox reflects the actual rendered position.
        # This lets chart_reader bypass fragile pixel-heuristic spine detection
        # (which breaks under blur/contrast perturbation) and use the true
        # plot area directly, cleanly separating "preprocessing" from the
        # perturbation robustness measurement.
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        dpi = fig.dpi
        fig_h = fig.get_figheight() * dpi
        plot_bbox = (
            int(bbox.x0),                  # x_left
            int(bbox.x1),                  # x_right
            int(fig_h - bbox.y1),          # y_top  (matplotlib y-axis is bottom-up; image is top-down)
            int(fig_h - bbox.y0),          # y_bottom
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
    else:
        plot_bbox = None

    max_idx = int(np.argmax(values))
    question = f"What is the {metric_name.lower()} for {categories[max_idx]}, and is it the highest category?"
    answer = f"{categories[max_idx]} has {metric_name.lower()} of {values[max_idx]:.1f} {units}, which is the highest."
    evidence_text = "; ".join(f"{c}: {v:.1f} {units}" for c, v in zip(categories, values))

    return SyntheticChart(
        image=image,
        chart_type=chart_type,
        title=title,
        categories=categories,
        values=[round(v, 2) for v in values],
        units=units,
        question=question,
        answer=answer,
        evidence_text=evidence_text,
        style_seed=seed,
        plot_bbox=plot_bbox,
    )


def generate_dataset(n: int, seed: int = 0, style_shift_fraction: float = 0.0, render_image: bool = True) -> list[SyntheticChart]:
    """Generate a list of synthetic charts, optionally injecting style-shifted samples.

    ``style_shift_fraction`` controls what fraction of the tail of the
    dataset is rendered in the OOD style — used directly by the
    embedding-drift example (P0-04) to simulate a production
    distribution shift arriving partway through a stream.
    """
    n_shift = int(n * style_shift_fraction)
    charts = []
    for i in range(n):
        is_shifted = i >= (n - n_shift)
        charts.append(generate_synthetic_chart(seed=seed + i, style_shift=is_shifted, render_image=render_image))
    return charts
