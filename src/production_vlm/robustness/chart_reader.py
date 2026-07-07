"""Pixel-based bar chart reader proxy task for measuring perturbation robustness."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class ReadResult:
    predicted_max_category_index: int
    bar_heights_px: list[int]
    correct: bool


def _estimate_background_color(arr: np.ndarray) -> np.ndarray:
    """Sample the four image corners to estimate the background color adaptively."""
    h, w = arr.shape[:2]
    margin = max(2, min(h, w) // 40)
    corners = np.concatenate(
        [
            arr[:margin, :margin].reshape(-1, 3),
            arr[:margin, -margin:].reshape(-1, 3),
            arr[-margin:, :margin].reshape(-1, 3),
            arr[-margin:, -margin:].reshape(-1, 3),
        ],
        axis=0,
    )
    return corners.mean(axis=0)


def _is_background(pixel: np.ndarray, bg_color: np.ndarray, tolerance: float = 25.0) -> bool:
    return bool(np.linalg.norm(pixel.astype(np.float64) - bg_color) <= tolerance)


def _find_plot_area_bounds(arr: np.ndarray) -> tuple[int, int, int, int]:
    """Locate axes bounds by finding dark spine lines, relative to background.

    Fallback when plot_bbox is not provided by the chart generator.
    Brittle under blur/contrast perturbation by design -- prefer
    passing plot_bbox from SyntheticChart when available.
    """
    h, w = arr.shape[:2]
    bg_brightness = _estimate_background_color(arr).mean()
    pixel_brightness = arr.astype(np.float64).sum(axis=-1) / 3.0
    dark_threshold = bg_brightness * 0.6
    is_dark = pixel_brightness < dark_threshold

    col_dark_counts = is_dark.sum(axis=0)
    row_dark_counts = is_dark.sum(axis=1)

    # Max-pool over a small window to tolerate blur-smearing of the 1px spine.
    window = 3
    pad = window // 2
    row_dark_max = np.array(
        [np.pad(row_dark_counts, pad, mode="edge")[i : i + window].max() for i in range(len(row_dark_counts))]
    )
    col_dark_max = np.array(
        [np.pad(col_dark_counts, pad, mode="edge")[i : i + window].max() for i in range(len(col_dark_counts))]
    )

    left_candidates = np.where(col_dark_max > h * 0.4)[0]
    x_left = int(left_candidates[0]) if len(left_candidates) > 0 else 0

    bottom_candidates = np.where(row_dark_max > w * 0.4)[0]
    y_bottom = int(bottom_candidates[-1]) if len(bottom_candidates) > 0 else h - 1
    y_top = int(bottom_candidates[0]) + 1 if len(bottom_candidates) > 0 else int(h * 0.10)
    x_right = w - int(w * 0.03)

    return x_left, x_right, y_top, y_bottom


def read_tallest_bar(
    image: Image.Image,
    n_bars: int,
    true_max_index: int,
    plot_bbox: tuple[int, int, int, int] | None = None,
) -> ReadResult:
    """Estimate which bar is tallest from pixels alone.

    When ``plot_bbox`` (x_left, x_right, y_top, y_bottom) is supplied
    directly from the chart generator, spine-detection is bypassed
    entirely -- the bounds are exact regardless of how the image has
    been perturbed, cleanly isolating the robustness measurement to
    the bar-height reading step rather than confounding it with
    preprocessing brittleness.
    """
    arr = np.asarray(image.convert("RGB"))
    if plot_bbox is not None:
        x_left, x_right, y_top, y_bottom = plot_bbox
    else:
        x_left, x_right, y_top, y_bottom = _find_plot_area_bounds(arr)

    plot_width = x_right - x_left
    bg_color = _estimate_background_color(arr)

    bar_heights = []
    for i in range(n_bars):
        x_center = x_left + int(plot_width * (i + 0.5) / n_bars)
        x_lo = max(x_left, x_center - 3)
        x_hi = min(x_right, x_center + 3)
        column = arr[y_top:y_bottom, x_lo:x_hi, :]

        if column.shape[0] == 0 or column.shape[1] == 0:
            bar_heights.append(0)
            continue

        # Skip the first few rows to clear the top axis spine, which sits just
        # inside the y_top boundary from matplotlib's get_window_extent() and
        # would otherwise always be detected as non-background before any bar.
        spine_skip = 3
        topmost_nonbg_row = column.shape[0]
        for row_idx in range(spine_skip, column.shape[0]):
            row_pixels = column[row_idx]
            nonbg = sum(0 if _is_background(p, bg_color) else 1 for p in row_pixels)
            if nonbg > len(row_pixels) // 2:
                topmost_nonbg_row = row_idx
                break

        bar_heights.append(column.shape[0] - topmost_nonbg_row)

    predicted_idx = int(np.argmax(bar_heights))
    return ReadResult(
        predicted_max_category_index=predicted_idx,
        bar_heights_px=bar_heights,
        correct=(predicted_idx == true_max_index),
    )
