"""Embedding-space drift detection for vision/VLM encoders.

Implements two complementary detectors used in production CV/VLM
monitoring (per 2026 enterprise reports citing lack of drift detection
as a leading cause of silent production failure):

1. ``CosineDriftDetector`` — tracks the distribution of cosine
   similarities between incoming-batch embeddings and a reference
   centroid (or reference set). A shift in the mean/variance of this
   distribution signals representation drift (e.g. lighting, style,
   sensor, or domain shift).

2. ``EWMADriftDetector`` — applies exponentially-weighted moving
   average control limits (statistical process control, SPC) to a
   scalar drift score over a stream of batches, giving an online,
   alertable signal rather than a one-shot test.

Both are intentionally dependency-light (numpy/scipy only) so they can
be dropped into any embedding pipeline (DINOv3, SigLIP-2, CLIP, or a
VLM's own vision tower) without committing to a specific model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, a_min=1e-12, a_max=None)


@dataclass
class DriftResult:
    is_drift: bool
    score: float
    threshold: float
    reference_mean_similarity: float
    batch_mean_similarity: float
    p_value: float | None = None
    details: dict = field(default_factory=dict)


class CosineDriftDetector:
    """Detect distribution shift via cosine similarity to a reference centroid.

    Reference embeddings are typically captured from a known-good
    validation set or early production traffic. New batches are
    compared via a two-sample Kolmogorov-Smirnov test on the
    similarity distributions, which is robust to non-Gaussian shapes
    common in real embedding spaces.
    """

    def __init__(self, reference_embeddings: np.ndarray, alpha: float = 0.01) -> None:
        if reference_embeddings.ndim != 2:
            raise ValueError("reference_embeddings must be (N, D)")
        self.reference_embeddings = _l2_normalize(np.asarray(reference_embeddings, dtype=np.float64))
        self.centroid = _l2_normalize(self.reference_embeddings.mean(axis=0, keepdims=True))[0]
        self.alpha = alpha
        self.reference_similarities = self.reference_embeddings @ self.centroid

    def score_batch(self, batch_embeddings: np.ndarray) -> DriftResult:
        batch = _l2_normalize(np.asarray(batch_embeddings, dtype=np.float64))
        batch_similarities = batch @ self.centroid

        ks_stat, p_value = stats.ks_2samp(self.reference_similarities, batch_similarities)

        is_drift = bool(p_value < self.alpha)
        return DriftResult(
            is_drift=is_drift,
            score=float(ks_stat),
            threshold=self.alpha,
            reference_mean_similarity=float(self.reference_similarities.mean()),
            batch_mean_similarity=float(batch_similarities.mean()),
            p_value=float(p_value),
            details={
                "reference_std": float(self.reference_similarities.std()),
                "batch_std": float(batch_similarities.std()),
                "batch_size": int(batch.shape[0]),
            },
        )


class EWMADriftDetector:
    """Online SPC-style drift monitor over a stream of scalar scores.

    Maintains an exponentially-weighted moving average of an incoming
    score stream (e.g. per-batch mean cosine similarity from
    ``CosineDriftDetector``) and flags points that fall outside
    control limits derived from a **frozen baseline** standard
    deviation — the same logic used in classic manufacturing SPC
    (Shewhart control charts with an EWMA-smoothed centerline),
    applied here to embedding drift.

    Design note: a naive implementation that continuously re-estimates
    variance from the incoming stream is self-defeating under a real
    step-change, because the jump itself inflates the variance
    estimate and widens the control limits just when they need to
    stay tight to catch it. This implementation instead estimates the
    in-control standard deviation once from the first ``baseline_n``
    points (assumed drift-free) and freezes it, so a later step-change
    is judged against a stable yardstick rather than one the shift
    itself has corrupted.
    """

    def __init__(self, lam: float = 0.2, n_sigma: float = 3.0, warmup: int = 5, baseline_n: int = 5) -> None:
        self.lam = lam
        self.n_sigma = n_sigma
        self.warmup = warmup
        self.baseline_n = baseline_n
        self._mean: float | None = None
        self._baseline_std: float | None = None
        self._history: list[float] = []

    def update(self, value: float) -> DriftResult:
        self._history.append(value)
        n = len(self._history)

        if self._mean is None:
            self._mean = value
        else:
            self._mean = self.lam * value + (1 - self.lam) * self._mean

        if self._baseline_std is None and n >= self.baseline_n:
            # Freeze the baseline std from the first `baseline_n` points only,
            # before any drift-driven EWMA update can have contaminated it.
            self._baseline_std = float(np.std(self._history[: self.baseline_n], ddof=0)) or 1e-6

        std = self._baseline_std if self._baseline_std is not None else float(np.std(self._history, ddof=0)) or 1e-6
        upper = self._mean + self.n_sigma * std
        lower = self._mean - self.n_sigma * std

        warmed_up = n > self.warmup and self._baseline_std is not None
        is_drift = bool(warmed_up and (value > upper or value < lower))

        return DriftResult(
            is_drift=is_drift,
            score=float(value),
            threshold=float(self.n_sigma * std),
            reference_mean_similarity=float(self._mean),
            batch_mean_similarity=float(value),
            details={
                "upper_control_limit": float(upper),
                "lower_control_limit": float(lower),
                "n": n,
                "baseline_std": std,
            },
        )


def select_for_active_learning(
    drift_results: list[DriftResult],
    embeddings: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """Rank samples by drift/uncertainty proxy and return indices of the top_k most informative.

    Uses distance-from-centroid as an uncertainty proxy: samples whose
    embeddings sit farthest from the reference centroid are most
    likely to be OOD or under-represented, and are prioritized for
    human labeling in an active learning loop.

    Limitation: the centroid is computed from ``embeddings`` itself
    (the batch being ranked), not from an external reference set. This
    is the right behavior for the shipped use case (an entire batch
    has drifted together, and this ranks samples *within* that batch
    by how far they've moved), but it means a single extreme outlier
    within an otherwise tight, stable batch can paradoxically appear
    *more* similar to the centroid rather than less, once its own
    magnitude is large enough to dominate the mean used to compute
    that centroid. If you need to rank samples against a fixed,
    externally-defined reference distribution, compute the centroid
    from a separate, uncontaminated reference set and pass in cosine
    similarities directly rather than relying on this function's
    internal centroid.
    """
    if len(drift_results) == 0:
        return np.array([], dtype=int)
    centroid = _l2_normalize(embeddings.mean(axis=0, keepdims=True))[0]
    sims = _l2_normalize(embeddings) @ centroid
    order = np.argsort(sims)  # ascending similarity == descending novelty
    return order[: min(top_k, len(order))]
