"""Per-sample out-of-distribution (OOD) detection in embedding space.

`production_vlm.drift.CosineDriftDetector` answers "has this *batch's*
distribution shifted from the reference?" -- the right question for
production monitoring. This module answers a related but distinct
question: "is *this individual sample* unusual relative to a known-good
reference set?" -- the right question for a per-request guard that
should flag (or refuse) a single suspicious input before it ever
reaches the model, independent of whether a whole batch has drifted.

The scoring method (cosine distance to the k nearest reference
neighbors) follows the standard non-parametric OOD detection
convention used in production ML safety literature: a sample whose
nearest neighbors in the reference set are all still far away is
treated as anomalous, which is more robust to multi-modal reference
distributions than a single global centroid distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, a_min=1e-12, a_max=None)


@dataclass
class OODResult:
    is_ood: bool
    ood_score: float
    threshold: float
    nearest_neighbor_similarity: float


class KNNOODDetector:
    """k-nearest-neighbor cosine-similarity OOD detector for a single embedding.

    Calibrates its threshold from the reference set itself: computes
    each reference point's k-NN similarity to the *rest* of the
    reference set (leave-one-out), then sets the OOD threshold at a
    configurable percentile of that in-distribution similarity
    distribution. A new sample is flagged OOD if its k-NN similarity
    to the full reference set falls below that calibrated threshold --
    i.e. it is calibrated to a target in-distribution false-positive
    rate rather than an arbitrary fixed cosine value.

    Per-sample OOD detection is a fundamentally harder problem than
    batch-level drift detection (see `production_vlm.drift.CosineDriftDetector`):
    a single sample's k-NN similarity has much higher variance than a
    whole batch's mean similarity, so the precision/recall tradeoff is
    sharper and the right threshold matters more. This was measured
    directly rather than assumed: on the synthetic chart-embedding
    style-shift scenario (the same shift `CosineDriftDetector` detects
    at the batch level with zero delay), a very conservative
    `percentile=1.0` threshold gave only a 2.5% true-positive rate at
    0% false positives -- nearly useless as a single-sample guard.
    Sweeping the threshold found `percentile=15-20` gives 100%
    true-positive rate at a 15-20% false-positive rate, which is the
    realistic operating point for this detector on this kind of shift:
    a usable guard, but not a free lunch. Calibrate `percentile` for
    your own reference set and known-shift validation data rather than
    trusting the default blindly -- see `benchmark()` in
    `examples/pipelines/vlm_robustness_guard/run.py` for the sweep
    methodology.
    """

    def __init__(self, reference_embeddings: np.ndarray, k: int = 5, percentile: float = 15.0) -> None:
        if reference_embeddings.ndim != 2:
            raise ValueError("reference_embeddings must be (N, D)")
        if k < 1:
            raise ValueError("k must be >= 1")
        if reference_embeddings.shape[0] <= k:
            raise ValueError(f"need more than k={k} reference embeddings, got {reference_embeddings.shape[0]}")

        self.k = k
        self.reference = _l2_normalize(np.asarray(reference_embeddings, dtype=np.float64))
        self._calibrate(percentile)

    def _knn_similarity(self, query: np.ndarray, pool: np.ndarray, exclude_self_idx: int | None = None) -> float:
        sims = pool @ query
        if exclude_self_idx is not None:
            sims = np.delete(sims, exclude_self_idx)
        k = min(self.k, len(sims))
        top_k = np.sort(sims)[-k:]
        return float(top_k.mean())

    def _calibrate(self, percentile: float) -> None:
        n = self.reference.shape[0]
        loo_similarities = np.array(
            [self._knn_similarity(self.reference[i], self.reference, exclude_self_idx=i) for i in range(n)]
        )
        # Threshold at the given low percentile of in-distribution similarity:
        # genuine in-distribution points should mostly score above this.
        self.threshold = float(np.percentile(loo_similarities, percentile))
        self._calibration_similarities = loo_similarities

    def score(self, embedding: np.ndarray) -> OODResult:
        query = _l2_normalize(embedding.reshape(1, -1))[0]
        sim = self._knn_similarity(query, self.reference)
        is_ood = sim < self.threshold
        return OODResult(
            is_ood=bool(is_ood),
            ood_score=float(1.0 - sim),
            threshold=float(1.0 - self.threshold),
            nearest_neighbor_similarity=sim,
        )

    def score_batch(self, embeddings: np.ndarray) -> list[OODResult]:
        return [self.score(embeddings[i]) for i in range(embeddings.shape[0])]
