#!/usr/bin/env python
"""Stdlib-only smoke verification, mirroring tests/*.py assertions without requiring pytest.

This script exists because some environments (offline CI runners,
restricted sandboxes) may not have network access to install
pytest/pytest-asyncio. It exercises the same core invariants as the
real pytest suite under tests/ using only the standard library plus
the project's own runtime dependencies (numpy/scipy/pyyaml), so the
package's correctness can still be verified end-to-end.

This is NOT a replacement for `pytest tests/` -- run that in any
environment with network access (`pip install -e ".[dev]"` first) for
the canonical, fuller test suite. This script is a reduced-dependency
fallback verifier only.

Run:
    python scripts/verify_no_pytest.py
"""

from __future__ import annotations

import asyncio
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from production_vlm.config import ConfigError, ExperimentConfig  # noqa: E402
from production_vlm.drift import CosineDriftDetector, EWMADriftDetector, select_for_active_learning  # noqa: E402
from production_vlm.eval import faithfulness_score, grounding_score, numeric_accuracy  # noqa: E402
from production_vlm.robustness import (  # noqa: E402
    GuardDecision,
    HallucinationGuard,
    KNNOODDetector,
    NaturalPerturbation,
    apply_perturbation,
)
from production_vlm.robustness.chart_reader import read_tallest_bar  # noqa: E402
from production_vlm.utils.batching_queue import BatchingQueue  # noqa: E402
from production_vlm.utils.synthetic_charts import generate_dataset, generate_synthetic_chart  # noqa: E402
from production_vlm.utils.vision_encoder import SyntheticEmbeddingProxy  # noqa: E402

_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(name: str, condition: bool) -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(name)
        print(f"  FAIL: {name}")


def run_async(coro):
    return asyncio.run(coro)


def test_config() -> None:
    print("test_config")
    raw = {
        "name": "t",
        "model": {"checkpoint": "org/model", "checkpoint_pinned_date": "2026-01-01"},
        "lora": {"rank": 16, "alpha": 32},
        "data": {"dataset_name": "x", "dataset_pinned_date": "n/a", "max_samples": 50},
        "train": {"epochs": 1, "learning_rate": 2e-4},
        "eval": {"num_eval_samples": 10},
    }
    cfg = ExperimentConfig.from_dict(raw)
    check("config parses valid input", cfg.name == "t" and cfg.lora.rank == 16)

    try:
        bad = dict(raw)
        bad["lora"] = {"rank": 64, "alpha": 8}
        ExperimentConfig.from_dict(bad)
        check("alpha<rank raises ConfigError", False)
    except ConfigError:
        check("alpha<rank raises ConfigError", True)

    try:
        bad2 = {k: v for k, v in raw.items() if k != "model"}
        ExperimentConfig.from_dict(bad2)
        check("missing model key raises ConfigError", False)
    except ConfigError:
        check("missing model key raises ConfigError", True)


def test_eval() -> None:
    print("test_eval")
    r = numeric_accuracy("Revenue was $4.21M", "Revenue was $4.2M", tolerance=0.02)
    check("numeric_accuracy within tolerance", r.score == 1.0)

    r2 = numeric_accuracy("Revenue was $10M", "Revenue was $4.2M", tolerance=0.02)
    check("numeric_accuracy outside tolerance fails", r2.score == 0.0)

    g = grounding_score("revenue increased sharply", "revenue increased sharply this quarter")
    check("grounding_score full overlap", g.score == 1.0)

    f = faithfulness_score(
        prediction="The value appears moderate",
        reference="South has throughput of 67.6 req/s, which is the highest.",
        evidence_text="South: 67.6 req/s",
    )
    check("faithfulness_score weak answer scores low", f.score < 0.5)


def test_drift() -> None:
    print("test_drift")
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(200, 32))

    detector = CosineDriftDetector(reference, alpha=0.01)
    idx = rng.choice(len(reference), size=40, replace=False)
    result_no_drift = detector.score_batch(reference[idx])
    check("cosine detector: no false alarm on resample", result_no_drift.is_drift is False)

    shift_rng = np.random.default_rng(99)
    shift = shift_rng.normal(size=32)
    shift /= np.linalg.norm(shift)
    shifted = reference[:40] + 20.0 * shift
    result_drift = detector.score_batch(shifted)
    check("cosine detector: detects injected shift", result_drift.is_drift is True)

    ewma = EWMADriftDetector(lam=0.3, n_sigma=2.0, warmup=4, baseline_n=6)
    stable = list(0.8 + rng.normal(0, 0.005, size=6))
    shifted_signal = list(0.6 + rng.normal(0, 0.005, size=6))
    flags = [ewma.update(v).is_drift for v in stable + shifted_signal]
    check("ewma detector: flags onset of step change", flags[6] is True)

    al_selected = select_for_active_learning([result_drift], shifted, top_k=5)
    check("active learning: returns requested top_k", len(al_selected) == 5)


def test_synthetic_charts() -> None:
    print("test_synthetic_charts")
    c1 = generate_synthetic_chart(seed=7, render_image=False)
    c2 = generate_synthetic_chart(seed=7, render_image=False)
    check("synthetic chart reproducible by seed", c1.categories == c2.categories and c1.values == c2.values)

    region_group = {"North", "South", "East", "West", "Central"}
    quarter_group = {"Q1", "Q2", "Q3", "Q4"}
    market_group = {"EU", "US", "APAC", "LATAM"}
    coherent = all(
        set(generate_synthetic_chart(seed=s, render_image=False).categories) <= region_group
        or set(generate_synthetic_chart(seed=s, render_image=False).categories) <= quarter_group
        or set(generate_synthetic_chart(seed=s, render_image=False).categories) <= market_group
        for s in range(20)
    )
    check("synthetic chart categories from coherent group", coherent)

    rendered = generate_synthetic_chart(seed=1, render_image=True)
    check("render_image=True produces a PIL image", rendered.image is not None and rendered.image.mode == "RGB")

    ds = generate_dataset(n=15, seed=0, render_image=False)
    check("generate_dataset returns requested count", len(ds) == 15)


def test_vision_encoder() -> None:
    print("test_vision_encoder")
    encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=12.0)
    normal_charts = [generate_synthetic_chart(seed=i, render_image=False) for i in range(60)]
    normal_emb = encoder.encode_charts(normal_charts, style_shift_flags=[False] * 60)
    shifted_charts = [generate_synthetic_chart(seed=1000 + i, style_shift=True, render_image=False) for i in range(60)]
    shifted_emb = encoder.encode_charts(shifted_charts, style_shift_flags=[True] * 60)

    centroid = normal_emb.mean(axis=0)
    normal_dist = np.linalg.norm(normal_emb - centroid, axis=1).mean()
    shifted_dist = np.linalg.norm(shifted_emb - centroid, axis=1).mean()
    check("style shift increases distance from centroid", shifted_dist > normal_dist * 1.2)

    encoder2a = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=42)
    encoder2b = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=42)
    chart = generate_synthetic_chart(seed=1, render_image=False)
    e1 = encoder2a.encode_charts([chart], style_shift_flags=[False])
    e2 = encoder2b.encode_charts([chart], style_shift_flags=[False])
    check("encoder deterministic given same seed", np.array_equal(e1, e2))


def test_batching_queue() -> None:
    print("test_batching_queue")

    def predict_fn(batch: np.ndarray) -> np.ndarray:
        return batch.sum(axis=1, keepdims=True) * np.ones((1, 4))

    try:
        BatchingQueue(predict_fn, max_batch_size=0, max_batch_wait_ms=10)
        check("rejects invalid max_batch_size", False)
    except ValueError:
        check("rejects invalid max_batch_size", True)

    async def _flush_on_size():
        queue = BatchingQueue(predict_fn, max_batch_size=4, max_batch_wait_ms=1000)
        queue.start()

        async def submit_one(i):
            return await queue.submit(np.full((3,), i, dtype=np.float32))

        results = await asyncio.gather(*[submit_one(i) for i in range(4)])
        await queue.stop()
        return results, queue.batches_served, queue.items_served

    results, batches_served, items_served = run_async(_flush_on_size())
    check("batching: flushes on max_batch_size", all(r[1] == 4 for r in results) and batches_served == 1 and items_served == 4)

    async def _flush_on_timeout():
        queue = BatchingQueue(predict_fn, max_batch_size=10, max_batch_wait_ms=30)
        queue.start()
        start = time.perf_counter()
        output, batch_size, *_ = await queue.submit(np.full((3,), 1.0, dtype=np.float32))
        elapsed_ms = (time.perf_counter() - start) * 1000
        await queue.stop()
        return batch_size, elapsed_ms

    batch_size, elapsed_ms = run_async(_flush_on_timeout())
    check("batching: lone request flushes after timeout", batch_size == 1 and elapsed_ms >= 25)

    async def _output_correctness():
        queue = BatchingQueue(predict_fn, max_batch_size=4, max_batch_wait_ms=50)
        queue.start()

        async def submit_one(value):
            arr = np.full((5,), value, dtype=np.float32)
            output, *_ = await queue.submit(arr)
            return value, output

        results = await asyncio.gather(*[submit_one(v) for v in [1.0, 2.0, 3.0]])
        await queue.stop()
        return results

    results = run_async(_output_correctness())
    check("batching: per-item output correctness", all(np.allclose(out, val * 5) for val, out in results))


def test_robustness() -> None:
    print("test_robustness")

    # Perturbation: severity 0 minimal change, severity 1 meaningful change
    chart = generate_synthetic_chart(seed=1, chart_type="bar", render_image=True)
    img = chart.image
    for kind in NaturalPerturbation.ALL:
        r0 = apply_perturbation(img, kind, 0.0)
        diff0 = np.abs(np.asarray(img, float) - np.asarray(r0.perturbed_image, float)).mean()
        check(f"perturbation {kind} sev=0 minimal change", diff0 <= 1.0)
        r1 = apply_perturbation(img, kind, 1.0)
        diff1 = np.abs(np.asarray(img, float) - np.asarray(r1.perturbed_image, float)).mean()
        check(f"perturbation {kind} sev=1 changes image", diff1 > 1.0)

    # Chart reader: 100% clean baseline
    correct = 0
    for i in range(15):
        c = generate_synthetic_chart(seed=i, chart_type="bar", render_image=True)
        r = read_tallest_bar(c.image, len(c.categories), int(np.argmax(c.values)), plot_bbox=c.plot_bbox)
        correct += r.correct
    check("chart reader 100% clean baseline", correct == 15)

    # Chart reader: brightness-robust (adaptive background detection)
    correct_bright = 0
    for i in range(15):
        c = generate_synthetic_chart(seed=i, chart_type="bar", render_image=True)
        p = apply_perturbation(c.image, "brightness", 0.8, seed=i)
        r = read_tallest_bar(p.perturbed_image, len(c.categories), int(np.argmax(c.values)), plot_bbox=c.plot_bbox)
        correct_bright += r.correct
    check("chart reader robust to brightness sev=0.8", correct_bright == 15)

    # OOD detector: calibrated FP/TP
    encoder = SyntheticEmbeddingProxy(embedding_dim=64, seed=0, shift_magnitude=12.0)
    ref_charts = [generate_synthetic_chart(seed=i, render_image=False) for i in range(150)]
    ref_emb = encoder.encode_charts(ref_charts, style_shift_flags=[False] * 150)
    ood_det = KNNOODDetector(ref_emb, k=5, percentile=15.0)
    holdout_emb = encoder.encode_charts(
        [generate_synthetic_chart(seed=2000 + i, render_image=False) for i in range(40)], [False] * 40
    )
    shifted_emb = encoder.encode_charts(
        [generate_synthetic_chart(seed=3000 + i, style_shift=True, render_image=False) for i in range(40)], [True] * 40
    )
    fp = sum(r.is_ood for r in ood_det.score_batch(holdout_emb)) / 40
    tp = sum(r.is_ood for r in ood_det.score_batch(shifted_emb)) / 40
    check("OOD detector FP rate <= 25%", fp <= 0.25)
    check("OOD detector TP rate >= 85%", tp >= 0.85)

    # Hallucination guard
    guard = HallucinationGuard()
    good = guard.check(
        "South shows throughput of 67.6 req/s, making it the highest.",
        "South has throughput of 67.6 req/s, which is the highest.",
        "South: 67.6 req/s; North: 50.0 req/s",
    )
    check("guard passes correct answer", good.decision == GuardDecision.PASS)
    bad = guard.check(
        "North has throughput of 999.9 which is the highest.",
        "South has throughput of 67.6 req/s, which is the highest.",
        "South: 67.6 req/s",
    )
    check("guard rejects hallucinated answer", bad.decision == GuardDecision.REJECT)
    check("guard returns fallback message on reject", bad.output_text == guard.config.fallback_message)


def main() -> int:
    suites = [
        test_config, test_eval, test_drift, test_synthetic_charts,
        test_vision_encoder, test_batching_queue, test_robustness,
    ]
    for suite in suites:
        try:
            suite()
        except Exception:
            global _FAIL
            _FAIL += 1
            _FAILURES.append(f"{suite.__name__} raised an exception")
            print(f"  EXCEPTION in {suite.__name__}:")
            traceback.print_exc()

    print(f"\n{_PASS} passed, {_FAIL} failed")
    if _FAILURES:
        print("Failures:")
        for f in _FAILURES:
            print(f"  - {f}")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
