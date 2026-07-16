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
from production_vlm.utils.observability import ObservabilityLogger  # noqa: E402
from production_vlm.utils.retraining import QueuedSample, RetrainingTrigger  # noqa: E402
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

    # Regression test: `from __future__ import annotations` makes dataclass
    # annotations lazy strings at runtime, which broke Literal validation via
    # a naive `cls.__annotations__` read (get_origin() on a string silently
    # returns None instead of Literal, so no error was ever raised). Fixed by
    # resolving via typing.get_type_hints(). This check ensures the fix holds.
    try:
        bad3 = dict(raw)
        bad3["model"] = {**raw["model"], "dtype": "int4"}
        ExperimentConfig.from_dict(bad3)
        check("invalid dtype literal raises ConfigError", False)
    except ConfigError:
        check("invalid dtype literal raises ConfigError", True)

    try:
        bad4 = dict(raw)
        bad4["train"] = {**raw.get("train", {}), "logging": "not_a_real_backend"}
        ExperimentConfig.from_dict(bad4)
        check("invalid logging literal raises ConfigError", False)
    except ConfigError:
        check("invalid logging literal raises ConfigError", True)


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

    # Regression test: baseline_n=5 (too small a calibration sample for a
    # 3-sigma band) produced a false alarm on genuinely stable data because
    # the std estimate was off by 2.6x on this exact seed. baseline_n=10
    # (matching standard SPC calibration-phase guidance) was verified
    # false-alarm-free across 20 seeds.
    ewma2 = EWMADriftDetector(lam=0.3, n_sigma=3.0, warmup=3, baseline_n=10)
    stable_rng = np.random.default_rng(0)
    stable_long = 0.8 + stable_rng.normal(0, 0.005, size=20)
    stable_flags = [ewma2.update(v).is_drift for v in stable_long]
    check("ewma detector: no false alarm on stable signal (baseline_n=10)", not any(stable_flags[10:]))

    al_selected = select_for_active_learning([result_drift], shifted, top_k=5)
    check("active learning: returns requested top_k", len(al_selected) == 5)

    # Regression test: a constant offset (+100.0) added to one sample's every
    # dimension, with the centroid computed from the same batch containing
    # it, made the "outlier" pull the centroid toward itself and appear
    # *more* similar to it, not less -- no magnitude of constant/directional
    # perturbation on a high-variance batch ever produced a rank-0 outlier.
    # Fixed test construction: a tight, low-variance cluster (stable
    # in-distribution embeddings) plus one genuinely distinct point, which
    # reliably ranks as most novel.
    cluster_rng = np.random.default_rng(0)
    base_point = cluster_rng.normal(size=32)
    al_batch = base_point + cluster_rng.normal(0, 0.05, size=(50, 32))
    outlier_dir = np.random.default_rng(1).normal(size=32)
    outlier_dir /= np.linalg.norm(outlier_dir)
    al_batch[0] = base_point + 1.0 * outlier_dir
    al_result = detector.score_batch(al_batch)
    al_selected2 = select_for_active_learning([al_result], al_batch, top_k=1)
    check("active learning: selects genuine outlier as most novel", 0 in al_selected2)


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
    check(
        "batching: flushes on max_batch_size",
        all(r[1] == 4 for r in results) and batches_served == 1 and items_served == 4,
    )

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


def test_observability_retraining() -> None:
    print("test_observability_retraining")
    import tempfile
    from pathlib import Path

    # ObservabilityLogger
    with tempfile.TemporaryDirectory() as td:
        lp = Path(td) / "events.jsonl"
        logger = ObservabilityLogger(lp, run_id="verify_test")
        logger.log_drift_event(0, 25, False, False, 0.1, 0.8, 0.76)
        logger.log_drift_event(1, 25, True, False, 0.55, 1e-7, 0.62, al_selected_count=5)
        check("obs: log file created", lp.exists())
        events = logger.read_all()
        check("obs: two events written", len(events) == 2)
        check("obs: schema_version present", all(e["schema_version"] == "1.0" for e in events))
        check("obs: seq increments", [e["seq"] for e in events] == [0, 1])
        check("obs: drift event fields", events[0]["event_type"] == "drift_check")
        check("obs: al_selected_count", events[1]["al_selected_count"] == 5)
        logger.log_ood_event(True, 0.72, 0.28, 0.35)
        logger.log_guard_event("reject", 0.15, 0.0, 0.30)
        summary = logger.summary()
        check("obs: summary drift batches", summary["drift"]["total_batches"] == 2)
        check("obs: summary ood flagged", summary["ood"]["flagged"] == 1)
        check("obs: summary guard rejected", summary["guard"]["rejected"] == 1)

    # RetrainingTrigger
    import numpy as np

    def make_sample(i):
        return QueuedSample(i, i // 5, float(np.random.default_rng(i).uniform()), "drift_ks")

    # Single enqueue fires at threshold
    fired = []
    t = RetrainingTrigger(queue_threshold=5, callback=lambda s: fired.append(len(s)), cooldown_s=0)
    for i in range(5):
        t.enqueue(make_sample(i))
    check("retrain: fires at threshold", fired == [5])
    check("retrain: queue cleared after fire", t.queue_size == 0)

    # Below threshold: no fire
    t2 = RetrainingTrigger(queue_threshold=10, callback=lambda s: None, cooldown_s=0)
    for i in range(4):
        t2.enqueue(make_sample(i))
    check("retrain: no fire below threshold", t2.queue_size == 4)

    # Batch enqueue fires multiple times correctly
    fired3 = []
    t3 = RetrainingTrigger(queue_threshold=3, callback=lambda s: fired3.append(len(s)), cooldown_s=0)
    t3.enqueue_batch([make_sample(i) for i in range(9)])
    check("retrain: batch fires 3x3 not 1x9", fired3 == [3, 3, 3])

    # Cooldown blocks second fire
    fired4 = []
    t4 = RetrainingTrigger(queue_threshold=3, callback=lambda s: fired4.append(1), cooldown_s=999)
    t4.enqueue_batch([make_sample(i) for i in range(9)])
    check("retrain: cooldown blocks multi-fire", len(fired4) == 1)

    # Error in callback recorded, not propagated
    def bad_cb(s):
        raise RuntimeError("fail")

    t5 = RetrainingTrigger(queue_threshold=3, callback=bad_cb, cooldown_s=0)
    for i in range(3):
        t5.enqueue(make_sample(i))
    h = t5.trigger_history
    check("retrain: error in callback recorded", len(h) == 1 and not h[0].success)

    # Invalid threshold raises
    try:
        RetrainingTrigger(queue_threshold=0)
        check("retrain: invalid threshold raises", False)
    except ValueError:
        check("retrain: invalid threshold raises", True)


def test_structured_extraction() -> None:
    print("test_structured_extraction")
    import importlib.util

    # Load the chart finetune run module without executing __main__
    spec = importlib.util.spec_from_file_location(
        "ft_run", str(Path(__file__).parent.parent / "examples" / "pipelines" / "vlm_chart_finetune" / "run.py")
    )
    ft_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ft_mod)

    charts = [generate_synthetic_chart(seed=i, chart_type="bar", render_image=False) for i in range(20)]

    # Every ground-truth extraction is schema-valid
    schema_keys = ft_mod._CHART_JSON_SCHEMA["required"]
    for c in charts[:5]:
        ex = ft_mod._extract_structured_json(c)
        check("struct: schema keys present", all(k in ex for k in schema_keys))
        check("struct: correct chart_type", ex["chart_type"] == c.chart_type)
        check("struct: series length matches categories", len(ex["series"]) == len(c.categories))

    # Fine-tuned (ground-truth) mode: perfect metrics
    r_fine = ft_mod._structured_extraction_accuracy(charts, noise_zero_shot=False)
    check("struct: finetuned schema validity 100%", r_fine["schema_validity_rate"] == 1.0)
    check("struct: finetuned numeric MAPE 0%", r_fine["numeric_extraction_mape"] == 0.0)
    check("struct: finetuned category coverage 100%", r_fine["category_coverage"] == 1.0)

    # Zero-shot mode: degraded (simulated errors injected)
    r_zero = ft_mod._structured_extraction_accuracy(charts, noise_zero_shot=True)
    check("struct: zero-shot schema validity <100%", r_zero["schema_validity_rate"] < 1.0)
    check("struct: zero-shot MAPE >0%", r_zero["numeric_extraction_mape"] > 0.0)


def test_kv_cache() -> None:
    print("test_kv_cache")
    from production_vlm.utils.kv_cache import (
        AttentionStrategy,
        ModelDecoderConfig,
        compare_strategies,
        compute_kv_cache_memory,
        visual_token_count,
    )

    check(
        "kv_cache: 336px visual tokens = 576 (SigLIP/LLaVA convention)", visual_token_count(336, patch_size=14) == 576
    )

    cfg = ModelDecoderConfig(n_query_heads=28, n_kv_heads_gqa=4)
    mha = compute_kv_cache_memory(cfg, AttentionStrategy.MHA, seq_len=1000)
    gqa = compute_kv_cache_memory(cfg, AttentionStrategy.GQA, seq_len=1000)
    mqa = compute_kv_cache_memory(cfg, AttentionStrategy.MQA, seq_len=1000)
    check("kv_cache: MHA relative_to_mha == 1.0", abs(mha.relative_to_mha - 1.0) < 1e-6)
    check("kv_cache: GQA reduces memory by head ratio", abs(gqa.relative_to_mha - 4 / 28) < 1e-6)
    check("kv_cache: memory ordering MHA > GQA > MQA", mha.kv_cache_mb > gqa.kv_cache_mb > mqa.kv_cache_mb)

    sw_cfg = ModelDecoderConfig(sliding_window_size=512)
    at_window = compute_kv_cache_memory(sw_cfg, AttentionStrategy.SLIDING_WINDOW, seq_len=512)
    beyond_window = compute_kv_cache_memory(sw_cfg, AttentionStrategy.SLIDING_WINDOW, seq_len=2000)
    check(
        "kv_cache: sliding window caps memory beyond window size",
        abs(at_window.kv_cache_mb - beyond_window.kv_cache_mb) < 1e-6,
    )

    results = compare_strategies(cfg, seq_lens=[400, 800, 1200])
    check(
        "kv_cache: compare_strategies returns all 4 strategies",
        set(results.keys()) == {"mha", "gqa", "mqa", "sliding_window"},
    )


def main() -> int:
    suites = [
        test_config,
        test_eval,
        test_drift,
        test_synthetic_charts,
        test_vision_encoder,
        test_batching_queue,
        test_robustness,
        test_observability_retraining,
        test_structured_extraction,
        test_kv_cache,
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
