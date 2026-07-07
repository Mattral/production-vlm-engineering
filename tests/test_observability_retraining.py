"""Unit tests for production observability, retraining trigger, and structured JSON extraction."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from production_vlm.utils.observability import ObservabilityLogger, PrometheusMetricsServer
from production_vlm.utils.retraining import QueuedSample, RetrainingTrigger

# ---------------------------------------------------------------------------
# ObservabilityLogger tests
# ---------------------------------------------------------------------------


class TestObservabilityLogger:
    @pytest.fixture
    def log_path(self, tmp_path) -> Path:
        return tmp_path / "events.jsonl"

    def test_creates_log_file_on_first_write(self, log_path):
        logger = ObservabilityLogger(log_path)
        logger.log_drift_event(0, 25, False, False, 0.1, 0.8, 0.76)
        assert log_path.exists()

    def test_each_event_is_valid_json_line(self, log_path):
        logger = ObservabilityLogger(log_path, run_id="test_run")
        logger.log_drift_event(0, 25, False, False, 0.1, 0.8, 0.76)
        logger.log_drift_event(1, 25, True, False, 0.55, 1e-7, 0.62)
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert all(e["schema_version"] == "1.0" for e in parsed)
        assert all(e["run_id"] == "test_run" for e in parsed)

    def test_seq_increments_per_event(self, log_path):
        logger = ObservabilityLogger(log_path)
        for i in range(5):
            logger.log_drift_event(i, 25, False, False, 0.1, 0.8, 0.75)
        events = logger.read_all()
        assert [e["seq"] for e in events] == list(range(5))

    def test_drift_event_has_required_fields(self, log_path):
        logger = ObservabilityLogger(log_path)
        logger.log_drift_event(
            batch_idx=3,
            batch_size=50,
            is_drift_ks=True,
            is_drift_ewma=False,
            ks_stat=0.58,
            p_value=2e-7,
            batch_mean_similarity=0.62,
            al_selected_count=5,
            extra={"true_drift_injected": True},
        )
        events = logger.read_all()
        e = events[0]
        assert e["event_type"] == "drift_check"
        assert e["batch_idx"] == 3
        assert e["is_drift_ks"] is True
        assert e["al_selected_count"] == 5
        assert e["extra"]["true_drift_injected"] is True

    def test_ood_event_logged(self, log_path):
        logger = ObservabilityLogger(log_path)
        logger.log_ood_event(is_ood=True, ood_score=0.72, nearest_neighbor_similarity=0.28, threshold=0.35)
        e = logger.read_all()[0]
        assert e["event_type"] == "ood_check"
        assert e["is_ood"] is True

    def test_guard_event_logged(self, log_path):
        logger = ObservabilityLogger(log_path)
        logger.log_guard_event("reject", 0.15, 0.0, 0.30)
        e = logger.read_all()[0]
        assert e["event_type"] == "guard_check"
        assert e["decision"] == "reject"

    def test_summary_counts_correctly(self, log_path):
        logger = ObservabilityLogger(log_path)
        for i in range(4):
            logger.log_drift_event(
                i,
                25,
                is_drift_ks=(i >= 2),
                is_drift_ewma=False,
                ks_stat=0.1,
                p_value=0.5,
                batch_mean_similarity=0.75,
                al_selected_count=3 if i >= 2 else 0,
            )
        logger.log_ood_event(True, 0.7, 0.3, 0.35)
        logger.log_guard_event("pass", 0.8, 0.9, 0.65)
        summary = logger.summary()
        assert summary["drift"]["total_batches"] == 4
        assert summary["drift"]["drift_flagged_ks"] == 2
        assert summary["drift"]["al_total_queued"] == 6
        assert summary["ood"]["flagged"] == 1
        assert summary["guard"]["passed"] == 1

    def test_thread_safety(self, log_path):
        logger = ObservabilityLogger(log_path)
        errors = []

        def writer():
            try:
                for i in range(10):
                    logger.log_drift_event(i, 25, False, False, 0.1, 0.5, 0.75)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        events = logger.read_all()
        assert len(events) == 50  # 5 threads × 10 events


# ---------------------------------------------------------------------------
# PrometheusMetricsServer tests (no-op when prometheus_client not installed)
# ---------------------------------------------------------------------------


class TestPrometheusMetricsServer:
    def test_instantiates_without_prometheus_client(self):
        """Should never raise even when prometheus_client is absent."""
        server = PrometheusMetricsServer(port=0)
        assert server is not None

    def test_start_is_safe_without_prometheus_client(self, capsys):
        server = PrometheusMetricsServer(port=9999)
        server.start()  # should not raise; prints a warning if not installed

    def test_record_methods_are_no_ops_without_prometheus_client(self):
        server = PrometheusMetricsServer(port=0)
        server.record_drift(ks_stat=0.5, is_drift_ks=True, is_drift_ewma=False, batch_mean_similarity=0.6, al_queued=3)
        server.record_ood(is_ood=True, ood_score=0.7)
        server.record_guard(decision="reject", faithfulness=0.1)
        # No assertion needed — just confirming no exception raised

    def test_available_flag_reflects_install_state(self):
        server = PrometheusMetricsServer(port=0)
        # We can't guarantee prometheus_client is installed in this env,
        # but the flag should be a boolean either way
        assert isinstance(server._available, bool)


# ---------------------------------------------------------------------------
# RetrainingTrigger tests
# ---------------------------------------------------------------------------


class TestRetrainingTrigger:
    def _make_sample(self, i: int) -> QueuedSample:
        return QueuedSample(
            embedding_index=i,
            batch_idx=i // 5,
            novelty_score=float(np.random.default_rng(i).uniform()),
            flagged_by="drift_ks",
        )

    def test_rejects_invalid_threshold(self):
        with pytest.raises(ValueError):
            RetrainingTrigger(queue_threshold=0)

    def test_fires_when_threshold_reached(self):
        fired = []
        trigger = RetrainingTrigger(queue_threshold=5, callback=lambda s: fired.append(len(s)), cooldown_s=0)
        for i in range(5):
            trigger.enqueue(self._make_sample(i))
        assert len(fired) == 1
        assert fired[0] == 5

    def test_does_not_fire_below_threshold(self):
        fired = []
        trigger = RetrainingTrigger(queue_threshold=10, callback=lambda s: fired.append(len(s)), cooldown_s=0)
        for i in range(4):
            trigger.enqueue(self._make_sample(i))
        assert len(fired) == 0
        assert trigger.queue_size == 4

    def test_queue_cleared_after_firing(self):
        trigger = RetrainingTrigger(queue_threshold=3, callback=lambda s: None, cooldown_s=0)
        for i in range(3):
            trigger.enqueue(self._make_sample(i))
        assert trigger.queue_size == 0

    def test_multiple_fires_on_large_batch(self):
        fired = []
        trigger = RetrainingTrigger(queue_threshold=3, callback=lambda s: fired.append(len(s)), cooldown_s=0)
        trigger.enqueue_batch([self._make_sample(i) for i in range(9)])
        assert fired == [3, 3, 3]  # three full batches of 3, not one batch of 9
        assert trigger.queue_size == 0

    def test_cooldown_prevents_immediate_refiring(self):
        fired = []
        trigger = RetrainingTrigger(queue_threshold=3, callback=lambda s: fired.append(1), cooldown_s=999)
        for i in range(6):
            trigger.enqueue(self._make_sample(i))
        assert len(fired) == 1  # only first batch fires; second blocked by cooldown

    def test_summary_reflects_history(self):
        trigger = RetrainingTrigger(queue_threshold=4, callback=lambda s: None, cooldown_s=0)
        for i in range(8):
            trigger.enqueue(self._make_sample(i))
        s = trigger.summary()
        assert s["trigger_count"] == 2
        assert len(s["runs"]) == 2
        assert all(r["n_samples"] == 4 for r in s["runs"])
        assert all(r["success"] for r in s["runs"])

    def test_callback_error_recorded_in_history(self):
        def bad_callback(samples):
            raise RuntimeError("simulated training failure")

        trigger = RetrainingTrigger(queue_threshold=3, callback=bad_callback, cooldown_s=0)
        for i in range(3):
            trigger.enqueue(self._make_sample(i))

        history = trigger.trigger_history
        assert len(history) == 1
        assert history[0].success is False
        assert "simulated training failure" in history[0].error

    def test_thread_safety_concurrent_enqueue(self):
        fired_counts = []
        lock = threading.Lock()

        def callback(samples):
            with lock:
                fired_counts.append(len(samples))

        trigger = RetrainingTrigger(queue_threshold=10, callback=callback, cooldown_s=0)

        def enqueuer(start):
            for i in range(5):
                trigger.enqueue(self._make_sample(start + i))

        threads = [threading.Thread(target=enqueuer, args=(i * 5,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 50 samples at threshold=10 → should produce approximately 5 fires
        assert trigger.trigger_count >= 4  # at least 40 samples processed into fires
        assert all(c == 10 for c in fired_counts)


# ---------------------------------------------------------------------------
# Structured JSON extraction tests
# ---------------------------------------------------------------------------


class TestStructuredExtraction:
    def test_schema_valid_on_ground_truth(self):
        """Extraction from ground-truth metadata must always produce a valid schema."""
        import sys

        ft_dir = Path(__file__).parents[1] / "examples" / "pipelines" / "vlm_chart_finetune"
        sys.path.insert(0, str(ft_dir))
        from run import _CHART_JSON_SCHEMA, _extract_structured_json

        from production_vlm.utils.synthetic_charts import generate_synthetic_chart

        for seed in range(10):
            chart = generate_synthetic_chart(seed=seed, render_image=False)
            extracted = _extract_structured_json(chart)
            assert all(k in extracted for k in _CHART_JSON_SCHEMA["required"])
            assert extracted["chart_type"] == chart.chart_type
            assert len(extracted["series"]) == len(chart.categories)

    def test_structured_accuracy_zero_shot_has_errors(self):
        """Zero-shot simulation should produce nonzero MAPE."""
        import sys

        ft_dir = Path(__file__).parents[1] / "examples" / "pipelines" / "vlm_chart_finetune"
        sys.path.insert(0, str(ft_dir))
        from run import _structured_extraction_accuracy

        from production_vlm.utils.synthetic_charts import generate_synthetic_chart

        charts = [generate_synthetic_chart(seed=i, chart_type="bar", render_image=False) for i in range(20)]
        result = _structured_extraction_accuracy(charts, noise_zero_shot=True)
        # Zero-shot should have non-perfect schema validity and nonzero MAPE
        assert result["schema_validity_rate"] < 1.0
        assert result["numeric_extraction_mape"] > 0.0

    def test_structured_accuracy_finetuned_is_perfect(self):
        """Fine-tuned simulation (exact GT) should score 100% schema valid, 0% MAPE."""
        import sys

        ft_dir = Path(__file__).parents[1] / "examples" / "pipelines" / "vlm_chart_finetune"
        sys.path.insert(0, str(ft_dir))
        from run import _structured_extraction_accuracy

        from production_vlm.utils.synthetic_charts import generate_synthetic_chart

        charts = [generate_synthetic_chart(seed=i, chart_type="bar", render_image=False) for i in range(20)]
        result = _structured_extraction_accuracy(charts, noise_zero_shot=False)
        assert result["schema_validity_rate"] == 1.0
        assert result["numeric_extraction_mape"] == 0.0
        assert result["category_coverage"] == 1.0
