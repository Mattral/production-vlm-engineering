"""Production observability for vision/VLM systems: metrics exposition and structured event logging.

Implements the observability pattern called out in P0-04 of the roadmap:
"Log metrics (drift score, sample count) and optional Prometheus exposition."

Two backends, both zero-hard-dependency on the monitoring stack:

1. ``PrometheusMetricsServer`` — exposes drift, OOD, and guard metrics in
   Prometheus text format via a background HTTP server. Requires
   ``prometheus_client`` (optional extra). Falls back to a no-op stub
   when not installed so every example can import this module safely.

2. ``ObservabilityLogger`` — structured JSONL event log with a clean
   schema, designed to be ingested by any log aggregator (Loki,
   Elasticsearch, CloudWatch, or just ``jq``). Zero dependencies.

Design rationale (from production MLOps systems):
   The two backends are complementary. Prometheus is for real-time
   alerting (PagerDuty/Grafana alert rules on drift_score > threshold).
   Structured logs are for forensics (correlate a drift event with the
   specific input batch that caused it, hours or days later). A system
   that only has one of these is incomplete for production use.

Usage:
    >>> from production_vlm.utils.observability import ObservabilityLogger, PrometheusMetricsServer
    >>> logger = ObservabilityLogger("outputs/my_run/events.jsonl")
    >>> logger.log_drift_event(batch_idx=6, is_drift=True, ks_stat=0.58, p_value=2e-7, batch_size=25)
    >>> # Optional Prometheus server (requires pip install prometheus_client):
    >>> server = PrometheusMetricsServer(port=9090)
    >>> server.start()
    >>> server.record_drift(ks_stat=0.58, is_drift=True)
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Structured event schema
# ---------------------------------------------------------------------------


@dataclass
class DriftEvent:
    event_type: str = "drift_check"
    timestamp_utc: float = field(default_factory=time.time)
    batch_idx: int = 0
    batch_size: int = 0
    is_drift_ks: bool = False
    is_drift_ewma: bool = False
    ks_stat: float = 0.0
    p_value: float | None = None
    batch_mean_similarity: float = 0.0
    ewma_mean: float = 0.0
    ewma_lower_cl: float = 0.0
    al_selected_count: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class OODEvent:
    event_type: str = "ood_check"
    timestamp_utc: float = field(default_factory=time.time)
    is_ood: bool = False
    ood_score: float = 0.0
    nearest_neighbor_similarity: float = 0.0
    threshold: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class GuardEvent:
    event_type: str = "guard_check"
    timestamp_utc: float = field(default_factory=time.time)
    decision: str = ""
    faithfulness_score: float = 0.0
    numeric_score: float = 0.0
    grounding_score: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Structured JSONL logger (zero dependencies)
# ---------------------------------------------------------------------------


class ObservabilityLogger:
    """Append-only structured event log in JSONL format.

    Schema is versioned via the ``schema_version`` field on every event,
    so log consumers can handle backwards-incompatible schema changes
    without breaking. Events are flushed immediately (no buffering) so
    a crash never loses the last event — important for forensics when
    investigating a production drift incident.
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self, log_path: str | Path, run_id: str = "") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run_{int(time.time())}"
        self._lock = threading.Lock()
        self._event_count = 0

    def _write(self, event: Any) -> None:
        record = {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "seq": self._event_count,
            **(asdict(event) if hasattr(event, "__dataclass_fields__") else event),
        }
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._event_count += 1

    def log_drift_event(
        self,
        batch_idx: int,
        batch_size: int,
        is_drift_ks: bool,
        is_drift_ewma: bool,
        ks_stat: float,
        p_value: float | None,
        batch_mean_similarity: float,
        ewma_mean: float = 0.0,
        ewma_lower_cl: float = 0.0,
        al_selected_count: int = 0,
        extra: dict | None = None,
    ) -> None:
        self._write(
            DriftEvent(
                batch_idx=batch_idx,
                batch_size=batch_size,
                is_drift_ks=is_drift_ks,
                is_drift_ewma=is_drift_ewma,
                ks_stat=ks_stat,
                p_value=p_value,
                batch_mean_similarity=batch_mean_similarity,
                ewma_mean=ewma_mean,
                ewma_lower_cl=ewma_lower_cl,
                al_selected_count=al_selected_count,
                extra=extra or {},
            )
        )

    def log_ood_event(
        self,
        is_ood: bool,
        ood_score: float,
        nearest_neighbor_similarity: float,
        threshold: float,
        extra: dict | None = None,
    ) -> None:
        self._write(
            OODEvent(
                is_ood=is_ood,
                ood_score=ood_score,
                nearest_neighbor_similarity=nearest_neighbor_similarity,
                threshold=threshold,
                extra=extra or {},
            )
        )

    def log_guard_event(
        self,
        decision: str,
        faithfulness_score: float,
        numeric_score: float,
        grounding_score: float,
        extra: dict | None = None,
    ) -> None:
        self._write(
            GuardEvent(
                decision=decision,
                faithfulness_score=faithfulness_score,
                numeric_score=numeric_score,
                grounding_score=grounding_score,
                extra=extra or {},
            )
        )

    def read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self) -> dict:
        events = self.read_all()
        drift_events = [e for e in events if e.get("event_type") == "drift_check"]
        ood_events = [e for e in events if e.get("event_type") == "ood_check"]
        guard_events = [e for e in events if e.get("event_type") == "guard_check"]
        return {
            "run_id": self.run_id,
            "total_events": len(events),
            "drift": {
                "total_batches": len(drift_events),
                "drift_flagged_ks": sum(1 for e in drift_events if e.get("is_drift_ks")),
                "drift_flagged_ewma": sum(1 for e in drift_events if e.get("is_drift_ewma")),
                "al_total_queued": sum(e.get("al_selected_count", 0) for e in drift_events),
            },
            "ood": {
                "total_checked": len(ood_events),
                "flagged": sum(1 for e in ood_events if e.get("is_ood")),
            },
            "guard": {
                "total_checked": len(guard_events),
                "passed": sum(1 for e in guard_events if e.get("decision") == "pass"),
                "flagged": sum(1 for e in guard_events if e.get("decision") == "flag"),
                "rejected": sum(1 for e in guard_events if e.get("decision") == "reject"),
            },
        }


# ---------------------------------------------------------------------------
# Prometheus exposition (optional -- graceful no-op if not installed)
# ---------------------------------------------------------------------------


class _NoOpCounter:
    def inc(self, amount: float = 1.0) -> None:
        pass

    def labels(self, **kw):
        return self


class _NoOpGauge:
    def set(self, value: float) -> None:
        pass

    def labels(self, **kw):
        return self


class _NoOpHistogram:
    def observe(self, value: float) -> None:
        pass

    def labels(self, **kw):
        return self


def _try_import_prometheus():
    try:
        from prometheus_client import Counter, Gauge, Histogram, start_http_server

        return Counter, Gauge, Histogram, start_http_server, True
    except ImportError:
        return _NoOpCounter, _NoOpGauge, _NoOpHistogram, None, False


class PrometheusMetricsServer:
    """Prometheus text-format metrics server for drift, OOD, and guard events.

    When ``prometheus_client`` is not installed, all methods are no-ops
    and ``start()`` prints a warning rather than raising — so code that
    calls ``server.record_drift(...)`` always works regardless of
    environment. Install the real client with:

        pip install prometheus_client

    Metric naming follows the Prometheus naming convention:
    ``<namespace>_<subsystem>_<name>_<unit>``.

    Scrape endpoint: ``http://host:<port>/metrics`` (standard Prometheus
    text format, compatible with Grafana, VictoriaMetrics, etc.)
    """

    NAMESPACE = "production_vlm"

    def __init__(self, port: int = 9090) -> None:
        self.port = port
        Counter, Gauge, Histogram, self._start_fn, self._available = _try_import_prometheus()

        # Drift metrics
        self.drift_batches_total = (
            Counter(
                f"{self.NAMESPACE}_drift_batches_total",
                "Total batches processed by the drift detector",
            )
            if self._available
            else _NoOpCounter()
        )

        self.drift_detected_total = (
            Counter(
                f"{self.NAMESPACE}_drift_detected_total",
                "Total batches where drift was flagged",
                ["detector"],
            )
            if self._available
            else _NoOpCounter()
        )

        self.drift_ks_stat = (
            Gauge(
                f"{self.NAMESPACE}_drift_ks_stat",
                "Most recent KS statistic from CosineDriftDetector",
            )
            if self._available
            else _NoOpGauge()
        )

        self.drift_batch_mean_similarity = (
            Gauge(
                f"{self.NAMESPACE}_drift_batch_mean_similarity",
                "Most recent batch mean cosine similarity to reference centroid",
            )
            if self._available
            else _NoOpGauge()
        )

        self.drift_al_queued_total = (
            Counter(
                f"{self.NAMESPACE}_drift_al_queued_total",
                "Total samples queued for active-learning labeling",
            )
            if self._available
            else _NoOpCounter()
        )

        # OOD metrics
        self.ood_checked_total = (
            Counter(
                f"{self.NAMESPACE}_ood_checked_total",
                "Total samples checked by the OOD detector",
            )
            if self._available
            else _NoOpCounter()
        )

        self.ood_flagged_total = (
            Counter(
                f"{self.NAMESPACE}_ood_flagged_total",
                "Total samples flagged as out-of-distribution",
            )
            if self._available
            else _NoOpCounter()
        )

        self.ood_score = (
            Histogram(
                f"{self.NAMESPACE}_ood_score",
                "OOD score distribution (1 - nearest_neighbor_similarity)",
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
            if self._available
            else _NoOpHistogram()
        )

        # Guard metrics
        self.guard_checked_total = (
            Counter(
                f"{self.NAMESPACE}_guard_checked_total",
                "Total answers checked by the hallucination guard",
            )
            if self._available
            else _NoOpCounter()
        )

        self.guard_decisions_total = (
            Counter(
                f"{self.NAMESPACE}_guard_decisions_total",
                "Guard decisions by outcome",
                ["decision"],
            )
            if self._available
            else _NoOpCounter()
        )

        self.guard_faithfulness = (
            Histogram(
                f"{self.NAMESPACE}_guard_faithfulness",
                "Faithfulness score distribution",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
            if self._available
            else _NoOpHistogram()
        )

    def start(self) -> None:
        """Start the HTTP metrics server in a background daemon thread."""
        if not self._available:
            print(
                "[observability] prometheus_client not installed -- metrics server not started. "
                "Install with: pip install prometheus_client"
            )
            return
        self._start_fn(self.port)
        print(f"[observability] Prometheus metrics server started on port {self.port} → /metrics")

    def record_drift(
        self,
        ks_stat: float,
        is_drift_ks: bool,
        is_drift_ewma: bool = False,
        batch_mean_similarity: float = 0.0,
        al_queued: int = 0,
    ) -> None:
        self.drift_batches_total.inc()
        self.drift_ks_stat.set(ks_stat)
        self.drift_batch_mean_similarity.set(batch_mean_similarity)
        if is_drift_ks:
            self.drift_detected_total.labels(detector="ks").inc()
        if is_drift_ewma:
            self.drift_detected_total.labels(detector="ewma").inc()
        if al_queued > 0:
            self.drift_al_queued_total.inc(al_queued)

    def record_ood(self, is_ood: bool, ood_score: float) -> None:
        self.ood_checked_total.inc()
        self.ood_score.observe(ood_score)
        if is_ood:
            self.ood_flagged_total.inc()

    def record_guard(self, decision: str, faithfulness: float) -> None:
        self.guard_checked_total.inc()
        self.guard_decisions_total.labels(decision=decision).inc()
        self.guard_faithfulness.observe(faithfulness)
