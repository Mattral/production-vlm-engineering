"""Model & data lineage tracking for production VLM pipelines.

Answers the production question drift/retraining monitoring alone can't:
given a deployed model artifact, what data, config, code version, and
upstream event produced it -- and what produced *that*? This is the
"data lineage" half of the P0-04 drift/retraining loop that was missing:
``CosineDriftDetector``/``EWMADriftDetector`` (see ``production_vlm.drift``)
already monitor feature drift continuously, and ``RetrainingTrigger`` (see
``production_vlm.utils.retraining``) already fires automated retraining on
threshold breach -- both real, both tested. What was missing is the
provenance graph connecting them: which artifact each retraining run
consumed and produced, and which drift event triggered it.

Design decision: this is NOT a wrapper around MLflow, DVC, or a Delta
Lake-backed feature store. Those are real, valid choices for a live
production deployment, but hard-adopting any of them here would make this
repo's core untestable without standing up external infrastructure --
directly against the project's established design (a CPU-only core with
zero hard dependencies on the monitoring/tracking stack; see
``utils/observability.py``'s ``PrometheusMetricsServer`` for the same
pattern applied to metrics instead of lineage). Instead:

- ``LineageStore`` is a narrow interface (2 methods): swap in a real
  backend for production without changing any caller.
- ``JSONLLineageStore`` is the default: append-only, zero dependencies,
  same schema-versioned JSONL pattern as ``ObservabilityLogger``.
- See ``docs/concepts/lineage.md`` for a documented MLflow/DVC adapter
  stub -- the same "extension path, not natively implemented" treatment
  P0-03 already gives TensorRT/OpenVINO for the identical reason
  (hardware/infra-specific, incompatible with a reproducible CPU-only repo).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = "1.0"


def content_hash(obj: Any) -> str:
    """Deterministic content hash for lineage artifact identity.

    Uses ``hashlib.sha256`` over a canonical (sorted-key) JSON encoding --
    deliberately NOT Python's built-in ``hash()``, which is randomized
    per-process by default. That exact mistake was already made once in
    this codebase (``SyntheticEmbeddingProxy``, documented in
    ``ROADMAP.md``/``CHANGELOG.md``): it silently made every
    embedding-derived metric non-reproducible across runs until it was
    replaced with ``zlib.crc32``. Lineage has the same requirement --
    two runs over identical config/data must resolve to the identical
    artifact hash, or lineage chains silently fork -- so it gets the
    same fix class applied up front rather than discovered by a bug.
    """
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def get_code_version() -> str:
    """Best-effort short git SHA of the current checkout; ``"unknown"`` if unavailable.

    Never raises. This repo is frequently used as an extracted archive
    or a shallow clone rather than a full git working tree -- exactly the
    environment this was developed and verified in -- so failure to
    resolve a SHA must degrade gracefully rather than break every caller
    that wants to record a lineage entry.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


@dataclass
class ArtifactRef:
    """A reference to one artifact in the lineage graph.

    ``artifact_type`` is a free-form label ("dataset", "config", "model",
    "evaluation") rather than an enum -- pipelines define their own
    artifact taxonomy; the lineage graph doesn't need to know it.
    """

    artifact_id: str
    artifact_type: str
    content_hash: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class LineageRecord:
    """One provenance edge: artifacts consumed -> one artifact produced.

    Forms a DAG when chained: ``produced.artifact_id`` of one record can
    appear in ``consumed`` of a later record (e.g. model_v1 is produced by
    one retraining run, then consumed as the base checkpoint for the next).
    """

    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    seq: int = 0
    timestamp_utc: float = field(default_factory=time.time)
    produced: dict = field(default_factory=dict)
    consumed: list = field(default_factory=list)
    code_version: str = "unknown"
    triggered_by: str | None = None
    extra: dict = field(default_factory=dict)


class LineageStore(Protocol):
    """Interface every lineage backend implements.

    Deliberately 2 methods. A production deployment implements this
    against MLflow's Model Registry, a DVC-tracked metadata store, or a
    Delta Lake table -- see ``docs/concepts/lineage.md`` for a worked
    adapter stub -- without any caller in this codebase changing.
    """

    def record(
        self,
        produced: ArtifactRef,
        consumed: list[ArtifactRef] | None = None,
        triggered_by: str | None = None,
        code_version: str | None = None,
        extra: dict | None = None,
    ) -> LineageRecord: ...

    def all_records(self) -> list[dict]: ...


class JSONLLineageStore:
    """Default zero-dependency lineage backend: append-only JSONL.

    Same design as ``ObservabilityLogger``: flushed immediately (no
    buffering), so a crash mid-run never loses the last recorded edge --
    the exact moment a forensic lineage query is most likely to be needed.
    Good for local dev, CI, and ``jq``-driven forensics; not a substitute
    for a real metadata service under concurrent multi-writer production
    load (again, see ``docs/concepts/lineage.md``).
    """

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, log_path: str | Path, run_id: str = "") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"run_{int(time.time())}"
        self._lock = threading.Lock()
        self._seq = 0
        # Resolved once per process, not per record: the code version is an
        # invariant of this run, and shelling out to git on every call would
        # be wasteful (and, worse, blocking, when called under a caller's
        # own lock -- as RetrainingTrigger._fire() does).
        self._default_code_version = get_code_version()

    def record(
        self,
        produced: ArtifactRef,
        consumed: list[ArtifactRef] | None = None,
        triggered_by: str | None = None,
        code_version: str | None = None,
        extra: dict | None = None,
    ) -> LineageRecord:
        entry = LineageRecord(
            run_id=self.run_id,
            seq=self._seq,
            produced=asdict(produced),
            consumed=[asdict(c) for c in (consumed or [])],
            code_version=code_version if code_version is not None else self._default_code_version,
            triggered_by=triggered_by,
            extra=extra or {},
        )
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(entry)) + "\n")
            self._seq += 1
        return entry

    def all_records(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_chain(self, artifact_id: str) -> list[dict]:
        """Walk backward from ``artifact_id`` through ``consumed`` parents.

        Returns the full provenance chain in causal (oldest-first) order --
        this is the actual question lineage tracking exists to answer:
        "this model is misbehaving in production; what data, config, and
        upstream retraining events produced it?" Recursive, and guards
        against re-visiting the same record twice if the graph has a
        diamond dependency (e.g. two runs sharing one base config).
        """
        by_produced_id: dict[str, list[dict]] = {}
        for rec in self.all_records():
            pid = rec["produced"].get("artifact_id")
            if pid is not None:
                by_produced_id.setdefault(pid, []).append(rec)

        chain: list[dict] = []
        seen: set[tuple[str, int]] = set()

        def _walk(aid: str) -> None:
            for rec in by_produced_id.get(aid, []):
                key = (rec["run_id"], rec["seq"])
                if key in seen:
                    continue
                seen.add(key)
                for consumed_ref in rec["consumed"]:
                    _walk(consumed_ref["artifact_id"])
                chain.append(rec)

        _walk(artifact_id)
        return chain
