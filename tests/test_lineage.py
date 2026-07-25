"""Unit tests for production_vlm.lineage: model/data provenance tracking."""

from __future__ import annotations

import json

from production_vlm.lineage import (
    ArtifactRef,
    JSONLLineageStore,
    content_hash,
    get_code_version,
)
from production_vlm.utils.retraining import QueuedSample, RetrainingTrigger


class TestContentHash:
    def test_deterministic_across_calls(self):
        obj = {"b": 2, "a": 1, "nested": [1, 2, 3]}
        assert content_hash(obj) == content_hash(obj)

    def test_key_order_does_not_affect_hash(self):
        assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})

    def test_different_content_gives_different_hash(self):
        assert content_hash({"a": 1}) != content_hash({"a": 2})

    def test_returns_short_hex_string(self):
        h = content_hash({"x": 1})
        assert len(h) == 16
        int(h, 16)  # raises ValueError if not valid hex


class TestGetCodeVersion:
    def test_never_raises_and_returns_a_string(self):
        # Must degrade gracefully outside a git working tree (this repo is
        # frequently used as an extracted archive, not a full git clone).
        result = get_code_version()
        assert isinstance(result, str)
        assert result != ""


class TestJSONLLineageStore:
    def test_record_and_all_records_round_trip(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl", run_id="run1")
        produced = ArtifactRef(artifact_id="model_v1", artifact_type="model", content_hash="abc")
        store.record(produced=produced)

        records = store.all_records()
        assert len(records) == 1
        assert records[0]["produced"]["artifact_id"] == "model_v1"
        assert records[0]["run_id"] == "run1"
        assert records[0]["schema_version"] == JSONLLineageStore.SCHEMA_VERSION

    def test_seq_increments_across_records(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        for i in range(3):
            store.record(produced=ArtifactRef(artifact_id=f"a{i}", artifact_type="model"))
        records = store.all_records()
        assert [r["seq"] for r in records] == [0, 1, 2]

    def test_all_records_empty_when_file_does_not_exist_yet(self, tmp_path):
        # __init__ creates the parent dir but not the file itself (nothing
        # written yet) -- all_records() must handle that, not raise.
        store = JSONLLineageStore(tmp_path / "subdir" / "lineage.jsonl")
        assert store.all_records() == []

    def test_file_is_valid_jsonl(self, tmp_path):
        path = tmp_path / "lineage.jsonl"
        store = JSONLLineageStore(path)
        store.record(produced=ArtifactRef(artifact_id="a", artifact_type="model"))
        store.record(produced=ArtifactRef(artifact_id="b", artifact_type="model"))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            json.loads(line)  # raises if not valid JSON

    def test_consumed_artifacts_are_persisted(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        consumed = [
            ArtifactRef(artifact_id="dataset_v1", artifact_type="dataset"),
            ArtifactRef(artifact_id="config_v1", artifact_type="config"),
        ]
        store.record(produced=ArtifactRef(artifact_id="model_v1", artifact_type="model"), consumed=consumed)
        record = store.all_records()[0]
        assert {c["artifact_id"] for c in record["consumed"]} == {"dataset_v1", "config_v1"}

    def test_triggered_by_is_persisted(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        store.record(produced=ArtifactRef(artifact_id="model_v1", artifact_type="model"), triggered_by="drift_ks")
        assert store.all_records()[0]["triggered_by"] == "drift_ks"


class TestGetChain:
    def test_single_hop_chain(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        store.record(
            produced=ArtifactRef(artifact_id="model_v1", artifact_type="model"),
            consumed=[ArtifactRef(artifact_id="dataset_v1", artifact_type="dataset")],
        )
        chain = store.get_chain("model_v1")
        assert len(chain) == 1
        assert chain[0]["produced"]["artifact_id"] == "model_v1"

    def test_multi_hop_chain_is_oldest_first(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        # model_base -> model_v1 -> model_v2, each consuming the previous.
        store.record(
            produced=ArtifactRef(artifact_id="model_v1", artifact_type="model"),
            consumed=[ArtifactRef(artifact_id="model_base", artifact_type="model")],
        )
        store.record(
            produced=ArtifactRef(artifact_id="model_v2", artifact_type="model"),
            consumed=[ArtifactRef(artifact_id="model_v1", artifact_type="model")],
        )
        chain = store.get_chain("model_v2")
        produced_ids = [r["produced"]["artifact_id"] for r in chain]
        assert produced_ids == ["model_v1", "model_v2"]  # causal order: oldest edge first

    def test_unknown_artifact_returns_empty_chain(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        store.record(produced=ArtifactRef(artifact_id="model_v1", artifact_type="model"))
        assert store.get_chain("model_does_not_exist") == []

    def test_diamond_dependency_does_not_duplicate_records(self, tmp_path):
        # Two runs both consume the same shared base config -- walking back
        # from each must not re-visit (and re-append) the shared ancestor's
        # producing record twice.
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        store.record(produced=ArtifactRef(artifact_id="config_v1", artifact_type="config"))
        store.record(
            produced=ArtifactRef(artifact_id="model_a", artifact_type="model"),
            consumed=[ArtifactRef(artifact_id="config_v1", artifact_type="config")],
        )
        store.record(
            produced=ArtifactRef(artifact_id="model_b", artifact_type="model"),
            consumed=[ArtifactRef(artifact_id="config_v1", artifact_type="config")],
        )
        store.record(
            produced=ArtifactRef(artifact_id="ensemble", artifact_type="model"),
            consumed=[
                ArtifactRef(artifact_id="model_a", artifact_type="model"),
                ArtifactRef(artifact_id="model_b", artifact_type="model"),
            ],
        )
        chain = store.get_chain("ensemble")
        produced_ids = [r["produced"]["artifact_id"] for r in chain]
        assert produced_ids.count("config_v1") == 1
        assert len(chain) == 4  # config_v1, model_a, model_b, ensemble -- each exactly once


class TestRetrainingTriggerLineageIntegration:
    """Verifies RetrainingTrigger emits real lineage edges, not just logs a retraining happened."""

    def test_no_lineage_recorded_when_store_not_provided(self, tmp_path):
        trigger = RetrainingTrigger(queue_threshold=2, callback=lambda s: None, cooldown_s=0)
        trigger.enqueue_batch(
            [QueuedSample(embedding_index=i, batch_idx=0, novelty_score=0.5, flagged_by="ood") for i in range(2)]
        )
        assert trigger.trigger_count == 1  # unaffected by lineage being absent

    def test_successful_run_records_one_lineage_edge(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        trigger = RetrainingTrigger(
            queue_threshold=2, callback=lambda s: None, cooldown_s=0, lineage_store=store, base_model_id="model_base"
        )
        trigger.enqueue_batch(
            [QueuedSample(embedding_index=i, batch_idx=0, novelty_score=0.5, flagged_by="drift_ks") for i in range(2)]
        )
        records = store.all_records()
        assert len(records) == 1
        assert records[0]["produced"]["artifact_id"] == "model_v1"
        consumed_ids = {c["artifact_id"] for c in records[0]["consumed"]}
        assert "model_base" in consumed_ids
        assert records[0]["triggered_by"] == "drift_ks"

    def test_failed_run_records_no_lineage_edge(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")

        def bad_callback(samples):
            raise RuntimeError("training job failed")

        trigger = RetrainingTrigger(queue_threshold=2, callback=bad_callback, cooldown_s=0, lineage_store=store)
        trigger.enqueue_batch(
            [QueuedSample(embedding_index=i, batch_idx=0, novelty_score=0.5, flagged_by="ood") for i in range(2)]
        )
        # A failed retraining run must not claim a model artifact was produced.
        assert store.all_records() == []

    def test_current_model_id_advances_after_each_successful_run(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        trigger = RetrainingTrigger(queue_threshold=2, callback=lambda s: None, cooldown_s=0, lineage_store=store)
        assert trigger.current_model_id == "model_base"
        trigger.enqueue_batch(
            [QueuedSample(embedding_index=i, batch_idx=0, novelty_score=0.5, flagged_by="ood") for i in range(2)]
        )
        assert trigger.current_model_id == "model_v1"

    def test_multiple_cycles_form_a_walkable_chain(self, tmp_path):
        store = JSONLLineageStore(tmp_path / "lineage.jsonl")
        trigger = RetrainingTrigger(queue_threshold=2, callback=lambda s: None, cooldown_s=0, lineage_store=store)
        for cycle in range(3):
            trigger.enqueue_batch(
                [
                    QueuedSample(embedding_index=i, batch_idx=cycle, novelty_score=0.5, flagged_by="drift_ewma")
                    for i in range(2)
                ]
            )
        assert trigger.current_model_id == "model_v3"
        chain = store.get_chain("model_v3")
        produced_ids = [r["produced"]["artifact_id"] for r in chain]
        assert produced_ids == ["model_v1", "model_v2", "model_v3"]
