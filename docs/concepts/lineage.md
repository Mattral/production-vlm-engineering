# Model & Data Lineage

## The question drift monitoring alone can't answer

`CosineDriftDetector`/`EWMADriftDetector` (see [Drift Detection](drift.md)) answer "has the input distribution shifted?" `RetrainingTrigger` answers "should we retrain right now?" Neither answers the question an on-call engineer actually asks three weeks later, when the model deployed today starts misbehaving: **what data, config, and code produced this exact artifact — and what produced the artifact before it?**

That's lineage: a queryable provenance graph connecting each model version to the batch of samples that triggered its retraining, the model version it was retrained from, and the code revision that ran the job. Without it, "which drift event caused this model to be retrained" is a question you answer by grepping timestamps across three separate log files, if you can answer it at all.

## Why not just add MLflow, DVC, or a Delta Lake table?

All three are legitimate, widely-used answers to this problem in a real production deployment. They are the wrong default for *this repository*, for the same reason TensorRT and OpenVINO are documented rather than wired in by default for `vlm_edge_inference`: they require standing up infrastructure (a tracking server, cloud storage remotes, a Spark-compatible lakehouse) that a `git clone && make setup` reproducibility promise can't depend on. This project's own tooling already draws that line consistently — `PrometheusMetricsServer` is optional with a graceful no-op fallback, `config.py` has zero hard dependency on pydantic — and hard-adopting a tracking platform here would be the one place that promise breaks.

The actual production-engineering answer isn't "pick a tool," it's "define the interface, so the tool is swappable." That's a narrower, more durable decision than it looks: whichever platform a given team standardizes on, every caller in this codebase — `RetrainingTrigger`, the drift loop, a future training script — talks to `LineageStore`, not to MLflow's or DVC's SDK directly.

## Architecture

```
production_vlm.lineage
├── ArtifactRef       # one node: id, type ("dataset"|"config"|"model"|"evaluation"), content hash
├── LineageRecord      # one edge: artifacts consumed -> one artifact produced, schema-versioned
├── LineageStore        # the interface (2 methods: record, all_records)
└── JSONLLineageStore    # default backend: append-only JSONL, same pattern as ObservabilityLogger
```

`get_chain(artifact_id)` walks backward through `consumed` references recursively, returning the full provenance chain in causal order — the multi-hop query an incident actually needs, not just the single most recent edge.

`RetrainingTrigger` takes an optional `lineage_store`. Every successful retraining run — and only a successful one; a failed run must not claim to have produced an artifact — emits one `LineageRecord` chaining the new model id back to the model it was retrained from and the batch of flagged samples that triggered it. `embedding_drift_active_learning` wires this in end-to-end: after a 12-batch run with drift injected at batch 6, the example prints something like

```
Lineage: outputs/.../lineage.jsonl (model_base → model_v1 → model_v2, 2 retraining edge(s))
```

and `get_chain("model_v2")` returns both edges, oldest first, each annotated with `triggered_by` (which detector fired) and the exact sample batch consumed.

## Content hashing, and a bug this project already made once

Artifact identity uses `hashlib.sha256` over a canonically-sorted JSON encoding — deliberately not Python's built-in `hash()`. That specific mistake already happened once in this codebase: `SyntheticEmbeddingProxy` used `hash()`, which is randomized per process by default, and silently made every embedding-derived metric non-reproducible across runs until it was replaced with `zlib.crc32` (documented in `CHANGELOG.md`/`ROADMAP.md`). Lineage has the identical requirement — two runs over identical inputs must resolve to the identical artifact hash, or the provenance graph silently forks — so it gets a deterministic hash from the start rather than as a bug fix.

## Extension path: a real backend

A production deployment implements `LineageStore` against whatever platform it already runs. An MLflow-backed adapter, for a team that already has a tracking server:

```python
import mlflow

class MLflowLineageStore:
    """LineageStore backed by MLflow's tracking API. Requires `pip install
    mlflow` and a reachable MLFLOW_TRACKING_URI -- neither of which this
    repo depends on by default. Swap in for JSONLLineageStore; every
    caller (RetrainingTrigger, the drift loop) is unaffected.
    """

    def record(self, produced, consumed=None, triggered_by=None,
               code_version=None, extra=None):
        with mlflow.start_run(run_name=produced.artifact_id) as run:
            mlflow.set_tags({
                "artifact_type": produced.artifact_type,
                "triggered_by": triggered_by or "",
                "consumed": ",".join(c.artifact_id for c in (consumed or [])),
            })
            mlflow.log_dict(
                {"produced": produced.artifact_id, "content_hash": produced.content_hash},
                "lineage.json",
            )
        return run.info.run_id

    def all_records(self):
        # mlflow.search_runs(...) filtered to this experiment, mapped
        # back into the same record shape JSONLLineageStore returns.
        ...
```

A DVC-backed store follows the same shape, writing a `dvc.yaml`-tracked metadata file per artifact instead of an MLflow run. Neither is implemented here for the same reason TensorRT/OpenVINO aren't: both need infrastructure a CPU-verifiable repo can't assume, and the interface — not the specific platform — is the part worth shipping by default.

## Why this matters for 2027

The EU AI Act's post-market monitoring obligations (see [Drift Detection](drift.md)) don't stop at "detect that something changed" — Article 72's continuous monitoring duty is paired with documentation requirements that, in practice, mean an auditor can ask "show me exactly what produced the model currently in production" and expect a real answer, not a best guess reconstructed from Slack history. Separately, as retraining loops close the gap between "drift detected" and "new model deployed" — the entire point of `RetrainingTrigger` — the number of model versions a system produces over its lifetime stops being small enough to track by hand. A team that ships one model a quarter can get away with a spreadsheet. A team whose drift detector can trigger a retrain every few hours cannot, and by 2027 that's the more common shape of a deployed VLM system, not the exception. Lineage tracking is what keeps "the model retrains itself" from becoming "nobody can explain why the model changed."
