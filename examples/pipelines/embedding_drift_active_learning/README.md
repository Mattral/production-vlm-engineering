# embedding_drift_active_learning

Production embedding-space drift detection with an active-learning triage loop.

## What this demonstrates

Enterprise CV/VLM deployment reports from early 2026 repeatedly cite the absence of
embedding-space drift detection as the leading cause of silent production failure: latency
and error rates look fine while the input distribution has quietly shifted (new camera, new
lighting, a new upstream rendering pipeline) and accuracy degrades with no signal until a
downstream complaint. This example builds the missing piece:

1. **`CosineDriftDetector`** (`production_vlm.drift`) -- a two-sample Kolmogorov-Smirnov test on
   the distribution of cosine similarities to a reference centroid. Distribution-free, robust
   to the non-Gaussian shape real embedding spaces actually have.
2. **`EWMADriftDetector`** -- an online, alertable SPC signal with a **frozen baseline
   standard deviation**. A naive implementation that continuously re-estimates variance from
   the incoming stream is self-defeating under a real step-change (the jump itself inflates
   the variance estimate and widens the control limits just when they need to stay tight) --
   this was a real bug caught during development; see the class docstring in
   `src/production_vlm/drift/__init__.py` for the full explanation.
3. **Active learning triage** -- when drift is flagged, rank the batch by distance from the
   reference centroid (free, label-free) and queue the most novel samples for human labeling.

## Run it

```bash
production-vlm run-example embedding_drift_active_learning
production-vlm benchmark embedding_drift_active_learning   # sensitivity sweep over drift magnitude
```

## What you'll see

A per-batch table showing the KS statistic, p-value, and both detectors' flags as a
synthetic stream transitions from in-distribution to a style-shifted (out-of-distribution)
regime starting at `stream.drift_starts_at_batch`. Both detectors are expected to fire with
**zero detection delay** at the default config's shift magnitude (12.0) -- this is a real,
measured result, not an asserted one.

The benchmark sweep is deliberately honest about sensitivity limits: it sweeps the injected
shift magnitude from subtle to obvious and reports where detection starts succeeding. At low
magnitudes the detector reliably *fails* to trigger -- this is the expected, correct
specificity/sensitivity tradeoff of any real drift monitor, and the example does not hide it.

## Swapping in a real vision encoder

By default this uses `SyntheticEmbeddingProxy` (`production_vlm.utils.vision_encoder`), which
derives embeddings from chart metadata rather than running a real model -- so the example
needs only numpy/scipy/matplotlib/pillow, no GPU, no network. To use a genuine embedding
space:

```python
from production_vlm.utils.vision_encoder import RealVisionEncoder

encoder = RealVisionEncoder("facebook/dinov2-base")  # or a DINOv3/SigLIP-2 checkpoint
embeddings = encoder.encode(list_of_pil_images)
```

`RealVisionEncoder` requires `pip install -e ".[ml]"` and network access to pull the
checkpoint. The drift-detection and active-learning code downstream is unchanged either way.

## Files

- `run.py` -- streaming loop, both detectors, active-learning selection, benchmark sweep.
- `../../../src/production_vlm/drift/__init__.py` -- the detector implementations.
- `../../../src/production_vlm/utils/vision_encoder.py` -- synthetic proxy + real encoder wrapper.
- `../../../configs/embedding_drift_active_learning.yaml` -- stream/detector/active-learning config.
