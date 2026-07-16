# Drift Detection

## Why embedding-space drift matters

A deployed VLM has two implicit contracts: (1) inputs come from a similar distribution to training, (2) the model's representation of those inputs remains stable. When either breaks — a new camera sensor, a new document format, a new chart-rendering library upstream — the model keeps running with no error signal. This is the most common cause of silent accuracy degradation in production CV/VLM systems.

Embedding-space drift detection catches this by monitoring *where in representation space* inputs land, rather than monitoring model outputs (which requires labels) or infrastructure metrics (which don't capture distribution shift).

## Cosine similarity to the reference centroid

The simplest meaningful 1D signal for drift is the cosine similarity between a new embedding and the centroid of a known-good reference set. If inputs are shifting, this distribution shifts too.

`CosineDriftDetector` turns this into a statistical test: for each incoming batch, compute the distribution of per-sample cosine similarities to the reference centroid, then run a two-sample Kolmogorov-Smirnov test against the same distribution over the reference set.

The KS test is used rather than a t-test or mean-difference test because:

- It's distribution-free: real embedding distributions are not Gaussian
- It's sensitive to shape changes (variance, skew), not just mean shifts
- A p-value threshold (`alpha`) gives a directly calibrated false-positive rate

## EWMA SPC control charts

`EWMADriftDetector` applies classic statistical process control (Shewhart + EWMA, Montgomery 2020[^1]) to the scalar stream of batch-mean similarities:

$$\hat{\mu}_t = \lambda \cdot x_t + (1 - \lambda) \cdot \hat{\mu}_{t-1}$$

A sample is flagged when $\hat{\mu}_t$ falls outside $[\hat{\mu}_{\text{baseline}} \pm n_\sigma \cdot \sigma_{\text{baseline}}]$, where $\sigma_{\text{baseline}}$ is the standard deviation of the **calibration period only**, computed once and frozen.

### Why freeze the baseline standard deviation?

This was a real design failure in the first implementation here. If $\sigma$ is continuously re-estimated from all incoming data:

1. A large shift arrives at time $t$
2. The delta $|x_t - \hat{\mu}_{t-1}|$ is huge, inflating the variance estimate
3. The control limits widen to accommodate the new variance
4. The point at time $t$ no longer exceeds the limits
5. The detector cannot detect the very event that caused the widening

Freezing $\sigma_{\text{baseline}}$ from the pre-shift calibration period sidesteps this entirely. The EWMA mean still adapts (so the centerline tracks slow trends), but the control limits stay calibrated to in-distribution variance.

## Detection delay

`detection_delay_batches = drift_detected_at_batch - drift_injected_at_batch`

In the default config, both detectors achieve zero detection delay — they flag drift at the exact first batch where it's injected. The benchmark sweep (`production-vlm benchmark embedding_drift_active_learning`) characterizes how this delay grows as the shift magnitude decreases toward the detection boundary.

## Why this matters for 2027

Regulatory pressure (the EU AI Act's post-market monitoring obligations for high-risk systems, and equivalents emerging elsewhere) is turning "documented, continuous distribution-shift monitoring" from a best practice into a compliance requirement for deployed ML systems, not just an internal engineering nicety. At the same time, the shift from batch-scored pipelines to streaming/agentic deployments — where a VLM's output directly triggers a downstream action rather than sitting in a dashboard for a human to review — collapses the time window between "the input distribution changed" and "something costly happened as a result." A drift detector that fires after a human notices a KPI dip is not fast enough for a system making autonomous decisions every few seconds. The pattern this module demonstrates — a persistent statistical test for dashboards, an onset-detecting SPC signal for alerting, both closing the loop into an active-learning/retraining trigger rather than just logging a number — is the shape "monitoring" needs to take when the thing being monitored can act on its own conclusions.

[^1]: Montgomery, D.C. (2020). *Introduction to Statistical Quality Control*, 8th edition. Wiley.
