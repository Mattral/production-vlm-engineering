# Citations & References

Every technique implemented in this repo is cited inline, in the docstring of the function or class that implements it — this page consolidates them into a single bibliography for convenience. For the *why* behind each technique, see the linked concept page.

## Parameter-efficient fine-tuning

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). **LoRA: Low-Rank Adaptation of Large Language Models**. *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
  → Implemented in: `vlm_chart_finetune/run.py` (`LoraConfig`, vision tower + LM adaptation)
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). **QLoRA: Efficient Finetuning of Quantized LLMs**. *NeurIPS 2023*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
  → Implemented in: `vlm_chart_finetune/run.py` (`BitsAndBytesConfig` 4-bit + LoRA combination)

## Evaluation metrics

- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
  → Implemented in: `production_vlm/eval/__init__.py` (`faithfulness_score`, adapted from retrieved-text to chart/image evidence)
- Masry, A., Long, D. X., Tan, J. Q., Joty, S., & Hoque, E. (2022). **ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning**. *ACL Findings 2022*. [arXiv:2203.10244](https://arxiv.org/abs/2203.10244)
  → Implemented in: `production_vlm/eval/__init__.py` (`numeric_accuracy` tolerance convention, 2% relative)

## Drift detection & statistical process control

- Massey, F. J. (1951). **The Kolmogorov-Smirnov Test for Goodness of Fit**. *Journal of the American Statistical Association*, 46(253), 68–78.
  → Implemented in: `production_vlm/drift/__init__.py` (`CosineDriftDetector`, two-sample KS test)
- Montgomery, D. C. (2020). **Introduction to Statistical Quality Control** (8th ed.). Wiley.
  → Implemented in: `production_vlm/drift/__init__.py` (`EWMADriftDetector`, frozen-baseline Shewhart/EWMA control chart)
- Settles, B. (2009). **Active Learning Literature Survey**. *University of Wisconsin-Madison Computer Sciences Technical Report 1648*.
  → Implemented in: `production_vlm/drift/__init__.py` (`select_for_active_learning`, distance-from-centroid novelty proxy)

## Robustness & adversarial testing

- Hendrycks, D., & Dietterich, T. (2019). **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**. *ICLR 2019*. [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)
  → Implemented in: `production_vlm/robustness/perturbations.py` (`NaturalPerturbation`, six corruption types with severity scaling)
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). **Towards Deep Learning Models Resistant to Adversarial Attacks**. *ICLR 2018*. [arXiv:1706.06083](https://arxiv.org/abs/1706.06083)
  → Implemented in: `production_vlm/robustness/perturbations.py` (`pgd_attack`, real gradient-based PGD) and `vlm_robustness_guard/run.py` (`run_adversarial_robustness`, numpy CPU-only proxy)

## Quantization & inference optimization

- Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2018). **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**. *CVPR 2018*. [arXiv:1712.05877](https://arxiv.org/abs/1712.05877)
  → Implemented in: `vlm_edge_inference/run.py` (`quantize_real_model`, ONNX Runtime dynamic INT8)

## Dynamic batching & serving

- NVIDIA Triton Inference Server documentation: **Dynamic Batching**. [docs.nvidia.com/deeplearning/triton-inference-server](https://docs.nvidia.com/deeplearning/triton-inference-server/)
  → Implemented in: `production_vlm/utils/batching_queue.py` (`BatchingQueue`, size-or-timeout flush pattern)

## Video / temporal reasoning (forward-looking, P1-04)

- Lin, B., Zhu, B., Ye, Y., Ning, M., Jin, P., & Yuan, L. (2023). **Video-LLaVA: Learning United Visual Representation by Alignment Before Projection**. [arXiv:2311.10122](https://arxiv.org/abs/2311.10122)
  → Referenced in: `vlm_video_temporal/run.py` (multi-frame prompt interleaving pattern)
- Fu, C., et al. (2024). **VITA: Towards Open-Source Interactive Omni Multimodal LLM**. [arXiv:2408.05211](https://arxiv.org/abs/2408.05211)
  → Referenced in: `vlm_video_temporal/run.py` (video + structured output pattern)

## Production observability

- Prometheus documentation: **Metric and Label Naming**. [prometheus.io/docs/practices/naming](https://prometheus.io/docs/practices/naming/)
  → Implemented in: `production_vlm/utils/observability.py` (`PrometheusMetricsServer`, namespace/subsystem/name/unit convention)

---

## How to cite this repository

If this repository's design patterns or implementations are useful in your own work:

```bibtex
@misc{production-vlm-engineering,
  title  = {Production VLM Engineering: Reproducible Pipelines for Multimodal Vision Systems},
  author = {Mattral},
  year   = {2026},
  url    = {https://github.com/Mattral/production-vlm-engineering}
}
```
