# vlm_chart_finetune

LoRA fine-tuning of a vision-language model on chart/document visual question answering.

## What this demonstrates

- Parameter-efficient fine-tuning (LoRA) applied to **both** the vision tower and language
  model projections of a VLM, not language-only LoRA (the 2025-2026 convention for
  multimodal adapters).
- A zero-download synthetic chart-QA dataset, so the pipeline has no external data
  dependency and the same generator can inject controlled out-of-distribution style shifts
  for the companion drift-detection example.
- Evaluation via numeric accuracy, grounding, and a RAGAS-inspired faithfulness score
  (`production_vlm.eval`) -- the right metrics for numeric chart answers, where exact-match and
  BLEU give misleading signal.

## Run it

```bash
production-vlm run-example vlm_chart_finetune
# or directly:
python -m examples.pipelines.vlm_chart_finetune.run
```

Override the config:

```bash
production-vlm run-example vlm_chart_finetune --config path/to/custom.yaml
```

## What you'll see

Without a CUDA device and the `ml` extra (`pip install -e ".[ml]"`) installed, the pipeline
runs a CPU-only smoke test: real synthetic data generation, real config validation, and a
real pass through the evaluation harness, with simulated model outputs standing in for
actual generations -- printed and saved to `results.json` as `ran_with_real_ml_stack: false`
so this is never ambiguous.

With a GPU and the `ml` extra installed, `train_real()` in `run.py` runs genuine LoRA
fine-tuning: loads `model.checkpoint` (default `Qwen/Qwen2-VL-2B-Instruct`, pinned per
`model.checkpoint_pinned_date` in the config -- re-verify the checkpoint still resolves
before trusting numbers, since hub checkpoints can be renamed/removed), applies 4-bit
quantization + LoRA (rank/alpha/dropout/target_modules all configurable), and trains with
gradient accumulation under a hard `max_train_minutes` wall-clock cap.

## Files

- `run.py` -- the pipeline: data generation, training (real or simulated), evaluation, report.
- `../../../configs/vlm_chart_finetune.yaml` -- experiment config (model, LoRA, data, train, eval).

## Swapping in a real dataset

Replace calls to `production_vlm.utils.synthetic_charts.generate_dataset` with a loader for
ChartQA, DocVQA, or any chart/document VQA dataset that yields `(image, question, answer,
evidence_text)` tuples -- the rest of the pipeline (LoRA setup, training loop, evaluation
metrics) is dataset-agnostic.
