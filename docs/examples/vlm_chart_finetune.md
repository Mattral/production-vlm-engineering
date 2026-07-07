# VLM Chart Fine-Tuning

**Pipeline:** `examples/pipelines/vlm_chart_finetune/`  
**Config:** `configs/vlm_chart_finetune.yaml`  
**P-level:** P0-02

## What it demonstrates

Efficient parameter-efficient fine-tuning (LoRA) of a vision-language model on chart/document visual question answering, with three evaluation metrics better suited to numeric chart answers than BLEU or exact-match.

## Key design decisions

### Adapting both modalities, not just language

The 2025-2026 convention for multimodal LoRA is to adapt both the vision tower's projection layers and the language model's attention projections, not language-only as was common in 2023-2024. The config exposes this explicitly:

```yaml
lora:
  target_vision_tower: true
  target_language_model: true
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### Zero-download synthetic data

The pipeline uses `production_vlm.utils.synthetic_charts` by default — no external dataset required. This makes the smoke-test path fast (<5s) and reproducible, and the same generator is used by the robustness sweep and drift detection examples, keeping the data contract consistent.

To use a real dataset (ChartQA, DocVQA, etc.), replace the `generate_dataset()` call in `run.py` with any loader that yields `(image, question, answer, evidence_text)` tuples. The evaluation harness and training loop are dataset-agnostic.

### Three metrics instead of BLEU

| Metric | What it measures | Why |
|---|---|---|
| `numeric_accuracy` | Fraction of reference numbers matched within relative tolerance | Chart answers are predominantly numeric; exact-match is too strict (formatting), BLEU is not meaningful for numbers |
| `grounding_score` | Fraction of content words in the prediction that appear in the source evidence | Catches answers that are fluent but not grounded |
| `faithfulness_score` | Weighted combination of the two above | Single comparable scalar for ranking runs |

## Run it

=== "CPU smoke test"
    ```bash
    production-vlm run-example vlm_chart_finetune
    # Generates 48 synthetic charts, runs real evaluation harness
    # with simulated model outputs, writes results.json in ~5s
    ```

=== "Real GPU fine-tuning"
    ```bash
    pip install -e ".[ml]"
    # Requires CUDA + ≥12GB VRAM for the default 2B checkpoint
    production-vlm run-example vlm_chart_finetune
    ```

## Results (CPU smoke-test)

| Setting | Mean Faithfulness |
|---|---|
| Zero-shot baseline | 0.000 |
| LoRA fine-tuned (simulated) | ~0.70 |

!!! warning "These are smoke-test numbers"
    The zero-shot baseline deliberately gives a weak answer (generic text with no chart values) and the "fine-tuned" result scores a lightly paraphrased correct answer through the real evaluation harness. This demonstrates the harness works and shows a plausible before/after gap, not real VLM faithfulness numbers. Run on a GPU with `pip install -e ".[ml]"` for genuine training results.

## Swapping the checkpoint

```yaml
# configs/vlm_chart_finetune.yaml
model:
  checkpoint: "Qwen/Qwen2-VL-7B-Instruct"   # upgrade from 2B default
  checkpoint_pinned_date: "2026-06-01"        # update this when you verify
```

Always update `checkpoint_pinned_date` when changing checkpoints — this records when you verified the checkpoint still resolves and behaves as expected, which matters when you come back to this config six months later.
