# LoRA Adaptation

## Background

Full fine-tuning of a 7B+ parameter VLM requires updating all weights, which is prohibitively expensive in memory (optimizer states alone exceed GPU VRAM for most hardware) and typically unnecessary — a pretrained VLM already has strong general visual and language representations; you're teaching it a task-specific distribution, not building it from scratch.

**LoRA** (Hu et al., 2021[^1]) addresses this by decomposing each weight update $\Delta W \in \mathbb{R}^{d \times k}$ as a product of two low-rank matrices:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$

Only $A$ and $B$ are trained; the original weights $W$ are frozen. The trainable parameter count scales with $r$ (rank), not with the full model size, giving 10-100× reduction in trainable parameters.

## Multimodal LoRA: adapting both towers

Language-only LoRA (adapting only the LM's projection matrices) was the dominant pattern through 2024. By 2025-2026, the convention shifted to adapting both the vision tower's projection layers and the language model — because chart/document QA requires the model to *read off* specific numeric values from the image, not just generate text conditioned on a generic visual embedding. Without adapting the vision tower, the visual representation of a bar chart looks the same to the LM before and after fine-tuning; the visual signal doesn't improve.

In this repo, both towers are adapted by default:

```yaml
lora:
  target_vision_tower: true
  target_language_model: true
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  rank: 16
  alpha: 32
```

## Rank and alpha

`rank` controls the capacity of the adapter (higher = more expressive, more parameters). `alpha` controls the scaling factor applied to the LoRA output: the effective learning rate contribution is `alpha / rank`. The constraint `alpha >= rank` is enforced by the config schema because `alpha < rank` produces an effective scaling less than 1, which typically under-trains the adapter.

Common working values for fine-tuning: `rank=8-32`, `alpha=2×rank`.

## Quantization + LoRA (QLoRA pattern)

The default config uses 4-bit quantization (`quantization: 4bit`) for the frozen base weights, loading them with `BitsAndBytesConfig(load_in_4bit=True)`. LoRA adapters are trained in bf16. This combination (QLoRA, Dettmers et al., 2023[^2]) enables fine-tuning 7B models on a single 12GB GPU card.

## Why this matters for 2027

As frontier VLMs continue to scale, full fine-tuning becomes economically irrational for the vast majority of real deployments — the gap between "a lab with a training cluster" and "a team that needs a model adapted to their chart format by Friday" only widens. Parameter-efficient adaptation stops being an optimization and becomes the *default* interface for customization: expect adapter marketplaces, mixture-of-adapter serving (swapping LoRA weights per-request without reloading the base model), and on-device personalization (an adapter small enough to ship and update over-the-air) to be standard deployment patterns rather than research curiosities. The pattern demonstrated here — adapt both modalities, quantize the frozen base, validate with a real before/after evaluation harness rather than eyeballing loss curves — is the shape that pattern takes in practice, not just the theory of it.

[^1]: Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
[^2]: Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
