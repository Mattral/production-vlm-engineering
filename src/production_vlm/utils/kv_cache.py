"""KV-cache memory analysis for efficient VLM decoding.

Implements the "memory-efficient decoding / attention optimizations"
requirement of P0-03 in the roadmap, distinct from (and complementary
to) the vision-encoder ONNX/quantization work in the rest of this
example: this module addresses the **language-model decoder's**
autoregressive generation step, where KV-cache memory -- not FLOPs --
is usually the binding constraint for long-context VLM inference
(a modern VLM commonly encodes 500-1500+ visual tokens per image before
a single word of text is generated, so the KV-cache for those visual
tokens dominates memory for the entire generation).

Four attention/cache strategies are compared, all via closed-form
memory arithmetic (no model weights needed, so this runs identically
on any machine):

1. **MHA** (Multi-Head Attention) -- the original Transformer
   attention (Vaswani et al., 2017): one K/V pair per query head.
   Baseline; largest KV-cache footprint.
2. **GQA** (Grouped-Query Attention, Ainslie et al., 2023) -- query
   heads are grouped and share K/V heads, cutting KV-cache size by
   the group factor with minimal quality loss. Used in Llama 3,
   Qwen2-VL, and most 2025-2026 open VLMs.
3. **MQA** (Multi-Query Attention, Shazeer, 2019) -- the extreme case
   of GQA: a single shared K/V head for all query heads. Maximum
   memory savings, some quality tradeoff.
4. **Sliding-window KV-cache** (Beltagy et al., 2020 Longformer-style;
   also used in Mistral) -- bounds the cache to the most recent W
   tokens regardless of sequence length, giving O(1) memory in
   sequence length rather than O(n). The right strategy when only
   local context matters for the generation step (common for chart/
   document QA, where the answer depends on recently-attended regions).

Reference for the broader efficient-decoding landscape this connects
to: FlashAttention-2 (Dao, 2023) for compute/memory-efficient exact
attention (orthogonal to cache *strategy*, applies to all four above),
PagedAttention (Kwon et al., 2023, the vLLM paper) for efficient
*memory management* of whichever cache strategy is chosen, and
speculative decoding (Leviathan et al., 2023; Chen et al., 2023) for
reducing the number of forward passes needed per generated token --
a complementary axis to cache-size reduction (see FlashSpec-style
speculative decoding for that angle; this module focuses purely on
per-token memory footprint).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AttentionStrategy(str, Enum):
    MHA = "mha"
    GQA = "gqa"
    MQA = "mqa"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class ModelDecoderConfig:
    """Decoder architecture parameters needed for KV-cache memory arithmetic.

    Defaults approximate a 7B-class VLM's language decoder (matching
    the Qwen2-VL-7B-class checkpoint referenced elsewhere in this repo)
    -- adjust to your actual checkpoint's config.json for real numbers.
    """

    n_layers: int = 28
    n_query_heads: int = 28
    head_dim: int = 128
    n_kv_heads_gqa: int = 4  # typical GQA group size for 7B-class models (e.g. Llama 3 8B uses 8)
    bytes_per_param: int = 2  # bf16/fp16
    sliding_window_size: int = 512  # tokens of local context retained


@dataclass
class KVCacheMemoryResult:
    strategy: str
    seq_len: int
    batch_size: int
    kv_cache_bytes: int
    kv_cache_mb: float
    relative_to_mha: float  # 1.0 = same as MHA, 0.25 = 4x smaller, etc.


def compute_kv_cache_memory(
    cfg: ModelDecoderConfig,
    strategy: AttentionStrategy,
    seq_len: int,
    batch_size: int = 1,
) -> KVCacheMemoryResult:
    """Closed-form KV-cache memory footprint for one strategy at one sequence length.

    Standard KV-cache memory formula:
        bytes = 2 (K and V) x n_layers x n_kv_heads x head_dim x seq_len x batch_size x bytes_per_param

    The four strategies differ only in `n_kv_heads` (and, for sliding
    window, in the effective `seq_len` once the cache is full):
        MHA:            n_kv_heads = n_query_heads
        GQA:            n_kv_heads = n_kv_heads_gqa  (< n_query_heads)
        MQA:            n_kv_heads = 1
        Sliding window: n_kv_heads = n_query_heads, but seq_len is capped
                        at sliding_window_size once the true sequence
                        exceeds it (older tokens are evicted from cache)
    """
    effective_seq_len = seq_len
    if strategy == AttentionStrategy.MHA:
        n_kv_heads = cfg.n_query_heads
    elif strategy == AttentionStrategy.GQA:
        n_kv_heads = cfg.n_kv_heads_gqa
    elif strategy == AttentionStrategy.MQA:
        n_kv_heads = 1
    elif strategy == AttentionStrategy.SLIDING_WINDOW:
        n_kv_heads = cfg.n_query_heads
        effective_seq_len = min(seq_len, cfg.sliding_window_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    kv_bytes = 2 * cfg.n_layers * n_kv_heads * cfg.head_dim * effective_seq_len * batch_size * cfg.bytes_per_param

    # Reference: full MHA at the same (uncapped) seq_len, for the relative-savings column.
    mha_bytes = 2 * cfg.n_layers * cfg.n_query_heads * cfg.head_dim * seq_len * batch_size * cfg.bytes_per_param

    return KVCacheMemoryResult(
        strategy=strategy.value,
        seq_len=seq_len,
        batch_size=batch_size,
        kv_cache_bytes=kv_bytes,
        kv_cache_mb=kv_bytes / (1024 * 1024),
        relative_to_mha=kv_bytes / mha_bytes,
    )


def compare_strategies(
    cfg: ModelDecoderConfig,
    seq_lens: list[int],
    batch_size: int = 1,
) -> dict[str, list[KVCacheMemoryResult]]:
    """Run all four strategies across a list of sequence lengths (e.g. growing visual-token counts)."""
    results: dict[str, list[KVCacheMemoryResult]] = {s.value: [] for s in AttentionStrategy}
    for strategy in AttentionStrategy:
        for seq_len in seq_lens:
            results[strategy.value].append(compute_kv_cache_memory(cfg, strategy, seq_len, batch_size))
    return results


def visual_token_count(image_size: int, patch_size: int = 14) -> int:
    """Number of visual tokens a ViT-style vision tower produces for a square image.

    E.g. a 336x336 image with patch_size=14 (SigLIP/CLIP convention)
    produces (336/14)^2 = 576 visual tokens -- these all enter the LM
    decoder's KV-cache before a single output token is generated,
    which is why KV-cache memory is dominated by the *visual* prefix
    for typical chart/document VQA prompts (a few dozen text tokens
    for the question, versus hundreds of visual tokens for the image).
    """
    patches_per_side = image_size // patch_size
    return patches_per_side * patches_per_side
