"""Unit tests for production_vlm.utils.kv_cache: KV-cache memory-efficient decoding analysis."""

from __future__ import annotations

import pytest

from production_vlm.utils.kv_cache import (
    AttentionStrategy,
    ModelDecoderConfig,
    compare_strategies,
    compute_kv_cache_memory,
    visual_token_count,
)


class TestVisualTokenCount:
    def test_standard_336px_siglip_convention(self):
        # 336 / 14 = 24 -> 24*24 = 576, the well-known LLaVA/SigLIP visual token count
        assert visual_token_count(336, patch_size=14) == 576

    def test_scales_quadratically_with_image_size(self):
        small = visual_token_count(224, patch_size=14)
        large = visual_token_count(448, patch_size=14)
        assert large == small * 4  # doubling side length -> 4x tokens


class TestComputeKVCacheMemory:
    def test_mha_is_baseline_relative_to_mha_is_one(self):
        cfg = ModelDecoderConfig()
        result = compute_kv_cache_memory(cfg, AttentionStrategy.MHA, seq_len=576)
        assert result.relative_to_mha == pytest.approx(1.0)

    def test_gqa_reduces_memory_by_head_ratio(self):
        cfg = ModelDecoderConfig(n_query_heads=28, n_kv_heads_gqa=4)
        result = compute_kv_cache_memory(cfg, AttentionStrategy.GQA, seq_len=576)
        expected_ratio = 4 / 28
        assert result.relative_to_mha == pytest.approx(expected_ratio, rel=1e-6)

    def test_mqa_uses_single_kv_head(self):
        cfg = ModelDecoderConfig(n_query_heads=28)
        result = compute_kv_cache_memory(cfg, AttentionStrategy.MQA, seq_len=576)
        expected_ratio = 1 / 28
        assert result.relative_to_mha == pytest.approx(expected_ratio, rel=1e-6)

    def test_sliding_window_matches_mha_below_window_size(self):
        cfg = ModelDecoderConfig(sliding_window_size=512)
        result = compute_kv_cache_memory(cfg, AttentionStrategy.SLIDING_WINDOW, seq_len=400)
        assert result.relative_to_mha == pytest.approx(1.0)

    def test_sliding_window_caps_memory_beyond_window_size(self):
        cfg = ModelDecoderConfig(sliding_window_size=512)
        at_window = compute_kv_cache_memory(cfg, AttentionStrategy.SLIDING_WINDOW, seq_len=512)
        beyond_window = compute_kv_cache_memory(cfg, AttentionStrategy.SLIDING_WINDOW, seq_len=2000)
        # Absolute memory must be identical (capped), even though seq_len differs 4x
        assert at_window.kv_cache_mb == pytest.approx(beyond_window.kv_cache_mb)

    def test_sliding_window_relative_savings_improve_with_longer_sequences(self):
        cfg = ModelDecoderConfig(sliding_window_size=512)
        short = compute_kv_cache_memory(cfg, AttentionStrategy.SLIDING_WINDOW, seq_len=600)
        long = compute_kv_cache_memory(cfg, AttentionStrategy.SLIDING_WINDOW, seq_len=6000)
        # As true sequence length grows past the window, relative memory (vs MHA) should shrink
        assert long.relative_to_mha < short.relative_to_mha

    def test_memory_scales_linearly_with_batch_size(self):
        cfg = ModelDecoderConfig()
        b1 = compute_kv_cache_memory(cfg, AttentionStrategy.MHA, seq_len=576, batch_size=1)
        b4 = compute_kv_cache_memory(cfg, AttentionStrategy.MHA, seq_len=576, batch_size=4)
        assert b4.kv_cache_mb == pytest.approx(b1.kv_cache_mb * 4)

    def test_memory_scales_linearly_with_seq_len_for_mha(self):
        cfg = ModelDecoderConfig()
        short = compute_kv_cache_memory(cfg, AttentionStrategy.MHA, seq_len=500)
        long = compute_kv_cache_memory(cfg, AttentionStrategy.MHA, seq_len=1000)
        assert long.kv_cache_mb == pytest.approx(short.kv_cache_mb * 2)

    def test_ordering_mha_gt_gqa_gt_mqa(self):
        """Memory ordering must hold: MHA >= GQA >= MQA at any fixed sequence length."""
        cfg = ModelDecoderConfig()
        mha = compute_kv_cache_memory(cfg, AttentionStrategy.MHA, seq_len=1000)
        gqa = compute_kv_cache_memory(cfg, AttentionStrategy.GQA, seq_len=1000)
        mqa = compute_kv_cache_memory(cfg, AttentionStrategy.MQA, seq_len=1000)
        assert mha.kv_cache_mb > gqa.kv_cache_mb > mqa.kv_cache_mb


class TestCompareStrategies:
    def test_returns_all_four_strategies(self):
        cfg = ModelDecoderConfig()
        results = compare_strategies(cfg, seq_lens=[500, 1000])
        assert set(results.keys()) == {"mha", "gqa", "mqa", "sliding_window"}

    def test_each_strategy_has_one_result_per_seq_len(self):
        cfg = ModelDecoderConfig()
        seq_lens = [400, 800, 1200]
        results = compare_strategies(cfg, seq_lens=seq_lens)
        for strategy_results in results.values():
            assert len(strategy_results) == len(seq_lens)
            assert [r.seq_len for r in strategy_results] == seq_lens
