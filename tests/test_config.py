"""Unit tests for production_vlm.config: ExperimentConfig validation."""

from __future__ import annotations

import pytest

from production_vlm.config import ConfigError, ExperimentConfig


def _valid_raw_config() -> dict:
    return {
        "name": "test_experiment",
        "model": {"checkpoint": "org/model", "checkpoint_pinned_date": "2026-01-01"},
        "lora": {"rank": 16, "alpha": 32},
        "data": {"dataset_name": "synthetic_charts", "dataset_pinned_date": "n/a", "max_samples": 50},
        "train": {"epochs": 1, "learning_rate": 0.0002},
        "eval": {"num_eval_samples": 10},
    }


class TestExperimentConfig:
    def test_valid_config_parses(self):
        cfg = ExperimentConfig.from_dict(_valid_raw_config())
        assert cfg.name == "test_experiment"
        assert cfg.lora.rank == 16
        assert cfg.train.epochs == 1

    def test_missing_required_key_raises_config_error(self):
        raw = _valid_raw_config()
        del raw["model"]
        with pytest.raises(ConfigError):
            ExperimentConfig.from_dict(raw)

    def test_lora_alpha_below_rank_raises(self):
        raw = _valid_raw_config()
        raw["lora"] = {"rank": 64, "alpha": 8}
        with pytest.raises(ConfigError):
            ExperimentConfig.from_dict(raw)

    def test_invalid_dtype_literal_raises(self):
        raw = _valid_raw_config()
        raw["model"]["dtype"] = "int4"
        with pytest.raises(ConfigError):
            ExperimentConfig.from_dict(raw)

    def test_negative_epochs_raises(self):
        raw = _valid_raw_config()
        raw["train"]["epochs"] = -1
        with pytest.raises(ConfigError):
            ExperimentConfig.from_dict(raw)

    def test_defaults_applied_when_optional_sections_omitted(self):
        raw = {
            "name": "minimal",
            "model": {"checkpoint": "org/model", "checkpoint_pinned_date": "2026-01-01"},
            "data": {"dataset_name": "x", "dataset_pinned_date": "n/a"},
        }
        cfg = ExperimentConfig.from_dict(raw)
        assert cfg.lora.rank == 32  # default
        assert cfg.train.seed == 42  # default

    def test_to_dict_roundtrip_preserves_values(self):
        cfg = ExperimentConfig.from_dict(_valid_raw_config())
        as_dict = cfg.to_dict()
        assert as_dict["lora"]["rank"] == 16
        assert as_dict["model"]["checkpoint"] == "org/model"

    def test_empty_checkpoint_raises(self):
        raw = _valid_raw_config()
        raw["model"]["checkpoint"] = ""
        with pytest.raises(ConfigError):
            ExperimentConfig.from_dict(raw)
