"""Config schemas shared across all examples.

Implemented with stdlib ``dataclasses`` so the core package has zero
hard third-party dependencies beyond what's already needed for the
ML stack itself (numpy/scipy/pyyaml). This keeps `cv_playbook` usable
in minimal/CI/CPU-only environments without pulling in a config
framework just to validate a YAML file.

Examples load YAML, construct these dataclasses via
:meth:`ExperimentConfig.from_dict`, and get fail-fast validation in
``__post_init__`` instead of an error deep inside a training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal, get_args, get_origin


class ConfigError(ValueError):
    """Raised when an experiment config fails validation."""


def _check_literal(value: Any, type_hint: Any, name: str) -> None:
    if get_origin(type_hint) is Literal:
        allowed = get_args(type_hint)
        if value not in allowed:
            raise ConfigError(f"{name}={value!r} not in allowed values {allowed}")


@dataclass
class ModelConfig:
    checkpoint: str
    checkpoint_pinned_date: str
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    quantization: Literal["none", "4bit", "8bit"] = "none"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        _check_literal(self.dtype, self.__annotations__["dtype"], "dtype")
        _check_literal(self.quantization, self.__annotations__["quantization"], "quantization")
        if not self.checkpoint:
            raise ConfigError("model.checkpoint must not be empty")


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_vision_tower: bool = True
    target_language_model: bool = True
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])

    def __post_init__(self) -> None:
        if not (1 <= self.rank <= 256):
            raise ConfigError(f"lora.rank={self.rank} must be in [1, 256]")
        if self.alpha < 1:
            raise ConfigError(f"lora.alpha={self.alpha} must be >= 1")
        if not (0.0 <= self.dropout <= 1.0):
            raise ConfigError(f"lora.dropout={self.dropout} must be in [0, 1]")
        if self.alpha < self.rank:
            raise ConfigError(f"lora.alpha ({self.alpha}) should generally be >= lora.rank ({self.rank})")


@dataclass
class DataConfig:
    dataset_name: str
    dataset_pinned_date: str
    train_split: str = "train"
    eval_split: str = "validation"
    max_samples: int | None = None
    synthetic_chart_augmentation: bool = False
    image_max_side: int = 1024

    def __post_init__(self) -> None:
        if self.max_samples is not None and self.max_samples <= 0:
            raise ConfigError("data.max_samples must be positive if set")
        if self.image_max_side <= 0:
            raise ConfigError("data.image_max_side must be positive")


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 1
    batch_size: int = 2
    grad_accum_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_train_minutes: int | None = 30
    logging: Literal["none", "local", "wandb"] = "local"
    output_dir: str = "outputs/vlm_chart_finetune"

    def __post_init__(self) -> None:
        _check_literal(self.logging, self.__annotations__["logging"], "logging")
        if self.epochs <= 0:
            raise ConfigError("train.epochs must be positive")
        if self.batch_size <= 0:
            raise ConfigError("train.batch_size must be positive")
        if self.learning_rate <= 0:
            raise ConfigError("train.learning_rate must be positive")


@dataclass
class EvalConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["numeric_accuracy", "grounding_score", "faithfulness_score"]
    )
    num_eval_samples: int = 50

    def __post_init__(self) -> None:
        if self.num_eval_samples <= 0:
            raise ConfigError("eval.num_eval_samples must be positive")


@dataclass
class ExperimentConfig:
    name: str
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig

    @classmethod
    def from_dict(cls, raw: dict) -> ExperimentConfig:
        try:
            return cls(
                name=raw["name"],
                model=ModelConfig(**raw["model"]),
                lora=LoRAConfig(**raw.get("lora", {})),
                data=DataConfig(**raw["data"]),
                train=TrainConfig(**raw.get("train", {})),
                eval=EvalConfig(**raw.get("eval", {})),
            )
        except KeyError as e:
            raise ConfigError(f"Missing required config key: {e}") from e
        except TypeError as e:
            raise ConfigError(f"Invalid config field: {e}") from e

    def to_dict(self) -> dict:
        def _dc_to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {f.name: _dc_to_dict(getattr(obj, f.name)) for f in fields(obj)}
            return obj

        return _dc_to_dict(self)
