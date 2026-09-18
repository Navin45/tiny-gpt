"""Configuration dataclasses for the Tiny-GPT reference stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Hyperparameters that define the Transformer architecture."""

    vocab_size: int = 32_000
    max_seq_len: int = 2_048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    rope_theta: float = 500_000.0
    ffn_multiplier: float = 3.5
    moe_layers_every: int = 2
    n_experts: int = 8
    moe_top_k: int = 2
    moe_aux_loss_coef: float = 0.01
    dropout: float = 0.0
    use_checkpointing: bool = False
    # When True, packed pretraining sequences do not attend across <|eos|> boundaries.
    mask_document_boundaries: bool = False
    # Used with mask_document_boundaries; typically set from the tokenizer at launch.
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA")
        if not (1 <= self.moe_top_k <= self.n_experts):
            raise ValueError("moe_top_k must be between 1 and n_experts")


@dataclass
class TrainConfig:
    """Optimization and runtime settings for base pretraining."""

    batch_size: int = 2
    grad_accum_steps: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 10_000
    min_lr_ratio: float = 0.1
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 500
    seed: int = 1337
    token_bin: str = "artifacts/data/train.bin"
    val_token_bin: str = "artifacts/data/val.bin"
    checkpoint_dir: str = "artifacts/checkpoints/base"


@dataclass
class Config:
    """Top-level configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        """Create the nested dataclasses from a YAML dictionary."""
        model_fields = ModelConfig.__dataclass_fields__
        train_fields = TrainConfig.__dataclass_fields__
        model_data = {k: v for k, v in raw.items() if k in model_fields}
        train_data = {k: v for k, v in raw.items() if k in train_fields}
        if "betas" in train_data:
            train_data["betas"] = tuple(train_data["betas"])
        return cls(ModelConfig(**model_data), TrainConfig(**train_data))
