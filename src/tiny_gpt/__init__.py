"""Tiny-GPT: from-scratch decoder-only LLM training for learning."""

from .config import Config, ModelConfig, TrainConfig
from .model import TinyGPT

__all__ = ["Config", "ModelConfig", "TrainConfig", "TinyGPT"]
__version__ = "0.1.0"
