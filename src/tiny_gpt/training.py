"""Distributed training utilities shared by the scripts."""

from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    """Initialize torchrun/DDP if launched with multiple processes."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP reference expects CUDA when WORLD_SIZE > 1")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    """Tear down the process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """Move the model to the device and wrap it with DDP when needed."""
    model.to(device)
    if dist.is_initialized():
        return DDP(model, device_ids=[device.index])
    return model


def cosine_lr(step: int, *, base_lr: float, warmup_steps: int, max_steps: int, min_lr_ratio: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Set the learning rate for all parameter groups."""
    for group in optimizer.param_groups:
        group["lr"] = lr
