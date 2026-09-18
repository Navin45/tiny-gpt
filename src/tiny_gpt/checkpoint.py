"""Checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically-ish save the current training state."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "extra": extra or {},
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> int:
    """Load a checkpoint and return the saved step."""
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("step", 0))
