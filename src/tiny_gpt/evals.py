"""Simple language-model evaluation helpers."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def evaluate_loss(
    model,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device,
    max_batches: int = 100,
) -> dict[str, float]:
    """Compute mean next-token loss and perplexity."""
    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i, (x, y) in enumerate(batches):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            loss, aux = model(x, targets=y)
            losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    mean_loss = sum(losses) / max(1, len(losses))
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0))}
