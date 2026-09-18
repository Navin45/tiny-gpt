#!/usr/bin/env python3
"""Base-model pretraining entry point."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from tiny_gpt.checkpoint import load_checkpoint, save_checkpoint
from tiny_gpt.config import Config
from tiny_gpt.data import TokenDataset
from tiny_gpt.model import TinyGPT
from tiny_gpt.tokenizer import load_tokenizer
from tiny_gpt.training import (
    cleanup_distributed,
    cosine_lr,
    seed_everything,
    set_optimizer_lr,
    setup_distributed,
    wrap_ddp,
)


def evaluate(model, loader, device, max_batches: int = 20) -> float:
    """Estimate validation cross-entropy over a small fixed number of batches."""
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                loss, aux = model(x, targets=y)
                total = (
                    loss + model.module.cfg.moe_aux_loss_coef * aux
                    if hasattr(model, "module")
                    else loss + model.cfg.moe_aux_loss_coef * aux
                )
            losses.append(total.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume model/optimizer state from",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer.json; required when mask_document_boundaries is enabled",
    )
    args = parser.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    cfg = Config.from_dict(raw)
    rank, world_size, _, device = setup_distributed()
    seed_everything(cfg.train.seed + rank)
    torch.set_float32_matmul_precision("high")

    if cfg.model.mask_document_boundaries:
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required when mask_document_boundaries is true")
        tok = load_tokenizer(args.tokenizer)
        eos_id = tok.token_to_id("<|eos|>")
        if eos_id is None:
            raise SystemExit("Tokenizer is missing <|eos|>")
        cfg.model.eos_token_id = eos_id

    train_ds = TokenDataset(
        cfg.train.token_bin, cfg.model.max_seq_len, seed=cfg.train.seed + 17 * rank
    )
    val_ds = TokenDataset(
        cfg.train.val_token_bin,
        cfg.model.max_seq_len,
        seed=cfg.train.seed + 101 + 17 * rank,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=0
    )
    train_iter = iter(train_loader)
    epoch = 0

    model = TinyGPT(cfg.model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, map_location=device) + 1
        if rank == 0:
            print(f"resumed from {args.resume} at step={start_step}")

    model = wrap_ddp(model, device)

    autocast_enabled = device.type == "cuda"
    running_loss = 0.0
    is_ddp = hasattr(model, "no_sync")

    for step in range(start_step, cfg.train.max_steps):
        optimizer.zero_grad(set_to_none=True)
        lr = cosine_lr(
            step,
            base_lr=cfg.train.learning_rate,
            warmup_steps=cfg.train.warmup_steps,
            max_steps=cfg.train.max_steps,
            min_lr_ratio=cfg.train.min_lr_ratio,
        )
        set_optimizer_lr(optimizer, lr)

        for micro in range(cfg.train.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                epoch += 1
                train_ds.set_epoch(epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            sync_ctx = (
                model.no_sync()
                if is_ddp and micro < cfg.train.grad_accum_steps - 1
                else nullcontext()
            )
            with sync_ctx:
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
                ):
                    ce_loss, aux_loss = model(x, targets=y)
                    base_cfg = model.module.cfg if hasattr(model, "module") else model.cfg
                    loss = (
                        ce_loss + base_cfg.moe_aux_loss_coef * aux_loss
                    ) / cfg.train.grad_accum_steps
                loss.backward()
            running_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optimizer.step()

        if step % cfg.train.log_every == 0 and rank == 0:
            avg = running_loss / cfg.train.log_every
            running_loss = 0.0
            print(f"step={step:06d} lr={lr:.3e} train_loss={avg:.4f} epoch={epoch}")

        if step > 0 and step % cfg.train.eval_every == 0 and rank == 0:
            val = evaluate(model, val_loader, device)
            print(
                f"step={step:06d} val_loss={val:.4f} "
                f"ppl={float(torch.exp(torch.tensor(val))):.2f}"
            )

        if step > 0 and step % cfg.train.save_every == 0 and rank == 0:
            raw_model = model.module if hasattr(model, "module") else model
            out = Path(cfg.train.checkpoint_dir) / f"step_{step}.pt"
            save_checkpoint(str(out), raw_model, optimizer, step)
            print(f"saved {out}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
