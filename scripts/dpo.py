#!/usr/bin/env python3
"""Minimal DPO reference implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from tiny_gpt.checkpoint import load_checkpoint, save_checkpoint
from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.posttrain import dpo_loss
from tiny_gpt.tokenizer import load_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--output",
        default="artifacts/checkpoints/dpo/final.pt",
        help="Where to write the DPO checkpoint",
    )
    args = p.parse_args()

    cfg = Config.from_dict(yaml.safe_load(open(args.config)))
    tok = load_tokenizer(args.tokenizer)
    cfg.model.vocab_size = tok.get_vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = TinyGPT(cfg.model).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    policy.load_state_dict(payload["model"])
    reference = TinyGPT(cfg.model).to(device)
    reference.load_state_dict(policy.state_dict())
    reference.eval()
    for p_ in reference.parameters():
        p_.requires_grad_(False)

    opt = torch.optim.AdamW(policy.parameters(), lr=1e-6, weight_decay=0.0)
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, policy, opt, map_location=device) + 1
        print(f"resumed DPO from {args.resume} at step={start_step}")

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]

    for step in range(start_step, args.steps):
        row = rows[step % len(rows)]
        prompt = torch.tensor([tok.encode(row["prompt"]).ids], device=device)
        chosen = torch.tensor([tok.encode(row["chosen"]).ids], device=device)
        rejected = torch.tensor([tok.encode(row["rejected"]).ids], device=device)
        loss = dpo_loss(policy, reference, prompt, chosen, rejected, beta=args.beta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 10 == 0:
            print(f"dpo_step={step:05d} loss={loss.item():.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(out), policy, opt, args.steps)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
