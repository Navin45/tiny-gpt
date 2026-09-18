#!/usr/bin/env python3
"""Compact verifier-based RL training example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from tiny_gpt.checkpoint import load_checkpoint, save_checkpoint
from tiny_gpt.config import Config
from tiny_gpt.model import TinyGPT
from tiny_gpt.posttrain import RLVRTrainer
from tiny_gpt.tokenizer import load_tokenizer


def exact_match_verifier(prompt: str, completion: str, expected: str) -> float:
    """Reward 1 for an exact expected answer, else 0."""
    return float(expected.strip() == completion.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--output",
        default="artifacts/checkpoints/rlvr/final.pt",
        help="Where to write the RLVR checkpoint",
    )
    args = p.parse_args()

    cfg = Config.from_dict(yaml.safe_load(open(args.config)))
    tok = load_tokenizer(args.tokenizer)
    cfg.model.vocab_size = tok.get_vocab_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyGPT(cfg.model).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-6)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, opt, map_location=device) + 1
        print(f"resumed RLVR from {args.resume} at step={start_step}")

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    for step in range(start_step, args.steps):
        row = rows[step % len(rows)]
        expected = str(row["answer"])
        verifier = lambda prompt, completion, e=expected: exact_match_verifier(
            prompt, completion, e
        )
        trainer = RLVRTrainer(model, tok, opt, verifier, device, group_size=4)
        mean_reward = trainer.train_prompt(row["prompt"])
        if step % 10 == 0:
            print(f"rlvr_step={step:05d} mean_group_reward={mean_reward:.4f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(str(out), model, opt, args.steps)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
